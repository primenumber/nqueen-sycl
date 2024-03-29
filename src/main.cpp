#include <chrono>
#include <iostream>
#include <random>

#include <sycl/sycl.hpp>

using namespace sycl;

struct NQueenTasks {
  std::vector<uint32_t> left, middle, right;
  void push(uint32_t lbits, uint32_t mbits, uint32_t rbits) {
    left.push_back(lbits);
    middle.push_back(mbits);
    right.push_back(rbits);
  }
  size_t size() const { return std::size(left); }
};


void expand_impl(size_t n, size_t expansion, size_t depth, uint32_t left, uint32_t middle, uint32_t right,
    NQueenTasks& tasks) {
  uint32_t mask = (1 << n) - 1;
  if (depth == expansion) {
    tasks.push(left, middle, right);
    return;
  }
  auto remains = mask & ~(left | middle | right);
  while (remains) {
    auto bit = remains & -remains;
    auto next_l = (left | bit) << 1;
    auto next_m = middle | bit;
    auto next_r = (right | bit) >> 1;
    expand_impl(n, expansion, depth+1, next_l, next_m, next_r, tasks);
    remains ^= bit;
  }
}

NQueenTasks expand(size_t n, size_t expansion) {
  NQueenTasks tasks;
  
  expand_impl(n, expansion, 0, 0, 0, 0, tasks);
  return tasks;
}

int main(int argc, char** argv) {
  using std::size;
  using clk_t = std::chrono::high_resolution_clock;
  if (argc < 3) {
    std::cerr << "Usage: nqueen N expansion" << std::endl;
    exit(EXIT_FAILURE);
  }
  const size_t n = atol(argv[1]);
  const size_t expansion = atol(argv[2]);

  std::cerr << n << " " << expansion << std::endl;

  // create a kernel
  queue q;

  const auto start = clk_t::now();
  std::cerr << "Start expansion" << std::endl;
  const auto tasks = expand(n, expansion);
  const auto task_count = size(tasks);
  std::cerr << "Finished expansion, count=" << task_count << std::endl;

  // USM allocation using malloc_shared
  auto v_left = malloc_device<uint32_t>(task_count, q);
  auto v_middle = malloc_device<uint32_t>(task_count, q);
  auto v_right = malloc_device<uint32_t>(task_count, q);
  const size_t k = 1 << 16;
  auto result = malloc_device<size_t>(k, q);

  q.submit([&](handler& h) {
      h.memcpy(v_left, tasks.left.data(), task_count * sizeof(uint32_t));
  });
  q.submit([&](handler& h) {
      h.memcpy(v_middle, tasks.middle.data(), task_count * sizeof(uint32_t));
  });
  q.submit([&](handler& h) {
      h.memcpy(v_right, tasks.right.data(), task_count * sizeof(uint32_t));
  });
  q.wait();

  std::cerr << "Start" << std::endl;
  auto e = q.submit([&](handler& h) {
    h.parallel_for(k, [=](auto& idx) {
        constexpr size_t stack_depth = 24;
        const size_t remain_depth = n - expansion;
        const uint32_t mask = (1 << n) - 1;
        uint32_t stack_remains[stack_depth];
        size_t answer = 0;
        for (size_t i = idx; i < task_count; i += k) {
          uint32_t left = v_left[i];
          uint32_t middle = v_middle[i];
          uint32_t right = v_right[i];
          uint32_t right_under = 0;
          uint32_t left_over = 0;
          ptrdiff_t depth = 0;
          stack_remains[depth] = mask & ~(left | middle | right);
          auto stack_pop = [&] {
            --depth;
            uint32_t& remains = stack_remains[depth];
            uint32_t bit = remains & -remains;
            left >>= 1;
            left |= left_over << 31;
            left_over >>= 1;
            right <<= 1;
            right |= right_under >> 31;
            right_under <<= 1;
            left ^= bit;
            middle ^= bit;
            right ^= bit;
            remains ^= bit;
          };
          auto stack_push = [&] {
            uint32_t remains = stack_remains[depth];
            uint32_t bit = remains & -remains;
            left ^= bit;
            middle ^= bit;
            right ^= bit;
            left_over <<= 1;
            left_over |= left >> 31;
            left <<= 1;
            right_under >>= 1;
            right_under |= right << 31;
            right >>= 1;
            ++depth;
            stack_remains[depth] = mask & ~(left | middle | right);
          };
          while (depth >= 0) {
            uint32_t remains = stack_remains[depth];
            if (remains == 0) {
              if (depth >= remain_depth) {
                ++answer;
              }
              stack_pop();
              continue;
            }
            stack_push();
          }
        }
        result[idx] = answer;
    });
  });
  e.wait();

  std::vector<size_t> result_h(k);
  q.submit([&](handler& h) {
      h.memcpy(result_h.data(), result, k * sizeof(size_t));
  }).wait();
  size_t sum = 0;
  for (auto&& e : result_h) sum += e;
  std::cerr << sum << std::endl;
  const auto finish = clk_t::now();
  std::cerr << "Elapsed: " << std::chrono::duration<double, std::milli>(finish - start).count() << "ms" << std::endl;

  free(v_left);
  free(v_middle);
  free(v_right);
  free(result);

  return 0;
}
