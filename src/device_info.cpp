#include <chrono>
#include <iostream>
#include <random>

#include <sycl/sycl.hpp>

using namespace sycl;

void print_device_info(const device& d) {
  std::cout << "Vendor: " << d.get_info<info::device::vendor>() << std::endl;
  std::cout << "Name: " << d.get_info<info::device::name>() << std::endl;
  std::cout << "Max compute units: "  << d.get_info<info::device::max_compute_units>() << std::endl;
  std::cout << "Global Mem size: "  << d.get_info<info::device::global_mem_size>() << std::endl;
}

int main(int argc, char** argv) {
  device d;
  print_device_info(d);
  return 0;
}

