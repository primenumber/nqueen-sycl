TARGET   = nqueen
DEVICE_INFO = device_info
SRCS     = src/main.cpp
DI_SRCS     = src/device_info.cpp

CXX      = icpx
CXXFLAGS = -fsycl -I. -std=c++17
LDFLAGS  =

ifeq ($(BUILD_MODE), release)
    CXXFLAGS += -O3
else
    CXXFLAGS += -Og -g
endif

.PHONY: all build run clean
.DEFAULT_GOAL := all

# the same as build target
all: build

# build the project
build: $(TARGET) $(DEVICE_INFO)

$(TARGET): $(SRCS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

$(DEVICE_INFO): $(DI_SRCS)
	$(CXX) -o $@ $^ $(CXXFLAGS) $(LDFLAGS)

# run binary
run: $(TARGET)
	./$(TARGET)

# clean all
clean:
	-$(RM) $(TARGET)
