#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#include <iostream>
#include <vector>

int main() {
  try {
    // Query available platforms.
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (const auto& platform: platforms) {
      std::cout << "CL_PLATFORM_NAME: " << platform.getInfo<CL_PLATFORM_NAME>() << "\n";
      std::cout << "CL_PLATFORM_VENDOR: " << platform.getInfo<CL_PLATFORM_VENDOR>() << "\n";
      std::cout << "CL_PLATFORM_VERSION: " << platform.getInfo<CL_PLATFORM_VERSION>() << "\n";
    }

    // Choose the first available platform.
    auto platform = platforms.front();

    // Query available devices in platform.
    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    for (const auto& device: devices) {
      std::cout << "CL_DEVICE_NAME: " << device.getInfo<CL_DEVICE_NAME>() << "\n";
      std::cout << "CL_DEVICE_VENDOR: " << device.getInfo<CL_DEVICE_VENDOR>() << "\n";
      std::cout << "CL_DEVICE_VERSION: " << device.getInfo<CL_DEVICE_VERSION>() << "\n";
      std::cout << "CL_DEVICE_PROFILE: " << device.getInfo<CL_DEVICE_PROFILE>() << "\n";
    }

    // Choose the first available device in platform.
    auto device = devices.front();

    // Create a context with device.
    cl::Context context{device};
    // Create a command queue with context and device.
    cl::CommandQueue command_queue{context, device};

    // Use raw string store kernel source file.
    std::string kernel_str{R"(
kernel void vector_add(global const float *a, global const float *b, global float *c) {
  int i = get_global_id(0);
  c[i] = a[i] + b[i];
}
    )"};

    // Create a program with kernel source.
    std::vector<std::string> program_strings{kernel_str};
    cl::Program vector_add_program{context, program_strings};
    // Try to build kernel.
    try {
      vector_add_program.build("-cl-std=CL3.0");
    } catch (...) {
      // Print build log if build failed (exception thrown).
      cl_int build_err = CL_SUCCESS;
      auto build_info = vector_add_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&build_err);
      for (auto& pair: build_info) {
        std::cerr << pair.second << "\n";
      }
      return 1;
    }

    // Create host vectors.
    constexpr int vector_size = 1024 * 1024 * 64;
    std::vector<cl_float> h_a(vector_size, 1.0f);
    std::vector<cl_float> h_b(vector_size, 2.0f);
    std::vector<cl_float> h_c(vector_size, 0.0f);

    // Create device buffers.
    cl::Buffer d_a{context, h_a.begin(), h_a.end(), true};
    cl::Buffer d_b{context, h_b.begin(), h_b.end(), true};
    cl::Buffer d_c{context, CL_MEM_WRITE_ONLY, sizeof(float) * vector_size};

    // Create vector add kernel object (represented by cl::KernelFunctor).
    auto vector_add_kernel = cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer>{vector_add_program, "vector_add"};

    // Start computation.
    auto args = cl::EnqueueArgs(command_queue, cl::NDRange{1024}, cl::NDRange{64});
    vector_add_kernel(args, d_a, d_b, d_c);

    // Copy result from device buffer to host vector.
    cl::copy(command_queue, d_c, h_c.begin(), h_c.end());

    // Check calculation results.
    std::cout << "Result: " << h_c.front() << "\n";
  } catch (const cl::Error& err) {
    std::cerr << "CL ERROR: " << err.what() << "\n";
    return 1;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  } catch (...) {
    throw;
  }

  return 0;
}
