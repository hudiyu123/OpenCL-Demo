#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_TARGET_OPENCL_VERSION 300
#include <CL/opencl.hpp>

#include <iostream>
#include <vector>

std::string kernel{R"(
kernel void vector_add(global const float *a, global const float *b, global float *c) {
    int i = get_global_id(0);
    c[i] = a[i] + b[i];
}
)"};

int main() {
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    for (const auto& platform : platforms) {
        auto name = platform.getInfo<CL_PLATFORM_NAME>();
        auto vendor = platform.getInfo<CL_PLATFORM_VENDOR>();
        auto version = platform.getInfo<CL_PLATFORM_VERSION>();

        std::cout << name << "\n";
        std::cout << vendor << "\n";
        std::cout << version << "\n";
    }

    auto platform = platforms.front();

    std::vector<cl::Device> devices;
    platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

    for (const auto& device : devices) {
        auto name = device.getInfo<CL_DEVICE_NAME>();
        auto vendor = device.getInfo<CL_DEVICE_VENDOR>();
        auto version = device.getInfo<CL_DEVICE_VERSION>();
        auto profile = device.getInfo<CL_DEVICE_PROFILE>();

        std::cout << name << "\n";
        std::cout << vendor << "\n";
        std::cout << version << "\n";
        std::cout << profile << "\n";
    }

    auto device = devices.front();

    cl::Context context{device};

    cl::CommandQueue command_queue{context, device};

    std::vector<std::string> program_strings{kernel};

    cl::Program vector_add_program{context, program_strings};
    try {
        vector_add_program.build("-cl-std=CL3.0");
    } catch (...) {
        cl_int build_err = CL_SUCCESS;
        auto build_info = vector_add_program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(&build_err);
        for (auto &pair : build_info) {
            std::cerr << pair.second << "\n";
        }
        return 1;
    }

    constexpr int vector_size = 1024 * 1024 * 64;

    std::vector<float> h_a(vector_size, 1.0f);
    std::vector<float> h_b(vector_size, 2.0f);
    std::vector<float> h_c(vector_size, 0.0f);

    cl::Buffer d_a{context, h_a.begin(), h_a.end(), true};
    cl::Buffer d_b{context, h_b.begin(), h_b.end(), true};
    cl::Buffer d_c{context, CL_MEM_WRITE_ONLY, sizeof(float) * vector_size};

    cl::KernelFunctor<cl::Buffer, cl::Buffer, cl::Buffer> vector_add_kernel{vector_add_program, "vector_add"};

    auto args = cl::EnqueueArgs(command_queue, cl::NDRange{1024});
    vector_add_kernel(args, d_a, d_b, d_c);
    cl::copy(command_queue, d_c, h_c.begin(), h_c.end());

    std::cout << h_c.front() << "\n";

    return 0;
}
