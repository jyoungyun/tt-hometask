#include <vector>
#include <cmath>
#include <random>
#include <cstddef>
#include <algorithm>
#include <iostream>
#include <iomanip>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt;
using namespace tt::tt_metal;

constexpr uint32_t kRows = 32;
// TODO: kCols = 1024
constexpr uint32_t kCols = 32;

std::vector<bfloat16> make_input() {
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    std::vector<bfloat16> input(kRows * kCols);
    for (bfloat16& v : input) {
        v = bfloat16(dist(rng));
    }
    return input;
}

std::vector<bfloat16> exp_reference(const std::vector<bfloat16>& input) {
    std::vector<bfloat16> output(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        output[i] = bfloat16(std::exp(static_cast<float>(input[i])));
    }
    return output;
}

std::vector<bfloat16> softmax_reference(const std::vector<bfloat16>& input) {
    std::vector<float> foutput(input.size());
    std::vector<bfloat16> output(input.size());

    for (size_t row = 0; row < kRows; ++row) {
        int base = row * kCols;

        float max_val = static_cast<float>(input[base]);
        for (size_t col = 1; col < kCols; ++col) {
            max_val = std::max(max_val, static_cast<float>(input[base+col]));
        }

        float sum = 0.f;
        for (size_t col = 0; col < kCols; ++col) {
            float e = std::exp(static_cast<float>(input[base+col]) - max_val);
            foutput[base + col] = e;
            sum += e;
        }

        for (size_t col = 0; col < kCols; ++col) {
            output[base + col] = bfloat16(foutput[base + col] / sum);
        }
    }

    return output;
}

float max_abs_error(const std::vector<bfloat16>& a, const std::vector<bfloat16>& b) {
    if (a.size() != b.size()) {
        throw std::runtime_error("Size Mismatch!");
    }

    float max_err = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        float av = static_cast<float>(a[i]);
        float bv = static_cast<float>(b[i]);
        max_err = std::max(max_err, std::abs(av - bv));
    }
    return max_err;
}

int main() {
    // create mesh device and command queue
    auto mesh_device = distributed::MeshDevice::create_unit_mesh(0);
    auto& cq = mesh_device->mesh_command_queue();

    // create workload
    distributed::MeshWorkload workload;
    distributed::MeshCoordinateRange device_range = distributed::MeshCoordinateRange(mesh_device->shape());

    // create program
    Program program = CreateProgram();

    constexpr CoreCoord core = {0,0};
    // TODO: dual core
    const uint32_t n_tiles = 1;
    const uint32_t tile_size_bytes = sizeof(bfloat16) * kRows * 32;

    // create dram buffer
    distributed::DeviceLocalBufferConfig dram_config{
        .page_size = tile_size_bytes,
        .buffer_type = tt::tt_metal::BufferType::DRAM
    };
    distributed::ReplicatedBufferConfig max_buffer_config{
        .size = n_tiles * tile_size_bytes
    };
    distributed::ReplicatedBufferConfig sum_buffer_config{
        .size = n_tiles * tile_size_bytes
    };
    distributed::ReplicatedBufferConfig buffer_config{
        .size = n_tiles * tile_size_bytes
    };
    auto src0_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());
    auto max_dram_buffer = distributed::MeshBuffer::create(max_buffer_config, dram_config, mesh_device.get());
    auto sum_dram_buffer = distributed::MeshBuffer::create(sum_buffer_config, dram_config, mesh_device.get());
    auto dst_dram_buffer = distributed::MeshBuffer::create(buffer_config, dram_config, mesh_device.get());

    // create circular buffers
    constexpr auto cb_data_format = tt::DataFormat::Float16_b;
    constexpr uint32_t src0_cb_index = CBIndex::c_0;
    CircularBufferConfig cb_src0_config = CircularBufferConfig(
            n_tiles * tile_size_bytes,
            {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, tile_size_bytes);
    auto cb_src = CreateCircularBuffer(program, core, cb_src0_config);

    constexpr uint32_t max_cb_index = CBIndex::c_1;
    CircularBufferConfig cb_max_config = CircularBufferConfig(
            n_tiles * tile_size_bytes,
            {{max_cb_index, cb_data_format}})
            .set_page_size(max_cb_index, tile_size_bytes);
    auto cb_max = CreateCircularBuffer(program, core, cb_max_config);

    constexpr uint32_t sum_cb_index = CBIndex::c_2;
    CircularBufferConfig cb_sum_config = CircularBufferConfig(
            n_tiles * tile_size_bytes,
            {{sum_cb_index, cb_data_format}})
            .set_page_size(sum_cb_index, tile_size_bytes);
    auto cb_sum = CreateCircularBuffer(program, core, cb_sum_config);

    constexpr uint32_t dst_cb_index = CBIndex::c_16;
    CircularBufferConfig cb_dst_config = CircularBufferConfig(
            n_tiles * tile_size_bytes,
            {{dst_cb_index, cb_data_format}})
            .set_page_size(dst_cb_index, tile_size_bytes);
    auto cb_dst = CreateCircularBuffer(program, core, cb_dst_config);

    // create kernel
    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*max_dram_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*sum_dram_buffer).append_to(reader_compile_time_args);
    KernelHandle unary_reader_kernel_id = CreateKernel(
            program,
            "kernels/read_tile.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::RISCV_1_default,
                .compile_args = reader_compile_time_args});

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(*dst_dram_buffer).append_to(writer_compile_time_args);
    KernelHandle unary_writer_kernel_id = CreateKernel(
            program,
            "kernels/write_tile.cpp",
            core,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = writer_compile_time_args});

    KernelHandle eltwise_sfpu_kernel_id = CreateKernel(
            program,
            "kernels/eltwise_sfpu.cpp",
            core,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .math_approx_mode = false,
            });

    // initialize input data and golden data
    auto input = make_input();
    auto golden = softmax_reference(input);

    // write
    distributed::EnqueueWriteMeshBuffer(cq, src0_dram_buffer, input, false);

    std::vector<bfloat16> max_data(32*32, bfloat16(1.0f));
    distributed::EnqueueWriteMeshBuffer(cq, max_dram_buffer, max_data, false);

    std::vector<bfloat16> sum_data(32*32, bfloat16(1.0f));
    distributed::EnqueueWriteMeshBuffer(cq, sum_dram_buffer, sum_data, false);

    // set up the runtime arguments
    SetRuntimeArgs(program, eltwise_sfpu_kernel_id, core, {n_tiles});
    SetRuntimeArgs(program, unary_reader_kernel_id, core, {src0_dram_buffer->address(), max_dram_buffer->address(), sum_dram_buffer->address(), n_tiles});
    SetRuntimeArgs(program, unary_writer_kernel_id, core, {dst_dram_buffer->address(), n_tiles});

    // launch
    workload.add_program(device_range, std::move(program));
    distributed::EnqueueMeshWorkload(cq, workload, false);
    distributed::Finish(cq);

    // readback
    std::vector<bfloat16> output(input.size());
    distributed::EnqueueReadMeshBuffer(cq, output, dst_dram_buffer, true);

    float err = max_abs_error(output, golden);
    std::cout << "Max ABS error: " << err << std::endl;

    return 0;
}

