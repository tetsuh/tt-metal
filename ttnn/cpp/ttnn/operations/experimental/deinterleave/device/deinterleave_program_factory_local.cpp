#include "deinterleave_device_operation.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"

namespace ttnn::operations::experimental::deinterleave {
DeinterleaveLocalOperation::ProgramFactoryLocal::cached_program_t
DeinterleaveLocalOperation::ProgramFactoryLocal::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // TODO: implement
    Program program;

    tt::tt_metal::CoreRangeSet worker_grid;
    tt::tt_metal::KernelHandle read_kernel_id = CreateReadKernel(program, "", worker_grid);
    tt::tt_metal::KernelHandle write_kernel_id = CreateWriteKernel(program, "", worker_grid);

    return {std::move(program), {read_kernel_id, write_kernel_id, worker_grid}};
}

void DeinterleaveLocalOperation::ProgramFactoryLocal::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    const auto& program = cached_program.program;
    const auto& read_kernel_id = cached_program.shared_variables.read_kernel_id;
    const auto& write_kernel_id = cached_program.shared_variables.write_kernel_id;

    TT_FATAL(false, "to resolve overriding runtime args");
}
}  // namespace ttnn::operations::experimental::deinterleave
