#include <torch/extension.h>
#include <vector>
#include <tuple>
#include <string>


// CUDA forward declarations

std::tuple<std::vector<torch::Tensor>,const char*> spc_cuda(
    torch::Tensor quantized_points,
    int level);


// C++ interface

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

std::tuple<std::vector<torch::Tensor>,const char*> spc( torch::Tensor quantized_points, int level ) {
  CHECK_INPUT(quantized_points);
  return spc_cuda(quantized_points, level);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("spc", &spc, "spc (CUDA)");
}