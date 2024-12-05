# Building with SYCL Support

Cutlass 3 can be built with SYCL using the DPC++ compiler, enabling Cutlass
to support other SYCL-enabled devices. This enhancement allows for greater
flexibility and compatibility with diverse computational environments.

SYCL[1] is a royalty-free, cross-platform abstraction layer that enables
code for heterogeneous and offload processors to be written with modern
ISO C++. It provides APIs and abstractions to find devices and manage
resources for GPUs.

## Installing the DPC++ Compiler
One can use the nightly version of the DPC++ compiler, of which the prebuilt packages can be 
procured from [here](https://github.com/intel/llvm/releases), or build it from source as [described](https://github.com/intel/llvm/blob/sycl/sycl/doc/GetStartedGuide.md).

If building from source for the CUDA backend, a minimum CUDA toolkit version of 12.3 is recommended.
If using the pre-built nightlies, a nightly dated no older than 2024-08-02 is required.

## Building and Running on the SYCL backend
To build with the Intel open source `DPC++` compiler when using the SYCL backend
```bash
$ mkdir build && cd build

#set the CC and CXX compilers
export CC=/path/to/dpcpp/clang++
export CXX=/path/to/dpcpp/clang

# compiles for the NVIDIA Ampere GPU architecture
$ cmake -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=nvptx64-nvidia-cuda -DDPCPP_SYCL_ARCH=sm_80 ..

# compiles for the Intel Xe Core Architecture
$ cmake -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=intel_gpu_pvc ..
```
A complete example can be as follows (running on the Intel Data Center Max 1100) - 

```bash
$ cmake -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=intel_gpu_pvc ..

$ make pvc_gemm

$ ./examples/sycl/pvc/pvc_gemm

```
More examples on the Intel GPU can be found in the [sycl example folder](../../examples/sycl/pvc/)

A complete example when running on a A100, using the SYCL backend

```bash
$ cmake -DDPCPP_SYCL_TARGET=nvptx64-nvidia-cuda -DDPCPP_SYCL_ARCH=sm_80

$ make 14_ampere_tf32_tensorop_gemm_cute

$ ./examples/14_ampere_tf32_tensorop_gemm/14_ampere_tf32_tensorop_gemm_cute 

```

## Supported CUTLASS and CUTE Examples
Currently, not all CUTLASS and CUTE examples are supported with the SYCL backend.
as of now, the following are supported - 

CUTE Examples <br>
  * All the CUTE tutorials except `wgmma_sm90` is supported, of which
    * `sgemm_1`, `sgemm_2`, and `tiled_copy` can run on any SYCL device
    * `sgemm_sm80` and `sgemm_sm70` are Nvidia Ampere and Turing specific examples respectively.

CUTLASS Examples <br>
  * Example 14
  * We also provide various SYCL examples for the Intel Data Center Max range of GPUs
  
## SYCL Supported Architectures
At the time of writing, the SYCL backend supports all Nvidia architectures till Ampere, and the 
Intel Data Center Max series of GPUs is supported.


# References

[1] https://www.khronos.org/sycl/
