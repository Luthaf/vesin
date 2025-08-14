// CUDA Types Wrapper - Minimal CUDA type definitions for build-time independence
#ifndef CUDA_TYPES_WRAPPER_HPP
#define CUDA_TYPES_WRAPPER_HPP

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// CUDA Driver API types (from cuda.h)
typedef int CUresult;
typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef unsigned long long CUdeviceptr;

// CUDA Runtime API types (from cuda_runtime.h)
typedef int cudaError_t;
typedef void* cudaStream_t;

// NVRTC types (from nvrtc.h)
typedef int nvrtcResult;
typedef struct _nvrtcProgram* nvrtcProgram;

// dim3 structure for kernel launch parameters
struct dim3 {
    unsigned int x, y, z;
    
    dim3(unsigned int x = 1, unsigned int y = 1, unsigned int z = 1) : x(x), y(y), z(z) {}
};

// CUDA memory copy kinds
typedef enum cudaMemcpyKind {
    cudaMemcpyHostToHost = 0,
    cudaMemcpyHostToDevice = 1,
    cudaMemcpyDeviceToHost = 2,
    cudaMemcpyDeviceToDevice = 3,
    cudaMemcpyDefault = 4
} cudaMemcpyKind;

//global definition for CUDA memory types
typedef enum cudaMemoryType {
      cudaMemoryTypeUnregistered = 0,
      cudaMemoryTypeHost = 1,
      cudaMemoryTypeDevice = 2,
      cudaMemoryTypeManaged = 3
} cudaMemoryType;


// CUDA pointer attributes structure
typedef struct cudaPointerAttributes {
    enum cudaMemoryType type;
    int device;
    void* devicePointer;
    void* hostPointer;
} cudaPointerAttributes;

// CUDA Driver API constants
enum {
    CUDA_SUCCESS = 0,
    CUDA_ERROR_NOT_INITIALIZED = 3
};

// CUDA Runtime API constants
enum {
    cudaSuccess = 0
};

// NVRTC constants
enum {
    NVRTC_SUCCESS = 0
};

// CUDA device attributes
typedef enum CUdevice_attribute {
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK = 8,
    CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN = 97,
    CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK = 83,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75,
    CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76
} CUdevice_attribute;

// CUDA function attributes
typedef enum CUfunction_attribute {
    CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES = 8,
    CU_FUNC_ATTRIBUTE_NUM_REGS = 4
} CUfunction_attribute;

// CUDA pointer attributes for cuPointerGetAttribute
typedef enum CUpointer_attribute {
    CU_POINTER_ATTRIBUTE_CONTEXT = 1,
    CU_POINTER_ATTRIBUTE_MEMORY_TYPE = 2
} CUpointer_attribute;

// CUDA memory types
enum {
    CU_MEMORYTYPE_HOST = 0x01,
    CU_MEMORYTYPE_DEVICE = 0x02,
    CU_MEMORYTYPE_ARRAY = 0x03,
    CU_MEMORYTYPE_UNIFIED = 0x04
};

#ifdef __cplusplus
}
#endif

#endif // CUDA_TYPES_WRAPPER_HPP