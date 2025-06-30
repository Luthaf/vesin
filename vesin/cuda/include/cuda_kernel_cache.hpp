#ifndef CUDA_KERNEL_CACHE_HPP
#define CUDA_KERNEL_CACHE_HPP

#include <fstream>
#include <memory>
#include <string>
#include <typeinfo>
#include <unordered_map>
#include <vector>

#include <cuda.h>
#include <cxxabi.h>
#include <nvrtc.h>

#include "cuda_dynamic_linker.hpp"

/// @brief Demangles a C++ type name to make it human-readable.
std::string demangleTypeName(const std::string &name) {
#if defined(__GNUC__) || defined(__clang__)
  int status = 0;
  std::unique_ptr<char, void (*)(void *)> demangled_name(
      abi::__cxa_demangle(name.c_str(), 0, 0, &status), std::free);
  return (status == 0) ? demangled_name.get() : name;
#else
  throw std::runtime_error("demangling not supported using this toolchain.");
#endif
}

// For non-templated kernels, just return the raw function name
std::string getKernelName(const std::string &fn_name) { return fn_name; }

/// @brief Returns a demangled string representing the type T
template <typename T> std::string typeName() {
  return demangleTypeName(typeid(T).name());
}

/// @brief Builds a comma-separated list of type names from variadic template
/// parameters
template <typename T, typename... Ts>
void buildTemplateTypes(std::string &base) {
  base += typeName<T>();
  if constexpr (sizeof...(Ts) > 0) {
    base += ", ";
    buildTemplateTypes<Ts...>(base);
  }
}

/// @brief Convenience function to initiate type list building
template <typename T, typename... Ts> std::string buildTemplateTypes() {
  std::string result;
  buildTemplateTypes<T, Ts...>(result);
  return result;
}

/// @brief Returns kernel name including template types, e.g., kernel<int,
/// float>
template <typename T, typename... Ts>
std::string getKernelName(const std::string &fn_name) {
  std::string type_list = buildTemplateTypes<T, Ts...>();
  return fn_name + "<" + type_list + ">";
}

/// @brief Loads CUDA source code from a file
std::string load_cuda_source(const std::string &filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open file: " + filename);
  }
  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

/**
 * @class CachedKernel
 * @brief Represents a lazily compiled and cached CUDA kernel.
 *
 * Handles compilation, launching, and dynamic shared memory management.
 */
class CachedKernel {
public:
  CachedKernel(std::string kernel_name, std::string kernel_code,
               std::string source_name, std::vector<std::string> options)
      : kernel_name(std::move(kernel_name)),
        kernel_code(std::move(kernel_code)),
        source_name(std::move(source_name)), options(std::move(options)) {}

  CachedKernel() = default;
  CachedKernel(const CachedKernel &) = default;
  CachedKernel &operator=(const CachedKernel &) = default;

  /// @brief Set function-level attributes like max dynamic shared memory
  inline void setFuncAttribute(CUfunction_attribute attribute,
                               int value) const {
    CUDADRIVER_SAFE_CALL(
        CUDA_DRIVER_INSTANCE.cuFuncSetAttribute(function, attribute, value));
  }

  /// @brief Get function-level attributes like required shared memory size
  int getFuncAttribute(CUfunction_attribute attribute) const {
    int value;
    CUDADRIVER_SAFE_CALL(
        CUDA_DRIVER_INSTANCE.cuFuncGetAttribute(&value, attribute, function));
    return value;
  }

  /**
   * @brief Launch the compiled kernel on the GPU
   *
   * @param grid Grid dimensions
   * @param block Block dimensions
   * @param shared_mem_size Size of dynamic shared memory (in bytes)
   * @param cuda_stream CUDA stream for async execution
   * @param args Arguments to kernel
   * @param synchronize If true, waits for kernel to finish before returning
   */
  void launch(dim3 grid, dim3 block, size_t shared_mem_size, void *cuda_stream,
              std::vector<void *> args, bool synchronize = true) {
    if (!compiled) {
      this->compileKernel(args);
    }

    CUcontext currentContext = nullptr;
    CUresult result = CUDA_DRIVER_INSTANCE.cuCtxGetCurrent(&currentContext);
    if (result != CUDA_SUCCESS || !currentContext) {
      throw std::runtime_error(
          "CachedKernel::launch error getting current context.");
    }

    if (currentContext != context) {
      CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuCtxSetCurrent(context));
    }

    this->checkAndAdjustSharedMem(shared_mem_size);
    cudaStream_t cstream = reinterpret_cast<cudaStream_t>(cuda_stream);

    CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuLaunchKernel(
        function, grid.x, grid.y, grid.z, block.x, block.y, block.z,
        shared_mem_size, cstream, args.data(), 0));

    if (synchronize) {
      CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuCtxSynchronize());
    }

    if (currentContext != context) {
      CUDADRIVER_SAFE_CALL(
          CUDA_DRIVER_INSTANCE.cuCtxSetCurrent(currentContext));
    }
  }

private:
  /// @brief Adjust shared memory size if launch configuration requires more
  /// than the default
  void checkAndAdjustSharedMem(int query_shared_mem_size) {
    if (current_smem_size == 0) {
      CUdevice cuDevice;
      CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuCtxGetDevice(&cuDevice));

      CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuDeviceGetAttribute(
          &max_smem_size_optin,
          CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, cuDevice));
      int reserved_smem_per_block = 0;
      CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuDeviceGetAttribute(
          &reserved_smem_per_block,
          CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK, cuDevice));
      int curr_max_smem_per_block = 0;
      CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuDeviceGetAttribute(
          &curr_max_smem_per_block,
          CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK, cuDevice));
      current_smem_size = (curr_max_smem_per_block - reserved_smem_per_block);
    }

    if (query_shared_mem_size > current_smem_size) {
      if (query_shared_mem_size > max_smem_size_optin) {
        throw std::runtime_error(
            "CachedKernel::launch requested more smem than available.");
      }
      CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuFuncSetAttribute(
          function, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
          query_shared_mem_size));
      current_smem_size = query_shared_mem_size;
    }
  }

  /// @brief Compile the CUDA kernel source using NVRTC
  void compileKernel(std::vector<void *> &kernel_args) {
    this->initCudaDriver();

    CUcontext currentContext = nullptr;

    // Determine which context the memory resides in
    for (void *arg : kernel_args) {
      unsigned int memtype = 0;
      CUdeviceptr device_ptr = *reinterpret_cast<CUdeviceptr *>(arg);
      if (CUDA_DRIVER_INSTANCE.cuPointerGetAttribute(
              &memtype, CU_POINTER_ATTRIBUTE_MEMORY_TYPE, device_ptr) ==
              CUDA_SUCCESS &&
          memtype == CU_MEMORYTYPE_DEVICE) {
        CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuPointerGetAttribute(
            &currentContext, CU_POINTER_ATTRIBUTE_CONTEXT, device_ptr));
        if (currentContext)
          break;
      }
    }

    CUcontext query = nullptr;
    CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuCtxGetCurrent(&query));
    if (query != currentContext) {
      CUDADRIVER_SAFE_CALL(
          CUDA_DRIVER_INSTANCE.cuCtxSetCurrent(currentContext));
    }

    CUdevice cuDevice;
    CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuCtxGetDevice(&cuDevice));

    nvrtcProgram prog;
    NVRTC_SAFE_CALL(NVRTC_INSTANCE.nvrtcCreateProgram(
        &prog, kernel_code.c_str(), source_name.c_str(), 0, nullptr, nullptr));

    NVRTC_SAFE_CALL(
        NVRTC_INSTANCE.nvrtcAddNameExpression(prog, kernel_name.c_str()));

    std::vector<const char *> c_options;
    for (const auto &opt : options) {
      c_options.push_back(opt.c_str());
    }

    int major = 0, minor = 0;
    CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuDeviceGetAttribute(
        &major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, cuDevice));
    CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuDeviceGetAttribute(
        &minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, cuDevice));
    std::string smbuf =
        "--gpu-architecture=sm_" + std::to_string(major * 10 + minor);
    c_options.push_back(smbuf.c_str());

    if (NVRTC_INSTANCE.nvrtcCompileProgram(prog, c_options.size(),
                                           c_options.data()) != NVRTC_SUCCESS) {
      size_t logSize;
      NVRTC_SAFE_CALL(NVRTC_INSTANCE.nvrtcGetProgramLogSize(prog, &logSize));
      std::string log(logSize, '\0');
      NVRTC_SAFE_CALL(NVRTC_INSTANCE.nvrtcGetProgramLog(prog, &log[0]));
      throw std::runtime_error("Kernel compilation failed:\n" + log);
    }

    size_t ptxSize;
    NVRTC_SAFE_CALL(NVRTC_INSTANCE.nvrtcGetPTXSize(prog, &ptxSize));
    std::vector<char> ptxCode(ptxSize);
    NVRTC_SAFE_CALL(NVRTC_INSTANCE.nvrtcGetPTX(prog, ptxCode.data()));

    CUmodule module;
    CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuModuleLoadDataEx(
        &module, ptxCode.data(), 0, 0, 0));

    const char *lowered_name;
    NVRTC_SAFE_CALL(NVRTC_INSTANCE.nvrtcGetLoweredName(
        prog, kernel_name.c_str(), &lowered_name));
    CUfunction kernel;
    CUDADRIVER_SAFE_CALL(CUDA_DRIVER_INSTANCE.cuModuleGetFunction(
        &kernel, module, lowered_name));

    this->module = module;
    this->function = kernel;
    this->context = currentContext;
    this->compiled = true;

    NVRTC_SAFE_CALL(NVRTC_INSTANCE.nvrtcDestroyProgram(&prog));
  }

  /// @brief Initialize CUDA driver API if needed
  void initCudaDriver() {
    int deviceCount = 0;
    if (CUDA_DRIVER_INSTANCE.cuDeviceGetCount(&deviceCount) ==
        CUDA_ERROR_NOT_INITIALIZED) {
      if (CUDA_DRIVER_INSTANCE.cuInit(0) != CUDA_SUCCESS) {
        throw std::runtime_error("Failed to initialize CUDA driver.");
      }
    }
  }

  int current_smem_size = 0;
  int max_smem_size_optin = 0;
  CUmodule module = nullptr;
  CUfunction function = nullptr;
  CUcontext context = nullptr;
  bool compiled = false;

  std::string kernel_name;
  std::string kernel_code;
  std::string source_name;
  std::vector<std::string> options;
};

/**
 * @class KernelFactory
 * @brief Singleton factory for compiling and caching CUDA kernels.
 */
class KernelFactory {
public:
  /// @brief Access singleton instance
  static KernelFactory &instance() {
    static KernelFactory instance;
    return instance;
  }

  /// @brief Caches a kernel given the source string
  void cacheKernel(const std::string &kernel_name,
                   const std::string &source_code,
                   const std::string &source_name,
                   const std::vector<std::string> &options) {
    kernel_cache[kernel_name] = std::make_unique<CachedKernel>(
        kernel_name, source_code, source_name, options);
  }

  /// @brief Checks if kernel exists in cache
  bool hasKernel(const std::string &kernel_name) const {
    return kernel_cache.find(kernel_name) != kernel_cache.end();
  }

  /// @brief Retrieves a cached kernel
  CachedKernel *getKernel(const std::string &kernel_name) const {
    auto it = kernel_cache.find(kernel_name);
    if (it != kernel_cache.end())
      return it->second.get();
    throw std::runtime_error("Kernel not found in cache.");
  }

  /// @brief Create and cache kernel from file source (if not already present)
  CachedKernel *createFromSource(const std::string &kernel_name,
                                 const std::string &source_path,
                                 const std::string &source_name,
                                 const std::vector<std::string> &options) {
    if (!this->hasKernel(kernel_name)) {
      std::string kernel_code = load_cuda_source(source_path);
      this->cacheKernel(kernel_name, kernel_code, source_name, options);
    }
    return this->getKernel(kernel_name);
  }

  /// @brief Create and cache kernel from in-memory string (if not already
  /// present)
  CachedKernel *create(const std::string &kernel_name,
                       const std::string &source_variable,
                       const std::string &source_name,
                       const std::vector<std::string> &options) {
    if (!this->hasKernel(kernel_name)) {
      this->cacheKernel(kernel_name, source_variable, source_name, options);
    }
    return this->getKernel(kernel_name);
  }

private:
  KernelFactory() {}
  std::unordered_map<std::string, std::unique_ptr<CachedKernel>> kernel_cache;

  KernelFactory(const KernelFactory &) = delete;
  KernelFactory &operator=(const KernelFactory &) = delete;
};

#endif // CUDA_KERNEL_CACHE_HPP
