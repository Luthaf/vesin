#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <gpulite/gpulite.hpp>

#include "vesin.h"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// The pairs are `int64_t` in JAX (with `jax_enable_x64`) and `size_t` in vesin,
// and we copy or write them directly from one to the other
static_assert(sizeof(size_t) == sizeof(int64_t), "vesin-jax requires 64-bit size_t");

#if defined(_MSC_VER) && !defined(__clang__)
#define VESIN_JAX_EXPORT __declspec(dllexport)
#else
#define VESIN_JAX_EXPORT __attribute__((visibility("default")))
#endif

// The library is compiled with hidden visibility by default; explicitly export
// the FFI handlers and the state lifecycle functions, which are the only
// symbols used by jaxlib and the Python code.
extern "C" {
VESIN_JAX_EXPORT XLA_FFI_Error* vesin_jax_compute_cpu(XLA_FFI_CallFrame* call_frame);
VESIN_JAX_EXPORT XLA_FFI_Error* vesin_jax_compute_dynamic_cpu(XLA_FFI_CallFrame* call_frame);
VESIN_JAX_EXPORT XLA_FFI_Error* vesin_jax_copy_out_cpu(XLA_FFI_CallFrame* call_frame);

VESIN_JAX_EXPORT XLA_FFI_Error* vesin_jax_compute_CUDA(XLA_FFI_CallFrame* call_frame);
VESIN_JAX_EXPORT XLA_FFI_Error* vesin_jax_compute_dynamic_CUDA(XLA_FFI_CallFrame* call_frame);
VESIN_JAX_EXPORT XLA_FFI_Error* vesin_jax_copy_out_CUDA(XLA_FFI_CallFrame* call_frame);
}

/// Neighbor list for a single device, and the lock protecting it
class DeviceNeighbors {
public:
    VesinNeighborList neighbors;

    /// Lock this neighbor list, preventing concurrent use (for example when the
    /// same `NeighborList` is used from multiple threads). This emits a warning
    /// (once per process) if another thread is already using it.
    ///
    /// The warning is printed directly to stderr, since the FFI handlers run
    /// on threads managed by XLA, from which we can not safely call Python's
    /// `warnings.warn`.
    std::unique_lock<std::mutex> lock() {
        auto lock = std::unique_lock<std::mutex>(this->mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
            static std::once_flag warned;
            std::call_once(warned, [] {
                std::cerr << "Warning: the same vesin_jax.NeighborList is being "
                          << "used from multiple threads at the same time. The "
                          << "calculations will run one after the other, use a "
                          << "separate NeighborList in each thread to run them "
                          << "in parallel.\n";
            });
            lock.lock();
        }
        return lock;
    }

private:
    std::mutex mutex_;
};

/// State of a `vesin_jax.NeighborList`, kept between calls. A
/// `VesinNeighborList` can only be used with a single device, so this contains
/// one of them for each device the calculation runs on.
///
/// The same `VesinJaxState` would be used with multiple devices through
/// `shard_map`/`pmap`, since the state is created by JAX once when tracing, and
/// then re-used across multiple devices.
class VesinJaxState {
public:
    VesinJaxState() = default;

    ~VesinJaxState() {
        for (auto& [_, device] : this->devices_) {
            vesin_free(&device.neighbors);
        }
    }

    VesinJaxState(const VesinJaxState&) = delete;
    VesinJaxState& operator=(const VesinJaxState&) = delete;
    VesinJaxState(VesinJaxState&&) = delete;
    VesinJaxState& operator=(VesinJaxState&&) = delete;

    /// Get the neighbor list for the given `device`, creating it if needed
    DeviceNeighbors& get(VesinDevice device) {
        auto lock = std::lock_guard<std::mutex>(this->mutex_);
        // `std::map` never moves its elements, so the returned reference stays
        // valid when inserting new devices
        return this->devices_[device];
    }

private:
    struct VesinDeviceCompare {
        bool operator()(const VesinDevice& lhs, const VesinDevice& rhs) const {
            if (lhs.type != rhs.type) {
                return lhs.type < rhs.type;
            }
            return lhs.device_id < rhs.device_id;
        }
    };

    std::mutex mutex_;
    std::map<VesinDevice, DeviceNeighbors, VesinDeviceCompare> devices_;
};

/// Create a new state
extern "C" VESIN_JAX_EXPORT VesinJaxState* vesin_jax_state_create() {
    try {
        return new VesinJaxState();
    } catch (...) {
        return nullptr;
    }
}

/// Free a state created with `vesin_jax_state_create`
extern "C" VESIN_JAX_EXPORT void vesin_jax_state_free(VesinJaxState* state) {
    delete state;
}

/// Get the state from the corresponding attribute, which contains the address
/// returned by `vesin_jax_state_create`
static VesinJaxState* extract_state(int64_t state) {
    return reinterpret_cast<VesinJaxState*>(static_cast<uintptr_t>(state));
}

// ========================================================================== //
//                  Conversion between JAX and vesin types                    //
// ========================================================================== //

// Without `jax_enable_x64`, JAX only uses 32-bit arrays. The points and box
// are then given as `float` and the pairs should be returned as `int32_t`,
// while vesin works with `double` and `size_t`. The data is converted with the
// functions below, either on the CPU or with the small CUDA kernels.

static const char* CUDA_CONVERSION_CODE = R"(
__global__ void vesin_jax_float_to_double(
    const float* input,
    double* output,
    unsigned long long n
) {
    auto i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        output[i] = static_cast<double>(input[i]);
    }
}

__global__ void vesin_jax_size_t_to_int32(
    const unsigned long long* input,
    int* output,
    unsigned long long n
) {
    auto i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) {
        // `SIZE_MAX = ~0ull` is used by vesin for padding pseudo-pairs
        auto value = input[i];
        output[i] = value == ~0ull ? -1 : static_cast<int>(value);
    }
}
)";

using FloatToDoubleKernel = void(const float*, double*, uint64_t);
using SizeToInt32Kernel = void(const size_t*, int32_t*, uint64_t);

/// Get the configuration to launch a conversion kernel on `n` elements
static gpulite::LaunchConfig conversion_launch_config(void* stream, size_t n) {
    const size_t THREADS_PER_BLOCK = 256;
    auto config = gpulite::LaunchConfig();
    config.stream = static_cast<cudaStream_t>(stream);
    config.gridDim = dim3(static_cast<unsigned>((n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK));
    config.blockDim = dim3(THREADS_PER_BLOCK);
    return config;
}

/// Temporary memory, which stays valid until the end of the current FFI call.
/// On CUDA, the memory comes from XLA's allocator, and is ordered with respect
/// to the stream of the call. XLA does not support `ffi::ScratchAllocator` on
/// CPU, so we use memory allocated with `new` there instead.
class Workspace {
public:
    /// Create a workspace using the XLA `scratch` allocator, or host memory if
    /// `scratch` is `nullptr`
    explicit Workspace(ffi::ScratchAllocator& scratch): scratch_(scratch) {}
    explicit Workspace(): scratch_(std::nullopt) {}

    /// Allocate `size` bytes of memory
    void* allocate(size_t size) {
        if (!scratch_) {
            host_.emplace_back(new char[size]);
            return host_.back().get();
        }

        auto ptr = scratch_->get().Allocate(size, alignof(std::max_align_t));
        if (!ptr.has_value()) {
            throw std::runtime_error("failed to allocate temporary memory for vesin-jax");
        }
        return ptr.value();
    }

private:
    std::optional<std::reference_wrapper<ffi::ScratchAllocator>> scratch_;
    std::vector<std::unique_ptr<char[]>> host_;
};

/// Get the data in `buffer` (which contains either `float` or `double`) as
/// `double`, converting it in temporary memory if needed.
static const double* as_double(
    VesinDevice device,
    void* stream,
    Workspace& workspace,
    ffi::AnyBuffer buffer
) {
    if (buffer.element_type() == ffi::F64) {
        return buffer.typed_data<double>();
    }

    assert(buffer.element_type() == ffi::F32);
    auto n = buffer.element_count();
    const auto* input = buffer.typed_data<float>();
    auto* output = static_cast<double*>(workspace.allocate(n * sizeof(double)));

    if (n == 0) {
        return output;
    }

    if (device.type == VesinCPU) {
        for (size_t i = 0; i < n; i++) {
            output[i] = static_cast<double>(input[i]);
        }
    } else if (device.type == VesinCUDA) {
        auto& factory = gpulite::KernelFactory::instance(device.device_id);
        auto* kernel = factory.create<FloatToDoubleKernel>(
            "vesin_jax_float_to_double", CUDA_CONVERSION_CODE, "vesin_jax.cu", {}
        );
        kernel->launch(conversion_launch_config(stream, n), input, output, static_cast<uint64_t>(n));
    } else {
        throw std::runtime_error("vesin-jax only supports CPU and CUDA devices");
    }

    return output;
}

/// Copy `size` bytes from `src` to `dst`, where `dst` is in the memory of
/// `device`, and `src` is either in the memory of `device` or on the host
/// (according to `src_on_host`).
static void copy_to_device(
    VesinDevice device,
    void* stream,
    void* dst,
    const void* src,
    size_t size,
    bool src_on_host
) {
    if (size == 0) {
        return;
    }

    if (device.type == VesinCPU) {
        std::memcpy(dst, src, size);
    } else if (device.type == VesinCUDA) {
        // When copying from pageable host memory, `cudaMemcpyAsync` returns
        // once `src` has been copied to a staging buffer, so it does not need
        // to outlive this call.
        auto kind = src_on_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
        GPULITE_CUDART_CALL(cudaMemcpyAsync(dst, src, size, kind, static_cast<cudaStream_t>(stream)));
    } else {
        throw std::runtime_error("vesin-jax only supports CPU and CUDA devices");
    }
}

/// Set `size` bytes starting at `dst` (in the memory of `device`) to `value`
static void fill_bytes(VesinDevice device, void* stream, void* dst, int value, size_t size) {
    if (size == 0) {
        return;
    }

    if (device.type == VesinCPU) {
        std::memset(dst, value, size);
    } else if (device.type == VesinCUDA) {
        GPULITE_CUDART_CALL(cudaMemsetAsync(dst, value, size, static_cast<cudaStream_t>(stream)));
    } else {
        throw std::runtime_error("vesin-jax only supports CPU and CUDA devices");
    }
}

/// Copy `n_pairs` pairs from `pairs` to the `output` buffer (containing either
/// `int64_t` or `int32_t`), both in the memory of `device`.
static void copy_pairs(
    VesinDevice device,
    void* stream,
    ffi::Result<ffi::AnyBuffer>& output,
    const size_t (*pairs)[2],
    size_t n_pairs
) {
    if (output->element_type() == ffi::S64) {
        copy_to_device(
            device, stream, output->untyped_data(), pairs,
            n_pairs * sizeof(size_t[2]), /*src_on_host=*/false
        );
        return;
    }

    assert(output->element_type() == ffi::S32);
    auto n = 2 * n_pairs;
    const auto* input = reinterpret_cast<const size_t*>(pairs);
    auto* output_data = output->typed_data<int32_t>();

    if (n == 0) {
        return;
    }

    if (device.type == VesinCPU) {
        for (size_t i = 0; i < n; i++) {
            // `SIZE_MAX` is used by vesin for padding pseudo-pairs
            output_data[i] = input[i] == SIZE_MAX ? -1 : static_cast<int32_t>(input[i]);
        }
    } else if (device.type == VesinCUDA) {
        auto& factory = gpulite::KernelFactory::instance(device.device_id);
        auto* kernel = factory.create<SizeToInt32Kernel>(
            "vesin_jax_size_t_to_int32", CUDA_CONVERSION_CODE, "vesin_jax.cu", {}
        );
        kernel->launch(conversion_launch_config(stream, n), input, output_data, static_cast<uint64_t>(n));
    } else {
        throw std::runtime_error("vesin-jax only supports CPU and CUDA devices");
    }
}

// ========================================================================== //

static ffi::Error check_inputs(
    ffi::AnyBuffer points,
    ffi::AnyBuffer box,
    ffi::BufferR1<ffi::PRED> periodic
) {
    auto dtype = points.element_type();
    if (dtype != ffi::F32 && dtype != ffi::F64) {
        return ffi::Error::InvalidArgument("`points` must contain 32 or 64-bit floats in vesin");
    }

    if (box.element_type() != dtype) {
        return ffi::Error::InvalidArgument("`points` and `box` must have the same dtype in vesin");
    }

    auto points_shape = points.dimensions();
    if (points_shape.size() != 2 || points_shape[1] != 3) {
        return ffi::Error::InvalidArgument("`points` must be a [n_atoms x 3] array in vesin");
    }

    auto box_shape = box.dimensions();
    if (box_shape.size() != 2 || box_shape[0] != 3 || box_shape[1] != 3) {
        return ffi::Error::InvalidArgument("`box` must be a [3 x 3] array in vesin");
    }

    if (periodic.dimensions()[0] != 3) {
        return ffi::Error::InvalidArgument("`periodic` must be an array of 3 booleans in vesin");
    }

    return ffi::Error::Success();
}

/// Check that the `pairs` and `shifts` outputs can contain `n_pairs` pairs
static ffi::Error check_outputs(
    ffi::Result<ffi::AnyBuffer>& pairs,
    ffi::Result<ffi::BufferR2<ffi::S32>>& shifts,
    size_t n_pairs
) {
    auto dtype = pairs->element_type();
    if (dtype != ffi::S32 && dtype != ffi::S64) {
        return ffi::Error::Internal("`pairs` output buffer must contain 32 or 64-bit integers");
    }

    auto pairs_shape = pairs->dimensions();
    if (pairs_shape.size() != 2 || pairs_shape[0] != n_pairs || pairs_shape[1] != 2) {
        return ffi::Error::Internal("`pairs` output buffer has the wrong shape");
    }

    if (shifts->dimensions()[0] != n_pairs || shifts->dimensions()[1] != 3) {
        return ffi::Error::Internal("`shifts` output buffer has the wrong shape");
    }

    return ffi::Error::Success();
}

static ffi::Error run_vesin(
    VesinDevice device,
    void* stream,
    Workspace& workspace,
    ffi::AnyBuffer points,
    ffi::AnyBuffer box,
    ffi::BufferR1<ffi::PRED> periodic,
    double cutoff,
    bool full,
    bool sorted,
    int64_t padded_to,
    double skin,
    int64_t algorithm,
    int64_t n_threads,
    bool pre_allocated_outputs,
    bool allow_output_too_small,
    VesinNeighborList* neighbors
) {
    VesinOptions options;
    options.cutoff = cutoff;
    options.full = full;
    options.sorted = sorted;
    assert(padded_to >= 0);
    options.padded_to = static_cast<size_t>(padded_to);
    options.algorithm = static_cast<VesinAlgorithm>(algorithm);
    options.skin = skin;
    options.n_threads = static_cast<int32_t>(n_threads);

    // shifts are always returned: they are needed to recompute distances and
    // vectors with JAX operations, making the calculation differentiable
    options.return_shifts = true;
    options.return_distances = false;
    options.return_vectors = false;
    options.pre_allocated_outputs = pre_allocated_outputs;

    const auto* points_data = as_double(device, stream, workspace, points);
    const auto* box_data = as_double(device, stream, workspace, box);

    const char* error_message = nullptr;
    auto status = vesin_neighbors(
        reinterpret_cast<const double (*)[3]>(points_data),
        points.dimensions()[0],
        reinterpret_cast<const double (*)[3]>(box_data),
        periodic.typed_data(),
        device,
        stream,
        options,
        neighbors,
        &error_message
    );

    // with `allow_output_too_small`, the caller detects too small outputs by
    // comparing `neighbors->length` with the size of the outputs
    if (status == VESIN_OUTPUT_TOO_SMALL && allow_output_too_small) {
        return ffi::Error::Success();
    }

    if (status != VESIN_SUCCESS) {
        return ffi::Error::Internal(error_message);
    }

    return ffi::Error::Success();
}

/// Write `length` in the `n_pairs` buffer, which must contain a single 32 or
/// 64-bit integer
static ffi::Error write_n_pairs(
    VesinDevice device,
    void* stream,
    ffi::Result<ffi::AnyBuffer>& n_pairs,
    size_t length
) {
    if (n_pairs->element_count() != 1) {
        return ffi::Error::Internal("`n_pairs` output buffer has the wrong shape");
    }

    if (n_pairs->element_type() == ffi::S64) {
        auto length_64 = static_cast<int64_t>(length);
        copy_to_device(device, stream, n_pairs->untyped_data(), &length_64, sizeof(int64_t), /*src_on_host=*/true);
    } else if (n_pairs->element_type() == ffi::S32) {
        if (length > static_cast<size_t>(INT32_MAX)) {
            return ffi::Error::Internal(
                "the neighbor list contains too many pairs (" + std::to_string(length) +
                ") to be used without 64-bit arrays, please enable them with "
                "`jax.config.update('jax_enable_x64', True)`"
            );
        }
        auto length_32 = static_cast<int32_t>(length);
        copy_to_device(device, stream, n_pairs->untyped_data(), &length_32, sizeof(int32_t), /*src_on_host=*/true);
    } else {
        return ffi::Error::Internal("`n_pairs` output buffer must contain 32 or 64-bit integers");
    }

    return ffi::Error::Success();
}

static ffi::Error vesin_jax_compute_impl(
    VesinDevice device,
    void* stream,
    Workspace& workspace,
    ffi::AnyBuffer points,
    ffi::AnyBuffer box,
    ffi::BufferR1<ffi::PRED> periodic,
    ffi::Result<ffi::AnyBuffer> pairs,
    ffi::Result<ffi::BufferR2<ffi::S32>> shifts,
    ffi::Result<ffi::AnyBuffer> n_pairs,
    int64_t state,
    double cutoff,
    bool full,
    bool sorted,
    int64_t padded_to,
    int64_t fixed_n_pairs,
    double skin,
    int64_t algorithm,
    int64_t n_threads,
    bool return_n_pairs
) {
    auto error = check_inputs(points, box, periodic);
    if (error.failure()) {
        return error;
    }

    // exactly one of `padded_to` and `fixed_n_pairs` is set, and gives the
    // number of pairs in the outputs
    assert((padded_to > 0) != (fixed_n_pairs > 0));
    size_t capacity = 0;
    if (padded_to > 0) {
        capacity = points.dimensions()[0] * static_cast<size_t>(padded_to);
    } else {
        capacity = static_cast<size_t>(fixed_n_pairs);
    }

    error = check_outputs(pairs, shifts, capacity);
    if (error.failure()) {
        return error;
    }

    auto& data = extract_state(state)->get(device);
    auto lock = data.lock();
    auto& neighbors = data.neighbors;

    // The maximal number of pairs is known before the calculation, so vesin
    // can write directly in the output buffers allocated by XLA. The
    // `SIZE_MAX` index of padding pseudo-pairs becomes -1 when read as
    // `int64_t`. If the pairs are `int32_t`, vesin writes them to temporary
    // memory instead, and they are converted afterward.
    auto pairs_are_int64 = pairs->element_type() == ffi::S64;
    if (pairs_are_int64) {
        neighbors.pairs = reinterpret_cast<size_t (*)[2]>(pairs->untyped_data());
    } else {
        neighbors.pairs = static_cast<size_t (*)[2]>(workspace.allocate(capacity * sizeof(size_t[2])));
    }
    neighbors.shifts = reinterpret_cast<int32_t (*)[3]>(shifts->typed_data());
    neighbors.length = capacity;

    error = run_vesin(
        device, stream, workspace, points, box, periodic, cutoff, full, sorted,
        padded_to, skin, algorithm, n_threads, /*pre_allocated_outputs=*/true,
        /*allow_output_too_small=*/return_n_pairs, &neighbors
    );

    if (error.failure()) {
        return error;
    }

    // `length` is larger than the outputs if they are too small, and then
    // contains the number of pairs they should be able to hold
    auto too_small = neighbors.length > capacity;
    if (padded_to > 0 && !too_small && neighbors.length != capacity) {
        return ffi::Error::Internal(
            "vesin returned an unexpected number of pairs: expected " +
            std::to_string(capacity) + ", got " + std::to_string(neighbors.length)
        );
    }

    error = write_n_pairs(device, stream, n_pairs, neighbors.length);
    if (error.failure()) {
        return error;
    }

    // number of pairs vesin wrote in the outputs
    auto n_written = too_small ? 0 : neighbors.length;
    if (!pairs_are_int64) {
        copy_pairs(device, stream, pairs, neighbors.pairs, n_written);
    }

    // With `fixed_n_pairs`, vesin only wrote the first `n_written` pairs. The
    // remaining ones are filled with `(-1, -1)` pseudo-pairs (all bits set to
    // 1 in two's complement) and zero shifts.
    auto n_padding = capacity - n_written;
    auto pair_size = 2 * ffi::ByteWidth(pairs->element_type());
    fill_bytes(
        device, stream,
        static_cast<char*>(pairs->untyped_data()) + n_written * pair_size,
        0xFF,
        n_padding * pair_size
    );
    fill_bytes(
        device, stream,
        shifts->typed_data() + 3 * n_written,
        0,
        n_padding * sizeof(int32_t[3])
    );

    return ffi::Error::Success();
}

static ffi::Error vesin_jax_compute_dynamic_impl(
    VesinDevice device,
    void* stream,
    Workspace& workspace,
    ffi::AnyBuffer points,
    ffi::AnyBuffer box,
    ffi::BufferR1<ffi::PRED> periodic,
    ffi::Result<ffi::AnyBuffer> n_pairs,
    int64_t state,
    double cutoff,
    bool full,
    bool sorted,
    double skin,
    int64_t algorithm,
    int64_t n_threads
) {
    auto error = check_inputs(points, box, periodic);
    if (error.failure()) {
        return error;
    }

    auto& data = extract_state(state)->get(device);
    auto lock = data.lock();

    error = run_vesin(
        device, stream, workspace, points, box, periodic, cutoff, full, sorted,
        /*padded_to=*/0, skin, algorithm, n_threads, /*pre_allocated_outputs=*/false,
        /*allow_output_too_small=*/false, &data.neighbors
    );
    if (error.failure()) {
        return error;
    }

    // the neighbor list is kept inside the state, it will be copied to the
    // output buffers by a later call to `vesin_jax_copy_out`, once the number
    // of pairs is known
    return write_n_pairs(device, stream, n_pairs, data.neighbors.length);
}

// `n_pairs` is the output of `vesin_jax_compute_dynamic`. It is not used here,
// but makes sure this runs on the same device, after the calculation.
static ffi::Error vesin_jax_copy_out_impl(
    VesinDevice device,
    void* stream,
    Workspace& /* workspace */,
    ffi::AnyBuffer /* n_pairs */,
    ffi::Result<ffi::AnyBuffer> pairs,
    ffi::Result<ffi::BufferR2<ffi::S32>> shifts,
    int64_t state
) {
    auto& data = extract_state(state)->get(device);
    auto lock = data.lock();
    const auto& neighbors = data.neighbors;

    auto error = check_outputs(pairs, shifts, neighbors.length);
    if (error.failure()) {
        return error;
    }

    // there is no padding here, so all the pairs can be copied as-is
    copy_pairs(device, stream, pairs, neighbors.pairs, neighbors.length);
    copy_to_device(
        device, stream, shifts->typed_data(), neighbors.shifts,
        neighbors.length * sizeof(int32_t[3]), /*src_on_host=*/false
    );

    return ffi::Error::Success();
}

// ========================================================================== //
//                    FFI handlers for each platform                          //
// ========================================================================== //

/// Wrap one of the `_impl` functions above in a handler for the CPU platform
template <typename Impl>
static constexpr auto on_cpu(Impl impl) {
    return [impl](auto... args) -> ffi::Error {
        try {
            auto workspace = Workspace();
            return impl(VesinDevice{VesinCPU, 0}, nullptr, workspace, args...);
        } catch (const std::exception& e) {
            return ffi::Error::Internal(e.what());
        }
    };
}

/// Binding for the arguments that only exist on CPU, matching `on_cpu`
static auto cpu_binding() {
    return ffi::Ffi::Bind();
}

/// Wrap one of the `_impl` functions above in a handler for the CUDA platform,
/// running on the stream and device given by XLA
template <typename Impl>
static constexpr auto on_cuda(Impl impl) {
    return [impl](cudaStream_t stream, int32_t device_id, ffi::ScratchAllocator scratch, auto... args) -> ffi::Error {
        try {
            auto workspace = Workspace(scratch);
            return impl(VesinDevice{VesinCUDA, device_id}, stream, workspace, args...);
        } catch (const std::exception& e) {
            return ffi::Error::Internal(e.what());
        }
    };
}

/// Binding for the arguments that only exist on CUDA, matching `on_cuda`
static auto cuda_binding() {
    return ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::DeviceOrdinal>()
        .Ctx<ffi::ScratchAllocator>();
}

template <typename Binding>
static auto compute_binding(Binding binding) {
    return std::move(binding)
        .template Arg<ffi::AnyBuffer>()
        .template Arg<ffi::AnyBuffer>()
        .template Arg<ffi::BufferR1<ffi::PRED>>()
        .template Ret<ffi::AnyBuffer>()
        .template Ret<ffi::BufferR2<ffi::S32>>()
        .template Ret<ffi::AnyBuffer>()
        .template Attr<int64_t>("state")
        .template Attr<double>("cutoff")
        .template Attr<bool>("full")
        .template Attr<bool>("sorted")
        .template Attr<int64_t>("padded_to")
        .template Attr<int64_t>("fixed_n_pairs")
        .template Attr<double>("skin")
        .template Attr<int64_t>("algorithm")
        .template Attr<int64_t>("n_threads")
        .template Attr<bool>("return_n_pairs");
}

template <typename Binding>
static auto compute_dynamic_binding(Binding binding) {
    return std::move(binding)
        .template Arg<ffi::AnyBuffer>()
        .template Arg<ffi::AnyBuffer>()
        .template Arg<ffi::BufferR1<ffi::PRED>>()
        .template Ret<ffi::AnyBuffer>()
        .template Attr<int64_t>("state")
        .template Attr<double>("cutoff")
        .template Attr<bool>("full")
        .template Attr<bool>("sorted")
        .template Attr<double>("skin")
        .template Attr<int64_t>("algorithm")
        .template Attr<int64_t>("n_threads");
}

template <typename Binding>
static auto copy_out_binding(Binding binding) {
    return std::move(binding)
        .template Arg<ffi::AnyBuffer>()
        .template Ret<ffi::AnyBuffer>()
        .template Ret<ffi::BufferR2<ffi::S32>>()
        .template Attr<int64_t>("state");
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    vesin_jax_compute_cpu,
    on_cpu(vesin_jax_compute_impl),
    compute_binding(cpu_binding())
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    vesin_jax_compute_dynamic_cpu,
    on_cpu(vesin_jax_compute_dynamic_impl),
    compute_dynamic_binding(cpu_binding())
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    vesin_jax_copy_out_cpu,
    on_cpu(vesin_jax_copy_out_impl),
    copy_out_binding(cpu_binding())
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    vesin_jax_compute_CUDA,
    on_cuda(vesin_jax_compute_impl),
    compute_binding(cuda_binding())
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    vesin_jax_compute_dynamic_CUDA,
    on_cuda(vesin_jax_compute_dynamic_impl),
    compute_dynamic_binding(cuda_binding())
);

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    vesin_jax_copy_out_CUDA,
    on_cuda(vesin_jax_copy_out_impl),
    copy_out_binding(cuda_binding())
);
