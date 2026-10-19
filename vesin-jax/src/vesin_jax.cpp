#include <array>
#include <cassert>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <map>
#include <mutex>
#include <string>
#include <utility>

#include <gpulite/gpulite.hpp>

#include "vesin.h"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// The pairs are `int64_t` in JAX and `size_t` in vesin, and we copy or write
// them directly from one to the other
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

// ========================================================================== //

/// Get the state from the corresponding attribute, which contains the address
/// returned by `vesin_jax_state_create`
static VesinJaxState* extract_state(int64_t state) {
    return reinterpret_cast<VesinJaxState*>(static_cast<uintptr_t>(state));
}

static ffi::Error check_inputs(
    ffi::BufferR2<ffi::F64> points,
    ffi::BufferR2<ffi::F64> box,
    ffi::BufferR1<ffi::PRED> periodic
) {
    if (points.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("`points` must be a [n_atoms x 3] array in vesin");
    }

    if (box.dimensions()[0] != 3 || box.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("`box` must be a [3 x 3] array in vesin");
    }

    if (periodic.dimensions()[0] != 3) {
        return ffi::Error::InvalidArgument("`periodic` must be an array of 3 booleans in vesin");
    }

    return ffi::Error::Success();
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
    } else {
        // When copying from pageable host memory, `cudaMemcpyAsync` returns
        // once `src` has been copied to a staging buffer, so it does not need
        // to outlive this call.
        auto kind = src_on_host ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToDevice;
        GPULITE_CUDART_CALL(cudaMemcpyAsync(dst, src, size, kind, static_cast<cudaStream_t>(stream)));
    }
}

static ffi::Error run_vesin(
    VesinDevice device,
    void* stream,
    ffi::BufferR2<ffi::F64> points,
    ffi::BufferR2<ffi::F64> box,
    ffi::BufferR1<ffi::PRED> periodic,
    double cutoff,
    bool full,
    bool sorted,
    int64_t padded_to,
    double skin,
    int64_t algorithm,
    int64_t n_threads,
    bool pre_allocated_outputs,
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

    const char* error_message = nullptr;
    auto status = vesin_neighbors(
        reinterpret_cast<const double (*)[3]>(points.typed_data()),
        points.dimensions()[0],
        reinterpret_cast<const double (*)[3]>(box.typed_data()),
        periodic.typed_data(),
        device,
        stream,
        options,
        neighbors,
        &error_message
    );

    if (status != 0) {
        return ffi::Error::Internal(error_message);
    }

    return ffi::Error::Success();
}

static ffi::Error vesin_jax_compute_impl(
    VesinDevice device,
    void* stream,
    ffi::BufferR2<ffi::F64> points,
    ffi::BufferR2<ffi::F64> box,
    ffi::BufferR1<ffi::PRED> periodic,
    ffi::Result<ffi::BufferR2<ffi::S64>> pairs,
    ffi::Result<ffi::BufferR2<ffi::S32>> shifts,
    int64_t state,
    double cutoff,
    bool full,
    bool sorted,
    int64_t padded_to,
    double skin,
    int64_t algorithm,
    int64_t n_threads
) {
    auto error = check_inputs(points, box, periodic);
    if (error.failure()) {
        return error;
    }

    auto expected_length = points.dimensions()[0] * static_cast<size_t>(padded_to);
    if (pairs->dimensions()[0] != expected_length || pairs->dimensions()[1] != 2) {
        return ffi::Error::Internal("`pairs` output buffer has the wrong shape");
    }

    if (shifts->dimensions()[0] != expected_length || shifts->dimensions()[1] != 3) {
        return ffi::Error::Internal("`shifts` output buffer has the wrong shape");
    }

    auto& data = extract_state(state)->get(device);
    auto lock = data.lock();
    auto& neighbors = data.neighbors;

    // With padding, the number of pairs is known before the calculation, so
    // vesin can write directly in the output buffers allocated by XLA. The
    // `SIZE_MAX` index of padding pseudo-pairs becomes -1 when read as
    // `int64_t`.
    neighbors.pairs = reinterpret_cast<size_t (*)[2]>(pairs->typed_data());
    neighbors.shifts = reinterpret_cast<int32_t (*)[3]>(shifts->typed_data());
    neighbors.length = expected_length;

    error = run_vesin(
        device, stream, points, box, periodic, cutoff, full, sorted, padded_to,
        skin, algorithm, n_threads, /*pre_allocated_outputs=*/true, &neighbors
    );

    if (error.failure()) {
        return error;
    }

    if (neighbors.length != expected_length) {
        return ffi::Error::Internal(
            "vesin returned an unexpected number of pairs: expected " +
            std::to_string(expected_length) + ", got " + std::to_string(neighbors.length)
        );
    }

    return ffi::Error::Success();
}

static ffi::Error vesin_jax_compute_dynamic_impl(
    VesinDevice device,
    void* stream,
    ffi::BufferR2<ffi::F64> points,
    ffi::BufferR2<ffi::F64> box,
    ffi::BufferR1<ffi::PRED> periodic,
    ffi::Result<ffi::BufferR1<ffi::S64>> n_pairs,
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
        device, stream, points, box, periodic, cutoff, full, sorted, /*padded_to=*/0,
        skin, algorithm, n_threads, /*pre_allocated_outputs=*/false, &data.neighbors
    );
    if (error.failure()) {
        return error;
    }

    // the neighbor list is kept inside the state, it will be copied to the
    // output buffers by a later call to `vesin_jax_copy_out`, once the number
    // of pairs is known
    auto length = static_cast<int64_t>(data.neighbors.length);
    copy_to_device(device, stream, n_pairs->typed_data(), &length, sizeof(int64_t), /*src_on_host=*/true);

    return ffi::Error::Success();
}

// `n_pairs` is the output of `vesin_jax_compute_dynamic`. It is not used here,
// but makes sure this runs on the same device, after the calculation.
static ffi::Error vesin_jax_copy_out_impl(
    VesinDevice device,
    void* stream,
    ffi::BufferR1<ffi::S64> /* n_pairs */,
    ffi::Result<ffi::BufferR2<ffi::S64>> pairs,
    ffi::Result<ffi::BufferR2<ffi::S32>> shifts,
    int64_t state
) {
    auto& data = extract_state(state)->get(device);
    auto lock = data.lock();
    const auto& neighbors = data.neighbors;

    if (pairs->dimensions()[0] != neighbors.length || pairs->dimensions()[1] != 2) {
        return ffi::Error::Internal("`pairs` output buffer has the wrong shape");
    }

    if (shifts->dimensions()[0] != neighbors.length || shifts->dimensions()[1] != 3) {
        return ffi::Error::Internal("`shifts` output buffer has the wrong shape");
    }

    // there is no padding here, so all the pairs can be copied as-is
    copy_to_device(
        device, stream, pairs->typed_data(), neighbors.pairs,
        neighbors.length * sizeof(size_t[2]), /*src_on_host=*/false
    );
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
            return impl(VesinDevice{VesinCPU, 0}, nullptr, args...);
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
    return [impl](cudaStream_t stream, int32_t device_id, auto... args) -> ffi::Error {
        try {
            return impl(VesinDevice{VesinCUDA, device_id}, stream, args...);
        } catch (const std::exception& e) {
            return ffi::Error::Internal(e.what());
        }
    };
}

/// Binding for the arguments that only exist on CUDA, matching `on_cuda`
static auto cuda_binding() {
    return ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Ctx<ffi::DeviceOrdinal>();
}

template <typename Binding>
static auto compute_binding(Binding binding) {
    return std::move(binding)
        .template Arg<ffi::BufferR2<ffi::F64>>()
        .template Arg<ffi::BufferR2<ffi::F64>>()
        .template Arg<ffi::BufferR1<ffi::PRED>>()
        .template Ret<ffi::BufferR2<ffi::S64>>()
        .template Ret<ffi::BufferR2<ffi::S32>>()
        .template Attr<int64_t>("state")
        .template Attr<double>("cutoff")
        .template Attr<bool>("full")
        .template Attr<bool>("sorted")
        .template Attr<int64_t>("padded_to")
        .template Attr<double>("skin")
        .template Attr<int64_t>("algorithm")
        .template Attr<int64_t>("n_threads");
}

template <typename Binding>
static auto compute_dynamic_binding(Binding binding) {
    return std::move(binding)
        .template Arg<ffi::BufferR2<ffi::F64>>()
        .template Arg<ffi::BufferR2<ffi::F64>>()
        .template Arg<ffi::BufferR1<ffi::PRED>>()
        .template Ret<ffi::BufferR1<ffi::S64>>()
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
        .template Arg<ffi::BufferR1<ffi::S64>>()
        .template Ret<ffi::BufferR2<ffi::S64>>()
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
