#include <array>
#include <cstdint>
#include <cstring>

#include "vesin.h"

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// Buffer holding the address of a `VesinNeighborList` allocated by
// `vesin_jax_state_create`, passed as a (1,) int64 array on CPU
using StatePointer = ffi::BufferR1<ffi::S64>;

#if defined(_MSC_VER) && !defined(__clang__)
#define VESIN_JAX_EXPORT __declspec(dllexport)
#else
#define VESIN_JAX_EXPORT __attribute__((visibility("default")))
#endif

// The library is compiled with hidden visibility by default; explicitly export
// the FFI handlers and the state lifecycle functions, which are the only
// symbols used by jaxlib and the Python code.
VESIN_JAX_EXPORT XLA_FFI_DECLARE_HANDLER_SYMBOL(vesin_jax_compute);
VESIN_JAX_EXPORT XLA_FFI_DECLARE_HANDLER_SYMBOL(vesin_jax_compute_dynamic);
VESIN_JAX_EXPORT XLA_FFI_DECLARE_HANDLER_SYMBOL(vesin_jax_copy_out);

/// Create a new state/`VesinNeighborList`
extern "C" VESIN_JAX_EXPORT VesinNeighborList* vesin_jax_state_create() {
    auto* state = new VesinNeighborList();
    std::memset(state, 0, sizeof(VesinNeighborList));
    return state;
}

/// Free a state/`VesinNeighborList`
extern "C" VESIN_JAX_EXPORT void vesin_jax_state_free(VesinNeighborList* state) {
    if (state != nullptr) {
        vesin_free(state);
        delete state;
    }
}

// ========================================================================== //

/// Get a `VesinNeighborList` from the jax buffer
static VesinNeighborList* extract_state(const StatePointer& state_ptr) {
    return reinterpret_cast<VesinNeighborList*>(state_ptr.typed_data()[0]);
}


static ffi::Error check_inputs(ffi::BufferR2<ffi::F64> points, ffi::BufferR2<ffi::F64> box) {
    if (points.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("`points` must be a [n_atoms x 3] array in vesin");
    }

    if (box.dimensions()[0] != 3 || box.dimensions()[1] != 3) {
        return ffi::Error::InvalidArgument("`box` must be a [3 x 3] array in vesin");
    }

    return ffi::Error::Success();
}

static ffi::Error run_vesin(
    ffi::BufferR2<ffi::F64> points,
    ffi::BufferR2<ffi::F64> box,
    std::array<bool, 3> periodic,
    double cutoff,
    bool full,
    bool sorted,
    int64_t padded_to,
    double skin,
    int64_t algorithm,
    int64_t n_threads,
    VesinNeighborList* state
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

    VesinDevice device{VesinCPU, 0};

    const char* error_message = nullptr;
    auto status = vesin_neighbors(
        reinterpret_cast<const double (*)[3]>(points.typed_data()),
        points.dimensions()[0],
        reinterpret_cast<const double (*)[3]>(box.typed_data()),
        periodic.data(),
        device,
        options,
        state,
        &error_message
    );

    if (status != 0) {
        return ffi::Error::Internal(error_message);
    }

    return ffi::Error::Success();
}

static void copy_pairs(const VesinNeighborList& state, int64_t* pairs, int32_t* shifts) {
    for (size_t i = 0; i < state.length; i++) {
        pairs[2 * i] = static_cast<int64_t>(state.pairs[i][0]);
        // pseudo-pairs from padded lists use `SIZE_MAX` as the second index,
        // convert them to -1
        if (state.pairs[i][1] == SIZE_MAX) {
            pairs[2 * i + 1] = -1;
        } else {
            pairs[2 * i + 1] = static_cast<int64_t>(state.pairs[i][1]);
        }

        shifts[3 * i] = state.shifts[i][0];
        shifts[3 * i + 1] = state.shifts[i][1];
        shifts[3 * i + 2] = state.shifts[i][2];
    }
}

static ffi::Error vesin_jax_compute_impl(
    ffi::BufferR2<ffi::F64> points,
    ffi::BufferR2<ffi::F64> box,
    const StatePointer& state_ptr,
    ffi::Result<ffi::BufferR2<ffi::S64>> pairs,
    ffi::Result<ffi::BufferR2<ffi::S32>> shifts,
    bool periodic_x,
    bool periodic_y,
    bool periodic_z,
    double cutoff,
    bool full,
    bool sorted,
    int64_t padded_to,
    double skin,
    int64_t algorithm,
    int64_t n_threads
) {
    auto error = check_inputs(points, box);
    if (error.failure()) {
        return error;
    }

    auto* state = extract_state(state_ptr);

    auto periodic = std::array<bool, 3>{periodic_x, periodic_y, periodic_z};
    error = run_vesin(
        points, box, periodic, cutoff, full, sorted, padded_to, skin,
        algorithm, n_threads, state
    );
    if (error.failure()) {
        return error;
    }

    auto expected_length = points.dimensions()[0] * static_cast<size_t>(padded_to);
    if (state->length != expected_length) {
        return ffi::Error::Internal(
            "vesin returned an unexpected number of pairs: expected " +
            std::to_string(expected_length) + ", got " + std::to_string(state->length)
        );
    }

    if (pairs->dimensions()[0] != expected_length || pairs->dimensions()[1] != 2) {
        return ffi::Error::Internal("`pairs` output buffer has the wrong shape");
    }

    if (shifts->dimensions()[0] != expected_length || shifts->dimensions()[1] != 3) {
        return ffi::Error::Internal("`shifts` output buffer has the wrong shape");
    }

    copy_pairs(*state, pairs->typed_data(), shifts->typed_data());

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    vesin_jax_compute,
    vesin_jax_compute_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<StatePointer>()
        .Ret<ffi::BufferR2<ffi::S64>>()
        .Ret<ffi::BufferR2<ffi::S32>>()
        .Attr<bool>("periodic_x")
        .Attr<bool>("periodic_y")
        .Attr<bool>("periodic_z")
        .Attr<double>("cutoff")
        .Attr<bool>("full")
        .Attr<bool>("sorted")
        .Attr<int64_t>("padded_to")
        .Attr<double>("skin")
        .Attr<int64_t>("algorithm")
        .Attr<int64_t>("n_threads")
);

static ffi::Error vesin_jax_compute_dynamic_impl(
    ffi::BufferR2<ffi::F64> points,
    ffi::BufferR2<ffi::F64> box,
    const StatePointer& state_ptr,
    ffi::Result<ffi::BufferR1<ffi::S64>> n_pairs,
    bool periodic_x,
    bool periodic_y,
    bool periodic_z,
    double cutoff,
    bool full,
    bool sorted,
    double skin,
    int64_t algorithm,
    int64_t n_threads
) {
    auto error = check_inputs(points, box);
    if (error.failure()) {
        return error;
    }

    auto* state = extract_state(state_ptr);

    auto periodic = std::array<bool, 3>{periodic_x, periodic_y, periodic_z};
    error = run_vesin(
        points, box, periodic, cutoff, full, sorted, /*padded_to=*/0, skin,
        algorithm, n_threads, state
    );
    if (error.failure()) {
        return error;
    }

    // the neighbor list is kept inside the state, it will be copied to the
    // output buffers by a later call to `vesin_jax_copy_out`, once the number
    // of pairs is known
    n_pairs->typed_data()[0] = static_cast<int64_t>(state->length);

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    vesin_jax_compute_dynamic,
    vesin_jax_compute_dynamic_impl,
    ffi::Ffi::Bind()
        .Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<ffi::BufferR2<ffi::F64>>()
        .Arg<StatePointer>()
        .Ret<ffi::BufferR1<ffi::S64>>()
        .Attr<bool>("periodic_x")
        .Attr<bool>("periodic_y")
        .Attr<bool>("periodic_z")
        .Attr<double>("cutoff")
        .Attr<bool>("full")
        .Attr<bool>("sorted")
        .Attr<double>("skin")
        .Attr<int64_t>("algorithm")
        .Attr<int64_t>("n_threads")
);

static ffi::Error vesin_jax_copy_out_impl(
    const StatePointer& state_ptr,
    ffi::Result<ffi::BufferR2<ffi::S64>> pairs,
    ffi::Result<ffi::BufferR2<ffi::S32>> shifts
) {
    auto* state = extract_state(state_ptr);

    if (pairs->dimensions()[0] != state->length || pairs->dimensions()[1] != 2) {
        return ffi::Error::Internal("`pairs` output buffer has the wrong shape");
    }

    if (shifts->dimensions()[0] != state->length || shifts->dimensions()[1] != 3) {
        return ffi::Error::Internal("`shifts` output buffer has the wrong shape");
    }

    copy_pairs(*state, pairs->typed_data(), shifts->typed_data());

    return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    vesin_jax_copy_out,
    vesin_jax_copy_out_impl,
    ffi::Ffi::Bind()
        .Arg<StatePointer>()
        .Ret<ffi::BufferR2<ffi::S64>>()
        .Ret<ffi::BufferR2<ffi::S32>>()
);
