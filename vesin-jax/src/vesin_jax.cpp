#include <cstdint>
#include <cstdlib>

#include <algorithm>
#include <limits>
#include <map>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

#include <gpulite/gpulite.hpp>
#include <xla/ffi/api/ffi.h>

#include <vesin.h>

namespace ffi = xla::ffi;

// The FFI handlers are the only symbols exported by this library
#if defined(_WIN32)
    #define VESIN_JAX_EXPORT __declspec(dllexport)
#else
    #define VESIN_JAX_EXPORT __attribute__((visibility("default")))
#endif

// vesin writes the pairs as `size_t`, directly in buffers declared as `uint64`
static_assert(sizeof(size_t) == sizeof(uint64_t), "vesin-jax requires a 64-bit size_t");

namespace {

/// Pool of `VesinNeighborList`, re-using the memory allocated by vesin (scratch
/// buffers, Verlet cache, ...) across calls.
///
/// Entries are separated by state (one for each Python `NeighborList`, so each
/// gets its own Verlet cache), device and CUDA stream: vesin can return while
/// work is still queued on the stream, so an entry can only be re-used by later
/// work on the same stream. Entries are removed from the pool while in use.
class NeighborListPool {
public:
    using Key = std::tuple<int64_t, int32_t, int32_t, void*>;

    /// Get the global pool. It is leaked on purpose, so that it never releases
    /// CUDA memory after the CUDA libraries are unloaded at exit.
    static NeighborListPool& instance() {
        static auto* pool = new NeighborListPool();
        return *pool;
    }

    VesinNeighborList* acquire(const Key& key) {
        auto lock = std::lock_guard<std::mutex>(this->mutex_);
        auto& available = this->available_[key];
        if (available.empty()) {
            return new VesinNeighborList();
        }

        auto* neighbors = available.back();
        available.pop_back();
        return neighbors;
    }

    void release(const Key& key, VesinNeighborList* neighbors) {
        auto lock = std::lock_guard<std::mutex>(this->mutex_);
        if (this->freed_states_.count(std::get<0>(key)) != 0) {
            // the state was freed while this entry was in use
            free_neighbors(neighbors);
        } else {
            this->available_[key].push_back(neighbors);
        }
    }

    /// Release the memory used by the given state. Calculations can still use
    /// this state afterwards, but their entries are freed after each use.
    void free_state(int64_t state) {
        auto lock = std::lock_guard<std::mutex>(this->mutex_);
        this->freed_states_.insert(state);
        for (auto it = this->available_.begin(); it != this->available_.end();) {
            if (std::get<0>(it->first) == state) {
                for (auto* neighbors: it->second) {
                    free_neighbors(neighbors);
                }
                it = this->available_.erase(it);
            } else {
                ++it;
            }
        }
    }

private:
    static void free_neighbors(VesinNeighborList* neighbors) {
        vesin_free(neighbors);
        delete neighbors;
    }

    std::mutex mutex_;
    std::map<Key, std::vector<VesinNeighborList*>> available_;
    std::set<int64_t> freed_states_;
};

/// A `VesinNeighborList` taken from the pool for the duration of one calculation
class PooledNeighborList {
public:
    PooledNeighborList(int64_t state, VesinDevice device, void* stream):
        key_(state, device.type, device.device_id, stream),
        neighbors_(NeighborListPool::instance().acquire(key_)) {}

    ~PooledNeighborList() {
        NeighborListPool::instance().release(this->key_, this->neighbors_);
    }

    PooledNeighborList(const PooledNeighborList&) = delete;
    PooledNeighborList& operator=(const PooledNeighborList&) = delete;

    VesinNeighborList* get() {
        return this->neighbors_;
    }

private:
    NeighborListPool::Key key_;
    VesinNeighborList* neighbors_;
};

template <typename T>
T get_attribute(const ffi::Dictionary& attributes, std::string_view name) {
    auto value = attributes.get<T>(name);
    if (value.has_error()) {
        throw std::invalid_argument(
            "invalid attribute `" + std::string(name) + "`: " + value.error().message()
        );
    }
    return value.value();
}

VesinAlgorithm parse_algorithm(std::string_view algorithm) {
    if (algorithm == "auto") {
        return VesinAutoAlgorithm;
    } else if (algorithm == "brute_force") {
        return VesinBruteForce;
    } else if (algorithm == "cell_list") {
        return VesinCellList;
    } else {
        throw std::invalid_argument("unknown algorithm '" + std::string(algorithm) + "'");
    }
}

void check_shape(std::string_view name, ffi::Span<const int64_t> shape, std::vector<int64_t> expected) {
    auto matches = shape.size() == expected.size();
    for (size_t i = 0; matches && i < expected.size(); i++) {
        matches = expected[i] < 0 || shape[i] == expected[i];
    }

    if (!matches) {
        throw std::invalid_argument("`" + std::string(name) + "` has the wrong shape");
    }
}

/// Run the calculation, writing the pairs and shifts directly in the XLA
/// buffers. This returns the number of pairs in the neighbor list, which can be
/// larger than the capacity of the buffers; their content is then unspecified.
size_t compute_neighbors(
    VesinDevice device,
    void* stream,
    ffi::BufferR2<ffi::F64> points,
    ffi::BufferR2<ffi::F64> box,
    ffi::BufferR1<ffi::PRED> periodic,
    ffi::Result<ffi::BufferR2<ffi::U64>>& pairs,
    ffi::Result<ffi::BufferR2<ffi::S32>>& shifts,
    const ffi::Dictionary& attributes
) {
    check_shape("points", points.dimensions(), {-1, 3});
    check_shape("box", box.dimensions(), {3, 3});
    check_shape("periodic", periodic.dimensions(), {3});

    auto capacity = pairs->dimensions()[0];
    check_shape("pairs", pairs->dimensions(), {capacity, 2});
    check_shape("shifts", shifts->dimensions(), {capacity, 3});

    auto n_threads = get_attribute<int64_t>(attributes, "n_threads");
    if (n_threads < 0 || n_threads > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("invalid number of threads: " + std::to_string(n_threads));
    }

    auto options = VesinOptions();
    options.cutoff = get_attribute<double>(attributes, "cutoff");
    options.full = get_attribute<bool>(attributes, "full_list");
    options.sorted = get_attribute<bool>(attributes, "sorted");
    options.algorithm = parse_algorithm(get_attribute<std::string_view>(attributes, "algorithm"));
    options.skin = get_attribute<double>(attributes, "skin");
    options.n_threads = static_cast<int32_t>(n_threads);
    options.return_shifts = true;
    options.pre_allocated_outputs = true;
    options.cuda_stream = stream;

    auto state = get_attribute<int64_t>(attributes, "state");
    auto pooled = PooledNeighborList(state, device, stream);
    auto* neighbors = pooled.get();
    neighbors->pairs = reinterpret_cast<size_t (*)[2]>(pairs->typed_data());
    neighbors->shifts = reinterpret_cast<int32_t (*)[3]>(shifts->typed_data());
    neighbors->length = static_cast<size_t>(capacity);

    const char* error_message = nullptr;
    auto status = vesin_neighbors(
        reinterpret_cast<const double (*)[3]>(points.typed_data()),
        static_cast<size_t>(points.dimensions()[0]),
        reinterpret_cast<const double (*)[3]>(box.typed_data()),
        periodic.typed_data(),
        device,
        options,
        neighbors,
        &error_message
    );

    // the XLA buffers must not stay in the pool
    neighbors->pairs = nullptr;
    neighbors->shifts = nullptr;

    if (status != EXIT_SUCCESS && status != VESIN_OUTPUT_TOO_SMALL) {
        throw std::runtime_error(error_message);
    }

    return neighbors->length;
}

int32_t clamp_to_int32(size_t value) {
    return static_cast<int32_t>(std::min(value, static_cast<size_t>(std::numeric_limits<int32_t>::max())));
}

/// Run `function`, translating exceptions to FFI errors
template <typename Function>
ffi::Error catch_exceptions(Function function) {
    try {
        function();
        return ffi::Error::Success();
    } catch (const std::invalid_argument& e) {
        return ffi::Error::InvalidArgument(std::string("vesin: ") + e.what());
    } catch (const std::exception& e) {
        return ffi::Error::Internal(std::string("vesin: ") + e.what());
    }
}

ffi::Error neighbors_cpu(
    ffi::BufferR2<ffi::F64> points,
    ffi::BufferR2<ffi::F64> box,
    ffi::BufferR1<ffi::PRED> periodic,
    ffi::Result<ffi::BufferR2<ffi::U64>> pairs,
    ffi::Result<ffi::BufferR2<ffi::S32>> shifts,
    ffi::Result<ffi::BufferR0<ffi::S32>> n_pairs,
    ffi::Dictionary attributes
) {
    return catch_exceptions([&]() {
        auto length = compute_neighbors(
            {VesinCPU, 0}, nullptr, points, box, periodic, pairs, shifts, attributes
        );
        *n_pairs->typed_data() = clamp_to_int32(length);
    });
}

ffi::Error neighbors_cuda(
    cudaStream_t stream,
    ffi::BufferR2<ffi::F64> points,
    ffi::BufferR2<ffi::F64> box,
    ffi::BufferR1<ffi::PRED> periodic,
    ffi::Result<ffi::BufferR2<ffi::U64>> pairs,
    ffi::Result<ffi::BufferR2<ffi::S32>> shifts,
    ffi::Result<ffi::BufferR0<ffi::S32>> n_pairs,
    ffi::Dictionary attributes
) {
    return catch_exceptions([&]() {
        cudaPointerAttributes pointer;
        GPULITE_CUDART_CALL(cudaPointerGetAttributes(&pointer, points.untyped_data()));
        auto device = VesinDevice{VesinCUDA, pointer.device};

        auto length = clamp_to_int32(compute_neighbors(
            device, stream, points, box, periodic, pairs, shifts, attributes
        ));

        // `length` is in pageable host memory, so CUDA copies it to a staging
        // area before this call returns
        GPULITE_CUDART_CALL(cudaMemcpyAsync(
            n_pairs->untyped_data(), &length, sizeof(int32_t), cudaMemcpyHostToDevice, stream
        ));
    });
}

/// Add the arguments, results and attributes of the neighbors handlers to
/// `binding`
template <typename Binding>
auto bind_neighbors(Binding binding) {
    return std::move(binding)
        .template Arg<ffi::BufferR2<ffi::F64>>()  // points
        .template Arg<ffi::BufferR2<ffi::F64>>()  // box
        .template Arg<ffi::BufferR1<ffi::PRED>>() // periodic
        .template Ret<ffi::BufferR2<ffi::U64>>()  // pairs
        .template Ret<ffi::BufferR2<ffi::S32>>()  // shifts
        .template Ret<ffi::BufferR0<ffi::S32>>()  // n_pairs
        .Attrs();
}

} // namespace

extern "C" VESIN_JAX_EXPORT XLA_FFI_Error* vesin_jax_neighbors_cpu(XLA_FFI_CallFrame* call_frame);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    vesin_jax_neighbors_cpu,
    neighbors_cpu,
    bind_neighbors(ffi::Ffi::Bind())
);

extern "C" VESIN_JAX_EXPORT XLA_FFI_Error* vesin_jax_neighbors_cuda(XLA_FFI_CallFrame* call_frame);
XLA_FFI_DEFINE_HANDLER_SYMBOL(
    vesin_jax_neighbors_cuda,
    neighbors_cuda,
    bind_neighbors(ffi::Ffi::Bind().Ctx<ffi::PlatformStream<cudaStream_t>>())
);

/// Release the memory associated with the state of a `NeighborList`, called
/// when the corresponding Python object is garbage-collected
extern "C" VESIN_JAX_EXPORT void vesin_jax_free_state(int64_t state) {
    try {
        NeighborListPool::instance().free_state(state);
    } catch (...) {
        // nothing to do, the memory is leaked
    }
}
