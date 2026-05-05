#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <sstream>
#include <string>
#include <tuple>

#include <torch/torch.h>

#include <vesin.h>

namespace {

VesinDevice torch_to_vesin_device(torch::Device device) {
    if (device.is_cpu()) {
        return {VesinCPU, 0};
    } else if (device.is_cuda()) {
        return {VesinCUDA, device.index()};
    } else {
        throw std::runtime_error("device " + device.str() + " is not supported in vesin");
    }
}

torch::Device vesin_to_torch_device(VesinDevice device) {
    if (device.type == VesinCPU) {
        return torch::Device(torch::kCPU);
    } else if (device.type == VesinCUDA) {
        return torch::Device(torch::kCUDA, static_cast<int8_t>(device.device_id));
    } else {
        throw std::runtime_error("vesin device is not supported in torch");
    }
}

/// State of the neighbor list calculation, kept between calls. This contains
/// the memory allocated by vesin, which is re-used across calls, as well as the
/// cached neighbor list when using a skin.
///
/// This is registered as a TorchBind class, which allows it to be stored in
/// TorchScript and `torch.export` models.
class State: public torch::CustomClassHolder {
public:
    State() = default;
    ~State() override {
        vesin_free(&data_);
    }

    State(const State&) = delete;
    State& operator=(const State&) = delete;
    State(State&&) = delete;
    State& operator=(State&&) = delete;

    /// Memory allocated by vesin. This must only be used while holding the
    /// lock returned by `lock()`.
    VesinNeighborList* data() {
        return &data_;
    }

    /// Lock this state, preventing concurrent use (for example when the same
    /// model is executed from multiple threads). This emits a warning (once
    /// per process) if another thread is already using this state.
    std::unique_lock<std::mutex> lock() {
        auto lock = std::unique_lock<std::mutex>(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) {
            TORCH_WARN_ONCE(
                "the same vesin neighbor list is being used from multiple threads "
                "at the same time. The calculations will run one after the other, "
                "use a separate NeighborList (or a separate model) in each thread "
                "to run them in parallel."
            );
            lock.lock();
        }
        return lock;
    }

private:
    VesinNeighborList data_;
    std::mutex mutex_;
};

VesinAlgorithm algorithm_from_string(const std::string& algorithm) {
    if (algorithm == "auto") {
        return VesinAutoAlgorithm;
    } else if (algorithm == "brute_force") {
        return VesinBruteForce;
    } else if (algorithm == "cell_list") {
        return VesinCellList;
    } else {
        throw std::runtime_error(
            "unknown algorithm '" + algorithm + "', expected one of "
            "'auto', 'brute_force', or 'cell_list'"
        );
    }
}

/// Compute the neighbor list with the given `options`, without any autograd
/// integration. Pairs are always returned, the other outputs are empty tensors
/// if not requested in `options`.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> compute_neighbors(
    VesinNeighborList* data,
    torch::Tensor points,
    torch::Tensor box,
    torch::Tensor periodic,
    VesinOptions options,
    bool copy
) {
    // check input data
    if (points.device() != box.device()) {
        C10_THROW_ERROR(ValueError,
            "expected `points` and `box` to have the same device, got " +
            points.device().str() + " and " + box.device().str()
        );
    }
    auto vesin_device = torch_to_vesin_device(points.device());

    if (points.scalar_type() != box.scalar_type()) {
        C10_THROW_ERROR(ValueError,
            std::string("expected `points` and `box` to have the same dtype, got ") +
            torch::toString(points.scalar_type()) + " and " +
            torch::toString(box.scalar_type())
        );
    }

    if (points.scalar_type() != torch::kFloat64) {
        C10_THROW_ERROR(ValueError, "only float64 is supported for `points` and `box`");
    }

    if (points.sizes().size() != 2 || points.size(1) != 3) {
        std::ostringstream oss;
        oss << "`points` must be n x 3 tensor, but the shape is " << points.sizes();
        C10_THROW_ERROR(ValueError, oss.str());
    }

    if (box.sizes().size() != 2 || box.size(0) != 3 || box.size(1) != 3) {
        std::ostringstream oss;
        oss << "`box` must be 3 x 3 tensor, but the shape is " << points.sizes();
        C10_THROW_ERROR(ValueError, oss.str());
    }

    if (points.device() != periodic.device()) {
        periodic = periodic.to(points.device());
    }

    if (periodic.scalar_type() != torch::kBool) {
        C10_THROW_ERROR(ValueError, "`periodic` must be a tensor of bool");
    }

    if (periodic.sizes().size() == 0) {
        periodic = periodic.repeat({3});
    }

    if (periodic.sizes().size() != 1 || periodic.size(0) != 3) {
        C10_THROW_ERROR(ValueError, "`periodic` must be either a single bool or a tensor of 3 bool");
    }

    auto n_points = static_cast<size_t>(points.size(0));

    // reset vesin data if device changed
    if (data->device.type != VesinUnknownDevice && data->device.type != vesin_device.type) {
        vesin_free(data);
        std::memset(data, 0, sizeof(VesinNeighborList));
    }

    if (!points.is_contiguous()) {
        points = points.contiguous();
    }

    if (!box.is_contiguous()) {
        box = box.contiguous();
    }

    const char* error_message = nullptr;
    auto status = vesin_neighbors(
        reinterpret_cast<const double (*)[3]>(points.data_ptr<double>()),
        n_points,
        reinterpret_cast<const double (*)[3]>(box.data_ptr<double>()),
        periodic.data_ptr<bool>(),
        vesin_device,
        options,
        data,
        &error_message
    );

    if (status != EXIT_SUCCESS) {
        throw std::runtime_error(std::string("failed to compute neighbors: ") + error_message);
    }

    // wrap vesin data in tensors
    auto device = vesin_to_torch_device(data->device);
    auto size_t_options = torch::TensorOptions().device(device);
    if (sizeof(size_t) == sizeof(uint32_t)) {
        size_t_options = size_t_options.dtype(torch::kUInt32);
    } else if (sizeof(size_t) == sizeof(uint64_t)) {
        size_t_options = size_t_options.dtype(torch::kUInt64);
    } else {
        C10_THROW_ERROR(ValueError, "could not determine torch dtype matching size_t");
    }

    int64_t length = static_cast<int64_t>(data->length);
    auto pairs = torch::from_blob(data->pairs, {length, 2}, size_t_options)
                     .to(torch::kInt64);

    auto int32_options = torch::TensorOptions().device(device).dtype(torch::kInt32);
    auto shifts = torch::empty({0, 3}, int32_options);
    if (options.return_shifts) {
        shifts = torch::from_blob(data->shifts, {length, 3}, int32_options);
        if (copy) {
            shifts = shifts.clone();
        }
    }

    auto double_options = torch::TensorOptions().device(device).dtype(torch::kDouble);
    auto distances = torch::empty({0}, double_options);
    if (options.return_distances) {
        distances = torch::from_blob(data->distances, {length}, double_options);
        if (copy) {
            distances = distances.clone();
        }
    }

    auto vectors = torch::empty({0, 3}, double_options);
    if (options.return_vectors) {
        vectors = torch::from_blob(data->vectors, {length, 3}, double_options);
        if (copy) {
            vectors = vectors.clone();
        }
    }

    return {pairs, shifts, distances, vectors};
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> neighbors(
    c10::intrusive_ptr<State> state,
    torch::Tensor points,
    torch::Tensor box,
    torch::Tensor periodic,
    double cutoff,
    bool full_list,
    bool sorted,
    std::string algorithm,
    double skin,
    int64_t n_threads,
    bool return_shifts,
    bool return_distances,
    bool return_vectors,
    bool copy
) {
    if (n_threads == 0) {
        n_threads = static_cast<int64_t>(at::get_num_threads());
    }

    auto options = VesinOptions{
        /*cutoff=*/cutoff,
        /*full=*/full_list,
        /*sorted=*/sorted,
        /*algorithm=*/algorithm_from_string(algorithm),
        /*skin=*/skin,
        /*n_threads=*/static_cast<int32_t>(n_threads),
        /*return_shifts=*/return_shifts,
        /*return_distances=*/return_distances,
        /*return_vectors=*/return_vectors,
    };

    auto lock = state->lock();
    return compute_neighbors(
        state->data(),
        std::move(points),
        std::move(box),
        std::move(periodic),
        options,
        copy
    );
}

} // namespace

TORCH_LIBRARY(vesin, m) {
    m.class_<State>("State")
        .def(torch::init<>())
        // used by `torch.compile` and `torch.export` to create the fake version
        // of this class (see `_compile.py`). The state does not influence the
        // shape of the outputs, so there is nothing to inspect here.
        .def("__obj_flatten__", [](const c10::intrusive_ptr<State>&) {
            return std::tuple<>();
        })
        // the content of the state is not saved, loading a model creates a new
        // empty state instead. The integer is a version number for the
        // serialization format.
        .def_pickle(
            [](const c10::intrusive_ptr<State>&) -> int64_t {
                return 0;
            },
            [](int64_t) -> c10::intrusive_ptr<State> {
                return c10::make_intrusive<State>();
            }
        );

    m.def(
        "neighbors("
            "__torch__.torch.classes.vesin.State state, Tensor points, Tensor box, Tensor periodic, "
            "float cutoff, bool full_list, bool sorted, str algorithm, float skin, int n_threads, "
            "bool return_shifts, bool return_distances, bool return_vectors, bool copy"
        ") -> (Tensor pairs, Tensor shifts, Tensor distances, Tensor vectors)"
    );
}

TORCH_LIBRARY_IMPL(vesin, CPU, m) {
    m.impl("neighbors", neighbors);
}

TORCH_LIBRARY_IMPL(vesin, CUDA, m) {
    m.impl("neighbors", neighbors);
}

// `neighbors` is not differentiable, autograd is handled in Python by
// re-computing distances and vectors from pairs and shifts
TORCH_LIBRARY_IMPL(vesin, Autograd, m) {
    m.impl("neighbors", torch::CppFunction::makeFallthrough());
}
