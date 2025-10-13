#include <sstream>

#include <torch/torch.h>

#include <vesin.h>

#include "vesin_torch.hpp"

using namespace vesin_torch;

static VesinDevice torch_to_vesin_device(torch::Device device) {
    if (device.is_cpu()) {
        return {VesinCPU, 0};
    } else if (device.is_cuda()) {
        return {VesinCUDA, device.index()};
    } else {
        throw std::runtime_error("device " + device.str() + " is not supported in vesin");
    }
}

static torch::Device vesin_to_torch_device(VesinDevice device) {
    if (device.type == VesinCPU) {
        return torch::Device(torch::kCPU);
    } else if (device.type == VesinCUDA) {
        return torch::Device(torch::kCUDA, device.device_id);
    } else {
        throw std::runtime_error("vesin device is not supported in torch");
    }
}

/// Custom autograd function that only registers a custom backward corresponding
/// to the neighbors list calculation
class AutogradNeighbors: public torch::autograd::Function<AutogradNeighbors> {
public:
    static std::vector<torch::Tensor> forward(
        torch::autograd::AutogradContext* ctx,
        torch::Tensor points,
        torch::Tensor box,
        torch::Tensor periodic,
        torch::Tensor pairs,
        torch::optional<torch::Tensor> shifts,
        torch::optional<torch::Tensor> distances,
        torch::optional<torch::Tensor> vectors
    );

    static std::vector<torch::Tensor> backward(
        torch::autograd::AutogradContext* ctx,
        std::vector<torch::Tensor> outputs_grad
    );
};

NeighborListHolder::NeighborListHolder(double cutoff, bool full_list, bool sorted):
    cutoff_(cutoff),
    full_list_(full_list),
    sorted_(sorted),
    data_(nullptr) {
    data_ = new VesinNeighborList();
}

NeighborListHolder::~NeighborListHolder() {
    vesin_free(data_);
    delete data_;
}

std::vector<torch::Tensor> NeighborListHolder::compute(
    torch::Tensor points,
    torch::Tensor box,
    torch::Tensor periodic,
    std::string quantities,
    bool copy
) {
    // check input data
    if (points.device() != box.device()) {
        // clang-format off
        C10_THROW_ERROR(ValueError,
            "expected `points` and `box` to have the same device, got " +
            points.device().str() + " and " + box.device().str()
        );
        // clang-format on
    }
    auto vesin_device = torch_to_vesin_device(points.device());

    if (points.scalar_type() != box.scalar_type()) {
        // clang-format off
        C10_THROW_ERROR(ValueError,
            std::string("expected `points` and `box` to have the same dtype, got ") +
            torch::toString(points.scalar_type()) + " and " +
            torch::toString(box.scalar_type())
        );
        // clang-format on
    }
    if (points.scalar_type() != torch::kFloat64) {
        C10_THROW_ERROR(ValueError, "only float64 dtype is supported in vesin");
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

    if (periodic.sizes().size() != 1 || periodic.size(0) != 3 || periodic.scalar_type() != torch::kBool) {
        C10_THROW_ERROR(ValueError, "`periodic` must be a 1D tensor of 3 booleans");
    }
    auto periodic_cpu = periodic;
    if (!periodic_cpu.device().is_cpu()) {
        periodic_cpu = periodic_cpu.to(torch::kCPU);
    }
    if (!periodic_cpu.is_contiguous()) {
        periodic_cpu = periodic_cpu.contiguous();
    }
    const bool* periodic_data = periodic_cpu.data_ptr<bool>();

    auto any_periodic = periodic_data[0] || periodic_data[1] || periodic_data[2];
    if (!any_periodic) {
        box = torch::zeros({3, 3}, points.options());
    }

    // create calculation options
    auto n_points = static_cast<size_t>(points.size(0));

    if (data_->device.type != VesinUnknownDevice && data_->device.type != vesin_device.type) {
        vesin_free(data_);
        std::memset(data_, 0, sizeof(VesinNeighborList));
    }

    auto return_shifts = quantities.find('S') != std::string::npos;
    if (box.requires_grad()) {
        return_shifts = true;
    }

    auto return_distances = quantities.find('d') != std::string::npos;
    auto return_vectors = quantities.find('D') != std::string::npos;
    if ((points.requires_grad() || box.requires_grad()) &&
        (return_distances || return_vectors)) {
        // gradients requires both distances & vectors data to be present
        return_distances = true;
        return_vectors = true;
    }

    auto options = VesinOptions{
        /*cutoff=*/this->cutoff_,
        /*full=*/this->full_list_,
        /*sorted=*/this->sorted_,
        /*return_shifts=*/return_shifts,
        /*return_distances=*/return_distances,
        /*return_vectors=*/return_vectors,
    };

    if (!points.is_contiguous()) {
        points = points.contiguous();
    }

    if (!box.is_contiguous()) {
        box = box.contiguous();
    }

    const char* error_message = nullptr;
    auto status = vesin_neighbors(
        reinterpret_cast<const double (*)[3]>(points.data_ptr<double>()), n_points, reinterpret_cast<const double (*)[3]>(box.data_ptr<double>()), periodic_data, vesin_device, options, data_, &error_message
    );

    if (status != EXIT_SUCCESS) {
        throw std::runtime_error(std::string("failed to compute neighbors: ") + error_message);
    }

    // wrap vesin data in tensors
    auto size_t_options =
        torch::TensorOptions().device(vesin_to_torch_device(data_->device));
    if (sizeof(size_t) == sizeof(uint32_t)) {
        size_t_options = size_t_options.dtype(torch::kUInt32);
    } else if (sizeof(size_t) == sizeof(uint64_t)) {
        size_t_options = size_t_options.dtype(torch::kUInt64);
    } else {
        C10_THROW_ERROR(ValueError, "could not determine torch dtype matching size_t");
    }

    int64_t length = static_cast<int64_t>(data_->length);
    auto pairs = torch::from_blob(data_->pairs, {length, 2}, size_t_options)
                     .to(torch::kInt64);

    auto shifts = torch::Tensor();
    if (data_->shifts != nullptr) {
        auto int32_options = torch::TensorOptions()
                                 .device(vesin_to_torch_device(data_->device))
                                 .dtype(torch::kInt32);

        shifts = torch::from_blob(data_->shifts, {length, 3}, int32_options);

        if (copy) {
            shifts = shifts.clone();
        }
    }

    auto double_options = torch::TensorOptions()
                              .device(vesin_to_torch_device(data_->device))
                              .dtype(torch::kDouble);

    auto distances = torch::Tensor();
    if (data_->distances != nullptr) {
        distances = torch::from_blob(data_->distances, {length}, double_options);

        if (copy) {
            distances = distances.clone();
        }
    }

    auto vectors = torch::Tensor();
    if (data_->vectors != nullptr) {
        vectors = torch::from_blob(data_->vectors, {length, 3}, double_options);

        if (copy) {
            vectors = vectors.clone();
        }
    }

    // handle autograd
    if ((return_distances || return_vectors)) {
        // we use optional for these three because otherwise torch autograd
        // tries to access data inside the undefined `torch::Tensor()`.
        torch::optional<torch::Tensor> shifts_optional = torch::nullopt;
        if (shifts.defined()) {
            shifts_optional = shifts;
        }

        torch::optional<torch::Tensor> distances_optional = torch::nullopt;
        if (distances.defined()) {
            distances_optional = distances;
        }

        torch::optional<torch::Tensor> vectors_optional = torch::nullopt;
        if (vectors.defined()) {
            vectors_optional = vectors;
        }

        auto outputs =
            AutogradNeighbors::apply(points, box, periodic, pairs, shifts_optional, distances_optional, vectors_optional);

        if (return_distances && return_vectors) {
            distances = outputs[0];
            vectors = outputs[1];
        } else if (return_distances) {
            distances = outputs[0];
        } else {
            assert(return_vectors);
            vectors = outputs[0];
        }
    }

    // assemble the output
    auto output = std::vector<torch::Tensor>();
    for (auto c : quantities) {
        if (c == 'i') {
            output.push_back(pairs.index({torch::indexing::Slice(), 0}));
        } else if (c == 'j') {
            output.push_back(pairs.index({torch::indexing::Slice(), 1}));
        } else if (c == 'P') {
            output.push_back(pairs);
        } else if (c == 'S') {
            output.push_back(shifts);
        } else if (c == 'd') {
            output.push_back(distances);
        } else if (c == 'D') {
            output.push_back(vectors);
        } else {
            C10_THROW_ERROR(ValueError, "unexpected character in `quantities`: " + std::string(1, c));
        }
    }

    return output;
}

TORCH_LIBRARY(vesin, m) {
    std::string DOCSTRING;

    // clang-format off
    m.class_<NeighborListHolder>("_NeighborList")
        .def(
            torch::init<double, bool, bool>(), DOCSTRING,
            {torch::arg("cutoff"), torch::arg("full_list"), torch::arg("sorted") = false}
        )
        .def("compute", &NeighborListHolder::compute, DOCSTRING,
            {torch::arg("points"), torch::arg("box"), torch::arg("periodic"), torch::arg("quantities"), torch::arg("copy") = true}
        )
        ;
    // clang-format on
}

// ========================================================================== //
//                                                                            //
// ========================================================================== //

std::vector<torch::Tensor> AutogradNeighbors::forward(
    torch::autograd::AutogradContext* ctx,
    torch::Tensor points,
    torch::Tensor box,
    torch::Tensor periodic,
    torch::Tensor pairs,
    torch::optional<torch::Tensor> shifts,
    torch::optional<torch::Tensor> distances,
    torch::optional<torch::Tensor> vectors
) {
    auto shifts_tensor = shifts.value_or(torch::Tensor());
    auto distances_tensor = distances.value_or(torch::Tensor());
    auto vectors_tensor = vectors.value_or(torch::Tensor());

    ctx->save_for_backward(
        {points, box, periodic, pairs, shifts_tensor, distances_tensor, vectors_tensor}
    );

    auto return_distances = distances.has_value();
    auto return_vectors = vectors.has_value();
    ctx->saved_data["return_distances"] = return_distances;
    ctx->saved_data["return_vectors"] = return_vectors;

    // only return defined tensors to make sure torch can use `get_autograd_meta()`
    if (return_distances && return_vectors) {
        return {distances_tensor, vectors_tensor};
    } else if (return_distances) {
        return {distances_tensor};
    } else if (return_vectors) {
        return {vectors_tensor};
    } else {
        return {};
    }
}

std::vector<torch::Tensor> AutogradNeighbors::backward(
    torch::autograd::AutogradContext* ctx,
    std::vector<torch::Tensor> outputs_grad
) {
    auto saved_variables = ctx->get_saved_variables();
    auto points = saved_variables[0];
    auto box = saved_variables[1];
    auto periodic = saved_variables[2];

    auto pairs = saved_variables[3];
    auto shifts = saved_variables[4];
    auto distances = saved_variables[5];
    auto vectors = saved_variables[6];

    auto return_distances = ctx->saved_data["return_distances"].toBool();
    auto return_vectors = ctx->saved_data["return_vectors"].toBool();

    auto distances_grad = torch::Tensor();
    auto vectors_grad = torch::Tensor();
    if (return_distances && return_vectors) {
        distances_grad = outputs_grad[0];
        vectors_grad = outputs_grad[1];
    } else if (return_distances) {
        distances_grad = outputs_grad[0];
    } else if (return_vectors) {
        vectors_grad = outputs_grad[0];
    } else {
        // nothing to do
        return {
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
            torch::Tensor(),
        };
    }

    if (points.requires_grad() || box.requires_grad()) {
        // Do a first backward step from distances_grad to vectors_grad
        vectors_grad += distances_grad.index({torch::indexing::Slice(), torch::indexing::None}) * vectors / distances.index({torch::indexing::Slice(), torch::indexing::None});
    }

    auto points_grad = torch::Tensor();
    if (points.requires_grad()) {
        points_grad = torch::zeros_like(points);
        points_grad =
            torch::index_add(points_grad,
                             /*dim=*/0,
                             /*index=*/pairs.index({torch::indexing::Slice(), 1}),
                             /*source=*/vectors_grad,
                             /*alpha=*/1.0);
        points_grad =
            torch::index_add(points_grad,
                             /*dim=*/0,
                             /*index=*/pairs.index({torch::indexing::Slice(), 0}),
                             /*source=*/vectors_grad,
                             /*alpha=*/-1.0);
    }

    auto box_grad = torch::Tensor();
    auto periodic_cpu = periodic;
    if (!periodic_cpu.device().is_cpu()) {
        periodic_cpu = periodic_cpu.to(torch::kCPU);
    }
    if (!periodic_cpu.is_contiguous()) {
        periodic_cpu = periodic_cpu.contiguous();
    }
    const bool* periodic_data = periodic_cpu.data_ptr<bool>();
    bool any_periodic = periodic_data[0] || periodic_data[1] || periodic_data[2];
    if (any_periodic && box.requires_grad()) {
        auto cell_shifts = shifts.to(box.scalar_type());
        box_grad = cell_shifts.t().matmul(vectors_grad);
    }

    return {
        points_grad,
        box_grad,
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
        torch::Tensor(),
    };
}
