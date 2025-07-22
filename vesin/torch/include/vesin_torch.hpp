#ifndef VESIN_TORCH_HPP
#define VESIN_TORCH_HPP

#include <torch/data.h>

struct VesinNeighborList;

namespace vesin_torch {

class NeighborListHolder;

/// `NeighborListHolder` should be manipulated through a `torch::intrusive_ptr`
using NeighborList = torch::intrusive_ptr<NeighborListHolder>;

/// Neighbor list calculator compatible with TorchScript
class NeighborListHolder: public torch::CustomClassHolder {
public:
    /// Create a new calculator with the given `cutoff`.
    ///
    /// @param full_list whether pairs should be included twice in the output
    ///                  (both as `i-j` and `j-i`) or only once
    /// @param sorted whether pairs should be sorted in the output
    NeighborListHolder(double cutoff, bool full_list, bool sorted = false);
    ~NeighborListHolder();

    /// Compute the neighbor list for the system defined by `positions`, `box`,
    /// and `periodic`; returning the requested `quantities`.
    ///
    /// `quantities` can contain any combination of the following values:
    ///
    ///     - `"i"` to get the index of the first point in the pair
    ///     - `"j"` to get the index of the second point in the pair
    ///     - `"P"` to get the indexes of the two points in the pair simultaneously
    ///     - `"S"` to get the periodic shift of the pair
    ///     - `"d"` to get the distance between points in the pair
    ///     - `"D"` to get the distance vector between points in the pair
    ///
    /// @param points positions of all points in the system
    /// @param box bounding box of the system
    /// @param periodic should we use periodic boundary conditions?
    /// @param quantities quantities to return, defaults to "ij"
    /// @param copy should we copy the returned quantities, defaults to `true`.
    ///        Setting this to `False` might be a bit faster, but the returned
    ///        tensors are view inside this class, and will be invalidated
    ///        whenever this class is garbage collected or used to run a new
    ///        calculation.
    ///
    /// @returns a list of `torch::Tensor` as indicated by `quantities`.
    std::vector<torch::Tensor> compute(
        torch::Tensor points,
        torch::Tensor box,
        bool periodic,
        std::string quantities,
        bool copy = true
    );

private:
    double cutoff_;
    bool full_list_;
    bool sorted_;
    VesinNeighborList* data_;
};

} // namespace vesin_torch

#endif
