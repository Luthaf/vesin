#ifndef VESIN_TORCH_HPP
#define VESIN_TORCH_HPP

#include <torch/data.h>

#include <array>

// clang-format off
#if defined(VESIN_TORCH_EXPORTS)
    #if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
        #define VESIN_TORCH_API __attribute__((visibility("default")))
    #elif defined(_MSC_VER)
        #define VESIN_TORCH_API __declspec(dllexport)
    #else
        #define VESIN_TORCH_API
    #endif
#else
    #if defined(__clang__) || defined(__GNUC__) || defined(__GNUG__)
        #define VESIN_TORCH_API __attribute__((visibility("default")))
    #elif defined(_MSC_VER)
        #define VESIN_TORCH_API __declspec(dllimport)
    #else
        #define VESIN_TORCH_API
    #endif
#endif
// clang-format on

struct VesinNeighborList;

namespace vesin_torch {

class NeighborListHolder;

/// `NeighborListHolder` should be manipulated through a `torch::intrusive_ptr`
using NeighborList = torch::intrusive_ptr<NeighborListHolder>;

/// Neighbor list calculator compatible with TorchScript
class VESIN_TORCH_API NeighborListHolder: public torch::CustomClassHolder {
public:
    /// Create a new calculator with the given `cutoff`.
    ///
    /// @param cutoff the spherical cutoff radius
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
    /// @param periodic per-axis periodic boundary condition mask
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
        torch::Tensor periodic,
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
