#ifndef VESIN_OUTPUTS_HPP
#define VESIN_OUTPUTS_HPP

#include <stdexcept>
#include <string>

#include "vesin.h"

namespace vesin {

/// Pointers to the output arrays of a `VesinNeighborList`
struct OutputPointers {
    void* pairs = nullptr;
    void* shifts = nullptr;
    void* distances = nullptr;
    void* vectors = nullptr;

    /// Get the pointers currently in `neighbors`
    static OutputPointers from(const VesinNeighborList& neighbors) {
        return OutputPointers{
            static_cast<void*>(neighbors.pairs),
            static_cast<void*>(neighbors.shifts),
            static_cast<void*>(neighbors.distances),
            static_cast<void*>(neighbors.vectors),
        };
    }
};

/// Check the output arrays in `neighbors` at the start of a calculation.
/// `allocated` contains the arrays vesin allocated itself and left in
/// `neighbors` at the end of the previous calculation.
///
/// Without `options.pre_allocated_outputs`, the arrays must be exactly the
/// ones vesin left there. With it, all the arrays requested in `options` must
/// be set, and must not be arrays allocated by vesin.
inline void check_output_arrays(
    const VesinNeighborList& neighbors,
    const OutputPointers& allocated,
    VesinOptions options
) {
    auto current = OutputPointers::from(neighbors);
    const char* names[4] = {"pairs", "shifts", "distances", "vectors"};
    void* current_ptrs[4] = {current.pairs, current.shifts, current.distances, current.vectors};
    void* allocated_ptrs[4] = {allocated.pairs, allocated.shifts, allocated.distances, allocated.vectors};
    bool requested[4] = {true, options.return_shifts, options.return_distances, options.return_vectors};

    for (size_t i = 0; i < 4; i++) {
        auto name = std::string(names[i]);
        if (options.pre_allocated_outputs) {
            if (requested[i] && current_ptrs[i] == nullptr) {
                throw std::runtime_error(
                    "`options.pre_allocated_outputs` is set, but `neighbors." + name + "` is NULL"
                );
            }

            if (current_ptrs[i] != nullptr && current_ptrs[i] == allocated_ptrs[i]) {
                throw std::runtime_error(
                    "`neighbors." + name + "` was allocated by vesin, it can not "
                    "be used as a pre-allocated output array"
                );
            }
        } else if (current_ptrs[i] != allocated_ptrs[i]) {
            throw std::runtime_error(
                "`neighbors." + name + "` is not the array allocated by vesin in "
                "the last calculation. Output arrays can only be set by the caller "
                "with `options.pre_allocated_outputs`, and must be set back to NULL "
                "before calculations without it"
            );
        }
    }
}

/// Error for pre-allocated output arrays which are too small
inline std::runtime_error pre_allocated_too_small(size_t capacity, size_t n_pairs) {
    return std::runtime_error(
        "the pre-allocated output arrays can hold " + std::to_string(capacity) +
        " pairs, but this calculation needs " + std::to_string(n_pairs) + " pairs"
    );
}

} // namespace vesin


#endif // VESIN_OUTPUTS_HPP
