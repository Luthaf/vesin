#ifndef CUDA_PTR_REGISTRY_HPP
#define CUDA_PTR_REGISTRY_HPP

#include "cuda_ptr_manager.hpp"
#include <memory>
#include <mutex>
#include <unordered_map>

namespace vesin {
namespace cuda {

/// @brief Global registry to manage unique CudaPtrManager instances per
/// VesinNeighborList.
///
/// This class provides thread-safe access to CudaPtrManager instances. Each
/// `VesinNeighborList*` gets its own `CudaPtrManager`, ensuring that memory is
/// managed uniquely per neighbor list without modifying the `VesinNeighborList`
/// class itself.
class CudaPtrRegistry {
public:
  /// @brief Returns the CudaPtrManager associated with a given
  /// VesinNeighborList.
  ///        If none exists, a new one is created and stored.
  /// @param neighbors Pointer to a VesinNeighborList to get a manager for.
  /// @return Reference to the corresponding CudaPtrManager.
  static CudaPtrManager &get(VesinNeighborList *neighbors) {
    std::lock_guard<std::mutex> lock(mutex());

    // Lazily create a CudaPtrManager if it doesn't exist yet.
    auto &manager_ptr = registry()[neighbors];
    if (!manager_ptr) {
      manager_ptr = std::make_unique<CudaPtrManager>(neighbors);
    }
    return *manager_ptr;
  }

  /// @brief Removes the CudaPtrManager associated with the given
  /// VesinNeighborList.
  ///        This releases the memory manager and all associated device
  ///        allocations.
  /// @param neighbors Pointer to the VesinNeighborList whose manager should be
  /// removed.
  static void erase(VesinNeighborList *neighbors) {
    std::lock_guard<std::mutex> lock(mutex());
    registry().erase(neighbors);
  }

private:
  /// @brief Registry mapping VesinNeighborList pointers to their corresponding
  /// CudaPtrManagers.
  static std::unordered_map<VesinNeighborList *,
                            std::unique_ptr<CudaPtrManager>> &
  registry() {
    static std::unordered_map<VesinNeighborList *,
                              std::unique_ptr<CudaPtrManager>>
        instance;
    return instance;
  }

  static std::mutex &mutex() {
    static std::mutex m;
    return m;
  }
};

} // namespace cuda
} // namespace vesin

#endif // CUDA_PTR_REGISTRY_HPP
