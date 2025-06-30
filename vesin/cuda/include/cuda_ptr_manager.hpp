#ifndef CUDA_PTR_MANAGER_HPP
#define CUDA_PTR_MANAGER_HPP

#include "vesin.h"

namespace vesin {
namespace cuda {

/// @brief Manages GPU memory for a single VesinNeighborList instance.
///        This class allocates and frees device memory used by the neighbor
///        list, ensuring proper capacity and reallocation when needed.
class CudaPtrManager {
public:
  /// @brief Constructs the manager for a given VesinNeighborList.
  /// @param neighbors Pointer to a VesinNeighborList (must remain valid for the
  /// lifetime of the manager).
  explicit CudaPtrManager(VesinNeighborList *neighbors);

  /// @brief Frees all GPU memory owned by this manager.
  ~CudaPtrManager();

  /// @brief Ensures sufficient capacity for the given number of nodes.
  ///        If needed, re-allocates all device buffers.
  /// @param nnodes Number of nodes to allocate capacity for.
  void update_capacity(unsigned long nnodes);

  /// @brief Frees all currently allocated GPU buffers and resets internal
  /// state.
  void reset();

  /// @brief Returns the raw device pointer to the memory storing the neighbor
  /// list length.
  /// @return Device pointer to an `unsigned long` storing the neighbor list
  /// length.
  unsigned long *get_length_device_ptr() const;

private:
  /// @brief Pointer to the associated neighbor list (non-owning).
  VesinNeighborList *neighbors;

  /// @brief Device pointer storing the current number of edges in the neighbor
  /// list.
  unsigned long *length_ptr = nullptr;

  /// @brief Number of nodes currently supported by the allocated buffers.
  unsigned long capacity = 0;

  // Disable copying
  CudaPtrManager(const CudaPtrManager &) = delete;
  CudaPtrManager &operator=(const CudaPtrManager &) = delete;
};

} // namespace cuda
} // namespace vesin

#endif // CUDA_PTR_MANAGER_HPP
