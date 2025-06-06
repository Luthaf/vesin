namespace vesin {
namespace cuda {
/* wrapper for cases where cell data is on device. */
template <typename scalar_t>
__global__ void
compute_neighbours_cell_device(const scalar_t *positions, const scalar_t *cell,
                               int nnodes, scalar_t cutoff, int *pair_counter,
                               int *edge_indices, scalar_t *shifts);

} // namespace cuda
} // namespace vesin