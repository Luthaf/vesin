#include "simple_neighbourlist.cuh"
#include "vesin_cuda.hpp"
#include <cuda_runtime.h>

using namespace vesin::cuda;

#define WARP_SIZE 32
#define NWARPS 4

template <typename T>
void free_neighbors(VesinNeighborList<T>& neighbors) {

    assert(neighbors.device == VesinCUDA);

    if (!neighbors.vesin_manage_memory)
        return;

    if (neighbors.pairs)     cudaFree(neighbors.pairs);
    if (neighbors.shifts)    cudaFree(neighbors.shifts);
    if (neighbors.distances) cudaFree(neighbors.distances);
    if (neighbors.vectors)   cudaFree(neighbors.vectors);
    if (neighbors.length)    cudaFree(neighbors.length);

    neighbors = {};
}

template <typename T>
void ensure_neighborlist_capacity(VesinNeighborList<T>& neighbors, size_t required_capacity) {
    if (!neighbors.vesin_manage_memory)
        return;  // Skip if external system manages memory

    if (neighbors.capacity >= required_capacity)
        return;

    // Free old if needed
    if (neighbors.pairs)     cudaFree(neighbors.pairs);
    if (neighbors.shifts)    cudaFree(neighbors.shifts);
    if (neighbors.distances) cudaFree(neighbors.distances);
    if (neighbors.vectors)   cudaFree(neighbors.vectors);
    if (neighbors.length)    cudaFree(neighbors.length);

    // Allocate new
    cudaMalloc(&neighbors.pairs,     sizeof(size_t) * required_capacity * 2);
    cudaMalloc(&neighbors.shifts,    sizeof(int32_t) * required_capacity * 3);
    cudaMalloc(&neighbors.distances, sizeof(T) * required_capacity);
    cudaMalloc(&neighbors.vectors,   sizeof(T) * required_capacity * 3);
    cudaMalloc(&neighbors.length,    sizeof(size_t));
    cudaMemset(neighbors.length, 0, sizeof(size_t));

    neighbors.capacity = required_capacity;
}

template <typename T>
void neighbors(
    const T (*points)[3],          // [n_points][3] on device
    size_t n_points,
    const T cell[3][3],            // [3][3] on device
    VesinOptions options,
    VesinNeighborList& neighbors,   // outputs already allocated on device
    int cuda_stream
) {

    ensure_neighborlist_capacity<T>(neighbors, n_points * MAX_NEDGES_PER_NODE);

    const T* d_positions = reinterpret_cast<const T*>(points);
    const T* d_cell = reinterpret_cast<const T*>(cell);

    size_t* d_edge_indices =  reinterpret_cast<const T*>(neighbors.pairs);
    int32_t* d_shifts = reinterpret_cast<int32_t*>(neighbors.shifts);
    T* d_distances = reinterpret_cast<T*>(neighbors.distances);
    T* d_vectors = reinterpret_cast<T*>(neighbors.vectors);
    size_t* d_pair_counter = neighbors.length; // TODO need to make sure this is allocated on device

    
    // Configure kernel launch
    dim3 blockDim(WARP_SIZE, NWARPS); // 32 threads per warp, NWARPS warps per block
    dim3 gridDim((n_points + NWARPS - 1) / NWARPS); // enough blocks to cover all atoms

    compute_neighbours_cell_device<T><<<gridDim, blockDim, 0, cuda_stream>>>(
        d_positions,
        d_cell,
        static_cast<int>(n_points),
        options.cutoff,
        d_pair_counter,
        d_edge_indices,
        d_shifts
        /* full_list = true */ 
    );
}