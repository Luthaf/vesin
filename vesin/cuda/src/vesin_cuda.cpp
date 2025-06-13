#include "simple_neighbourlist.cuh"
#include "vesin_cuda.hpp"

#include <cuda_runtime.h>
#include <cassert>

using namespace vesin::cuda;

#define WARP_SIZE 32
#define NWARPS 4

void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {

    assert(neighbors.device == VesinCUDA);


    if (neighbors.pairs)     cudaFree(neighbors.pairs);
    if (neighbors.shifts)    cudaFree(neighbors.shifts);
    if (neighbors.distances) cudaFree(neighbors.distances);
    if (neighbors.vectors)   cudaFree(neighbors.vectors); 
    if (neighbors.length_cu)    cudaFree(neighbors.length_cu);

    neighbors = {};
}

void ensure_neighborlist_capacity(VesinNeighborList& neighbors, size_t nnodes) {
    
    assert(neighbors.device == VesinCUDA);

    if (neighbors.capacity_cu >= nnodes)
        return;

    // Free old if needed
    if (neighbors.pairs)     cudaFree(neighbors.pairs);
    if (neighbors.shifts)    cudaFree(neighbors.shifts);
    if (neighbors.distances) cudaFree(neighbors.distances);
    if (neighbors.vectors)   cudaFree(neighbors.vectors);
    if (neighbors.length_cu)    cudaFree(neighbors.length_cu);

    // Allocate new
    cudaMalloc(&neighbors.pairs,     sizeof(size_t) * nnodes * 2);
    cudaMalloc(&neighbors.shifts,    sizeof(int32_t) * nnodes * 3);
    cudaMalloc(&neighbors.distances, sizeof(double) * nnodes);
    cudaMalloc(&neighbors.vectors,   sizeof(double) * nnodes * 3);
    cudaMalloc(&neighbors.length_cu,  sizeof(size_t));
    cudaMemset(neighbors.length_cu, 0, sizeof(size_t));

    neighbors.capacity_cu = nnodes;
}

void vesin::cuda::neighbors(
    const double (*points)[3],          // [n_points][3] on device
    long n_points,
    const double cell[3][3],            // [3][3] on device
    VesinOptions options,
    VesinNeighborList& neighbors   // outputs already allocated on device
) {

    assert(neighbors.device == VesinCUDA);

    ensure_neighborlist_capacity(neighbors, n_points * VESIN_CUDA_MAX_NEDGES_PER_NODE);

    const double* d_positions = reinterpret_cast<const double*>(points);
    const double* d_cell = reinterpret_cast<const double*>(cell);

    unsigned long* d_edge_indices =  reinterpret_cast<unsigned long*>(neighbors.pairs);
    int* d_shifts = reinterpret_cast<int*>(neighbors.shifts);
    double* d_distances = reinterpret_cast<double*>(neighbors.distances);
    double* d_vectors = reinterpret_cast<double*>(neighbors.vectors);
    unsigned long* d_pair_counter = neighbors.length_cu;

    vesin::cuda::compute_simple_neighbourlist<double>(
                                 d_positions,
                                 d_cell, 
                                 n_points,
                                 (double) options.cutoff,
                                 d_pair_counter,
                                 d_edge_indices, 
                                 d_shifts);

}