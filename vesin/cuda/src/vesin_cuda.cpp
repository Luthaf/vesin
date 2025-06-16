#include "mic_neighbourlist.cuh"
#include "vesin_cuda.hpp"

#include <cuda_runtime.h>
#include <stdexcept>
#include <cassert>
#include <iostream>

using namespace vesin::cuda;

#define WARP_SIZE 32
#define NWARPS 4

static void ensure_is_device_pointer(const void* p, const char* name) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, p);
    if (err != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaPointerGetAttributes failed for " ) + name +
            ": " + cudaGetErrorString(err)
        );
    }

    if (attr.type != cudaMemoryTypeDevice) {
        throw std::runtime_error(
            std::string(name) + " is not a device pointer (type=" +
            std::to_string(attr.type) + ")"
        );
    }

}


void vesin::cuda::free_neighbors(VesinNeighborList& neighbors) {

    assert(neighbors.device == VesinCUDA);


    if (neighbors.pairs)     cudaFree(neighbors.pairs);
    if (neighbors.shifts)    cudaFree(neighbors.shifts);
    if (neighbors.distances) cudaFree(neighbors.distances);
    if (neighbors.vectors)   cudaFree(neighbors.vectors); 
    if (neighbors.length_cu)    cudaFree(neighbors.length_cu);

    neighbors = {};
}

bool update_neighbourlist_capacity(VesinNeighborList& neighbors, size_t nnodes) {
    
    assert(neighbors.device == VesinCUDA);

    if (neighbors.capacity_cu >= nnodes)
        return false;

    // Free old if needed
    if (neighbors.pairs)     cudaFree(neighbors.pairs);
    if (neighbors.shifts)    cudaFree(neighbors.shifts);
    if (neighbors.distances) cudaFree(neighbors.distances);
    if (neighbors.vectors)   cudaFree(neighbors.vectors);
    if (neighbors.length_cu)    cudaFree(neighbors.length_cu);

    size_t nax_nedges = (size_t) 1.2 * nnodes * VESIN_CUDA_MAX_NEDGES_PER_NODE;

    // Allocate more than we need so we're not reallocating frequently
    cudaMalloc(&neighbors.pairs,     sizeof(size_t) * nax_nedges * 2);
    cudaMalloc(&neighbors.shifts,    sizeof(int32_t) * nax_nedges * 3);
    cudaMalloc(&neighbors.distances, sizeof(double) * nax_nedges);
    cudaMalloc(&neighbors.vectors,   sizeof(double) * nax_nedges * 3);
    cudaMalloc(&neighbors.length_cu,  sizeof(size_t));
    cudaMemset(neighbors.length_cu, 0, sizeof(size_t));

    neighbors.capacity_cu = (size_t) 1.2 * nnodes;

    return true;
}

void vesin::cuda::neighbors(
    const double (*points)[3],
    long n_points,
    const double cell[3][3],
    VesinOptions options,
    VesinNeighborList& neighbors
) {

    assert(neighbors.device == VesinCUDA);
    assert(!neighbors.sort && "Sorting is not supported in CUDA version of Vesin");

    if (!update_neighbourlist_capacity(neighbors, n_points * VESIN_CUDA_MAX_NEDGES_PER_NODE)) {
        // If we didn't reallocate, we need to reset the length
        cudaMemset(neighbors.length_cu, 0, sizeof(size_t));
    }  

    const double* d_positions = reinterpret_cast<const double*>(points);
    const double* d_cell = reinterpret_cast<const double*>(cell);

    unsigned long* d_edge_indices =  reinterpret_cast<unsigned long*>(neighbors.pairs);
    int* d_shifts = reinterpret_cast<int*>(neighbors.shifts);
    double* d_distances = reinterpret_cast<double*>(neighbors.distances);
    double* d_vectors = reinterpret_cast<double*>(neighbors.vectors);
    unsigned long* d_pair_counter = neighbors.length_cu;

    // --- BEGIN DEVICE-PTR CHECKS ---
    ensure_is_device_pointer(d_positions,     "points");
    ensure_is_device_pointer(d_cell,          "cell");
    ensure_is_device_pointer(d_edge_indices,  "neighbors.pairs");
    ensure_is_device_pointer(d_shifts,        "neighbors.shifts");
    ensure_is_device_pointer(d_distances,     "neighbors.distances");
    ensure_is_device_pointer(d_vectors,       "neighbors.vectors");
    ensure_is_device_pointer(d_pair_counter,  "neighbors.length_cu");
    // --- END DEVICE-PTR CHECKS ---

    vesin::cuda::compute_mic_neighbourlist<double>(
                                 d_positions,
                                 d_cell, 
                                 n_points,
                                 (double) options.cutoff,
                                 d_pair_counter,
                                 d_edge_indices, 
                                 d_shifts,
                                 d_distances,
                                 d_vectors,
                                 options.return_shifts, 
                                 options.return_distances,
                                 options.return_vectors,  
                                 options.full);

}