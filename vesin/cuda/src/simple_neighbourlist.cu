#define NWARPS 4
#ifndef MAX_NEIGHBOURS_PER_ATOM
#define MAX_NEIGHBOURS_PER_ATOM 1024 // Make configurable
#endif
template <typename scalar_t> struct Vector3IO;

/* template structure for dealing with float3, double3 vectorized types */
template <> struct Vector3IO<float> {
  using scalar_t = float;
  using vec_t = float3;

  __device__ static void unpack(const vec_t &v, scalar_t &x0, scalar_t &x1,
                                scalar_t &x2) {
    x0 = v.x;
    x1 = v.y;
    x2 = v.z;
  }

  __device__ static vec_t pack(scalar_t x0, scalar_t x1, scalar_t x2) {
    return {x0, x1, x2};
  }
};

template <> struct Vector3IO<double> {
  using scalar_t = double;
  using vec_t = double3;

  __device__ static void unpack(const vec_t &v, scalar_t &x0, scalar_t &x1,
                                scalar_t &x2) {
    x0 = v.x;
    x1 = v.y;
    x2 = v.z;
  }

  __device__ static vec_t pack(scalar_t x0, scalar_t x1, scalar_t x2) {
    return {x0, x1, x2};
  }
};

template <typename scalar_t>
__device__ typename Vector3IO<scalar_t>::vec_t
operator+(const typename Vector3IO<scalar_t>::vec_t &a,
          const typename Vector3IO<scalar_t>::vec_t &b) {
  return Vector3IO<scalar_t>::pack(a.x + b.x, a.y + b.y, a.z + b.z);
}

template <typename scalar_t>
__device__ typename Vector3IO<scalar_t>::vec_t
operator-(const typename Vector3IO<scalar_t>::vec_t &a,
          const typename Vector3IO<scalar_t>::vec_t &b) {
  return Vector3IO<scalar_t>::pack(a.x - b.x, a.y - b.y, a.z - b.z);
}

template <typename scalar_t>
__device__ scalar_t dot(const typename Vector3IO<scalar_t>::vec_t &a,
                        const typename Vector3IO<scalar_t>::vec_t &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

template <typename scalar_t>
__device__ void invert_cell_matrix(const scalar_t *cell, scalar_t *inv_cell) {
  scalar_t a = cell[0], b = cell[1], c = cell[2];
  scalar_t d = cell[3], e = cell[4], f = cell[5];
  scalar_t g = cell[6], h = cell[7], i = cell[8];

  scalar_t det =
      a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
  scalar_t invdet = scalar_t(1.0) / det;

  inv_cell[0] = (e * i - f * h) * invdet;
  inv_cell[1] = (c * h - b * i) * invdet;
  inv_cell[2] = (b * f - c * e) * invdet;
  inv_cell[3] = (f * g - d * i) * invdet;
  inv_cell[4] = (a * i - c * g) * invdet;
  inv_cell[5] = (c * d - a * f) * invdet;
  inv_cell[6] = (d * h - e * g) * invdet;
  inv_cell[7] = (b * g - a * h) * invdet;
  inv_cell[8] = (a * e - b * d) * invdet;
}

template <typename scalar_t>
__device__ void
apply_periodic_boundary(Vector3IO<scalar_t>::vec_t &displacement,
                        const scalar_t *cell, const scalar_t *inv_cell) {
  vec_t fractional;
  vec_t cartesian_wrapped;

  fractional.x = displacement.x * inv_cell[0] + displacement.y * inv_cell[1] +
                 displacement.z * inv_cell[2];
  fractional.y = displacement.x * inv_cell[3] + displacement.y * inv_cell[4] +
                 displacement.z * inv_cell[5];
  fractional.z = displacement.x * inv_cell[6] + displacement.y * inv_cell[7] +
                 displacement.z * inv_cell[8];

  fractional.x -= round(fractional.x);
  fractional.y -= round(fractional.y);
  fractional.z -= round(fractional.z);

  cartesian_wrapped.x =
      fractional.x * cell[0] + fractional.y * cell[3] + fractional.z * cell[6];
  cartesian_wrapped.y =
      fractional.x * cell[1] + fractional.y * cell[4] + fractional.z * cell[7];
  cartesian_wrapped.z =
      fractional.x * cell[2] + fractional.y * cell[5] + fractional.z * cell[8];

  displacement = cartesian_wrapped;
}

template <typename scalar_t>
__device__ void compute_neighbours_full_impl(
    const scalar_t *positions, const scalar_t *cell, const int nnodes,
    const scalar_t cutoff, int *__restrict__ pair_counter,
    int *__restrict__ edge_indices, scalar_t *__restrict__ shifts) {

  using vec_t = typename Vector3IO<scalar_t>::vec_t;

  __shared__ int edge_pair_shared[NWARPS];
  __shared__ int edge_indices_shared[MAX_NEIGHBOURS_PER_ATOM * NWARPS];
  __shared__ scalar_t inv_cell[9];

  const int warp_id = threadIdx.y;
  const int thread_in_warp = threadIdx.x;
  const int node_index = blockIdx.x * blockDim.y + warp_id;
  const scalar_t cutoff2 = cutoff * cutoff;

  if (thread_in_warp == 0)
    edge_pair_shared[warp_id] = 0;

  if (cell != nullptr && thread_in_warp == 0 && warp_id == 0)
    invert_cell_matrix(cell, inv_cell);

  __syncthreads(); // Ensure inv_cell is ready

  if (node_index >= nnodes)
    return;

  vec_t *ri = reinterpret_cast<vec_t *>(&positions[node_index]);

  for (int j = thread_in_warp; j < nnodes; j += blockDim.x) {
    vec_t rj = reinterpret_cast<vec_t *>(&positions[j * 3]);

    vec_t disp = ri - rj;

    if (cell != nullptr)
      apply_periodic_boundary<scalar_t>(disp, cell, inv_cell);

    scalar_t dist2 = dot(disp, disp);

    if (dist2 < cutoff2 && dist2 > scalar_t(0.0)) {
      int edge_local = atomicAdd(&edge_pair_shared[warp_id], 1);
      if (edge_local < MAX_NEIGHBOURS_PER_ATOM) {
        edge_indices_shared[warp_id * MAX_NEIGHBOURS_PER_ATOM + edge_local] = j;
      }
    }
  }

  __syncwarp();

  int iglobal = 0;
  if (thread_in_warp == 0) {
    iglobal = atomicAdd(&pair_counter[0], edge_pair_shared[warp_id]);
  }

  iglobal = __shfl_sync(0xFFFFFFFF, iglobal, 0); // Broadcast iglobal

  int num_edges = edge_pair_shared[warp_id];
  for (int j = thread_in_warp; j < num_edges; j += WARP_SIZE) {
    int edge_idx = edge_indices_shared[warp_id * MAX_NEIGHBOURS_PER_ATOM + j];

    edge_indices[iglobal + j] = node_index; // receiver
    edge_indices[(nnodes * MAX_NEIGHBOURS_PER_ATOM) + iglobal + j] =
        edge_idx; // sender

    vec_t rj = reinterpret_cast<vec_t *>(&positions[edge_idx * 3]);
    vec_t disp = ri - rj;

    if (cell != nullptr)
      apply_periodic_boundary<scalar_t>(disp, cell, inv_cell);

    reinterpret_cast<vec_t &>(shifts[(iglobal + j) * 3]) = disp;
  }
}

template <typename scalar_t>
__global__ void
compute_neighbours_cell_device(const scalar_t *positions, const scalar_t *cell,
                               int nnodes, scalar_t cutoff, int *pair_counter,
                               int *edge_indices, scalar_t *shifts,
                               bool full_list) {
  if (full_list) {
    compute_neighbours_full_impl(positions, cell, nnodes, cutoff, pair_counter,
                                 edge_indices, shifts);
  } else {
    /*compute_neighbours_half_impl(positions, cell, nnodes, cutoff,
       pair_counter, edge_indices, shifts); */
  }
}
