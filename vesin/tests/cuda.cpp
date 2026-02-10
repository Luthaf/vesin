#include <catch2/catch_test_macros.hpp>

#ifdef VESIN_TESTS_WITH_CUDA

#include <cmath>
#include <thread>

#include <cuda_runtime.h>

#include <vesin.h>

void check_cuda(cudaError_t status) {
    if (status != cudaSuccess) {
        const char* message = cudaGetErrorString(status);
        FAIL(message);
    }
}

void run_cuda_test(int device_id) {
    check_cuda(cudaSetDevice(device_id));

    double points[][3] = {
        {0.0, 0.0, 0.0},
        {1.0, 1.0, 1.0},
        {2.0, 2.0, 2.0},
    };
    size_t n_points = 3;
    double (*d_points)[3] = nullptr;
    check_cuda(cudaMalloc(&d_points, sizeof(double) * n_points * 3));
    check_cuda(cudaMemcpy(d_points, points, sizeof(double) * n_points * 3, cudaMemcpyHostToDevice));

    double box[3][3] = {
        {0.0, 3.0, 3.0},
        {3.0, 0.0, 3.0},
        {3.0, 3.0, 0.0},
    };
    double (*d_box)[3] = nullptr;
    check_cuda(cudaMalloc(&d_box, sizeof(double) * 9));
    check_cuda(cudaMemcpy(d_box, box, sizeof(double) * 9, cudaMemcpyHostToDevice));

    bool periodic[3] = {true, true, true};
    bool* d_periodic = nullptr;
    check_cuda(cudaMalloc(&d_periodic, sizeof(bool) * 3));
    check_cuda(cudaMemcpy(d_periodic, periodic, sizeof(bool) * 3, cudaMemcpyHostToDevice));

    VesinNeighborList neighbors;

    auto options = VesinOptions();
    options.cutoff = 3.0;
    options.full = false;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    const char* error_message = nullptr;
    auto status = vesin_neighbors(
        d_points,
        n_points,
        d_box,
        d_periodic,
        {VesinDeviceKind::VesinCUDA, device_id},
        options,
        &neighbors,
        &error_message
    );

    REQUIRE(error_message == nullptr);
    REQUIRE(status == EXIT_SUCCESS);

    CHECK(neighbors.length == 5);
    CHECK(neighbors.pairs != nullptr);
    CHECK(neighbors.shifts != nullptr);
    CHECK(neighbors.distances != nullptr);
    CHECK(neighbors.vectors != nullptr);

    auto* h_pairs = static_cast<size_t (*)[2]>(malloc(sizeof(size_t) * neighbors.length * 2));
    check_cuda(cudaMemcpy(h_pairs, neighbors.pairs, sizeof(size_t) * neighbors.length * 2, cudaMemcpyDeviceToHost));

    auto* h_shifts = static_cast<int32_t (*)[3]>(malloc(sizeof(int32_t) * neighbors.length * 3));
    check_cuda(cudaMemcpy(h_shifts, neighbors.shifts, sizeof(int32_t) * neighbors.length * 3, cudaMemcpyDeviceToHost));

    auto* h_distances = static_cast<double*>(malloc(sizeof(double) * neighbors.length));
    check_cuda(cudaMemcpy(h_distances, neighbors.distances, sizeof(double) * neighbors.length, cudaMemcpyDeviceToHost));

    auto* h_vectors = static_cast<double (*)[3]>(malloc(sizeof(double) * neighbors.length * 3));
    check_cuda(cudaMemcpy(h_vectors, neighbors.vectors, sizeof(double) * neighbors.length * 3, cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < neighbors.length; ++i) {
        if (h_pairs[i][0] == 0 && h_pairs[i][1] == 2) {
            // we have three pairs between 0 and 2 with shifts (-1, 0, 0),
            // (0, -1, 0), and (0, 0, -1)
            CHECK(h_distances[i] == std::sqrt(6.0));

            if (h_shifts[i][0] == -1 && h_shifts[i][1] == 0 && h_shifts[i][2] == 0) {
                CHECK(h_vectors[i][0] == 2.0);
                CHECK(h_vectors[i][1] == -1.0);
                CHECK(h_vectors[i][2] == -1.0);
            } else if (h_shifts[i][0] == 0 && h_shifts[i][1] == -1 && h_shifts[i][2] == 0) {
                CHECK(h_vectors[i][0] == -1.0);
                CHECK(h_vectors[i][1] == 2.0);
                CHECK(h_vectors[i][2] == -1.0);
            } else if (h_shifts[i][0] == 0 && h_shifts[i][1] == 0 && h_shifts[i][2] == -1) {
                CHECK(h_vectors[i][0] == -1.0);
                CHECK(h_vectors[i][1] == -1.0);
                CHECK(h_vectors[i][2] == 2.0);
            } else {
                FAIL("Unexpected shift for pair (0, 2): (" + std::to_string(h_shifts[i][0]) + ", " + std::to_string(h_shifts[i][1]) + ", " + std::to_string(h_shifts[i][2]) + ")");
            }

        } else if ((h_pairs[i][0] == 0 && h_pairs[i][1] == 1) || (h_pairs[i][0] == 1 && h_pairs[i][1] == 2)) {
            // pairs between 0-1 or 1-2 should have zero shifts, distance
            // sqrt(3), and vector (1, 1, 1)
            CHECK(h_shifts[i][0] == 0);
            CHECK(h_shifts[i][1] == 0);
            CHECK(h_shifts[i][2] == 0);

            CHECK(h_distances[i] == std::sqrt(3.0));
            CHECK(h_vectors[i][0] == 1.0);
            CHECK(h_vectors[i][1] == 1.0);
            CHECK(h_vectors[i][2] == 1.0);
        } else {
            FAIL("Unexpected pair: (" + std::to_string(h_pairs[i][0]) + ", " + std::to_string(h_pairs[i][1]) + ")");
        }
    }

    // Clean up
    vesin_free(&neighbors);

    free(h_pairs);
    free(h_shifts);
    free(h_distances);
    free(h_vectors);

    check_cuda(cudaFree(d_points));
    check_cuda(cudaFree(d_box));
    check_cuda(cudaFree(d_periodic));
}

TEST_CASE("Test CUDA") {
    // get the number of CUDA devices
    int n_devices = 0;
    check_cuda(cudaGetDeviceCount(&n_devices));
    REQUIRE(n_devices > 0);

    // start multiple threads to test concurrent execution
    auto threads = std::vector<std::thread>();
    for (int thread_id = 0; thread_id < 10; ++thread_id) {
        std::thread t(run_cuda_test, thread_id % n_devices);
        threads.push_back(std::move(t));
    }

    for (auto& t : threads) {
        t.join();
    }
}

#else

TEST_CASE("CUDA tests are disabled") {}

#endif
