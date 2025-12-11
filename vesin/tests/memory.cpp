#include <catch2/catch_test_macros.hpp>

#include <vesin.h>

TEST_CASE("Re-use allocations") {
    double points[][3] = {
        {0.0, 0.0, 0.0},
    };
    size_t n_points = 1;

    double box[3][3] = {
        {0.0, 1.5, 1.5},
        {1.5, 0.0, 1.5},
        {1.5, 1.5, 0.0},
    };
    bool periodic[3] = {true, true, true};

    VesinNeighborList neighbors;

    auto compute_neighbors = [&](VesinOptions options) {
        const char* error_message = nullptr;
        auto status = vesin_neighbors(
            points,
            n_points,
            box,
            periodic,
            {VesinDeviceKind::VesinCPU, 0},
            options,
            &neighbors,
            &error_message
        );
        REQUIRE(status == EXIT_SUCCESS);
        REQUIRE(error_message == nullptr);
    };

    auto options = VesinOptions();
    options.cutoff = 3.4;
    options.full = false;
    options.return_shifts = false;
    options.return_distances = true;
    options.return_vectors = false;

    compute_neighbors(options);

    CHECK(neighbors.length == 9);
    CHECK(neighbors.pairs != nullptr);
    CHECK(neighbors.shifts == nullptr);
    CHECK(neighbors.distances != nullptr);
    CHECK(neighbors.vectors == nullptr);

    /***************************************************/
    options.cutoff = 6;
    compute_neighbors(options);

    CHECK(neighbors.length == 67);
    CHECK(neighbors.pairs != nullptr);
    CHECK(neighbors.shifts == nullptr);
    CHECK(neighbors.distances != nullptr);
    CHECK(neighbors.vectors == nullptr);

    /***************************************************/
    options.full = true;
    compute_neighbors(options);

    CHECK(neighbors.length == 134);
    CHECK(neighbors.pairs != nullptr);
    CHECK(neighbors.shifts == nullptr);
    CHECK(neighbors.distances != nullptr);
    CHECK(neighbors.vectors == nullptr);

    /***************************************************/
    options.cutoff = 4.5;
    options.full = false;
    options.return_shifts = true;
    options.return_distances = false;
    compute_neighbors(options);

    CHECK(neighbors.length == 27);
    CHECK(neighbors.pairs != nullptr);
    CHECK(neighbors.shifts != nullptr);
    CHECK(neighbors.distances == nullptr);
    CHECK(neighbors.vectors == nullptr);

    vesin_free(&neighbors);
}
