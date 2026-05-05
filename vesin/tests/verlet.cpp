#include <catch2/catch_test_macros.hpp>

#include "../src/verlet.hpp"

TEST_CASE("Verlet recompute keeps allocation capacity across shorter output") {
    double box[3][3] = {{0.0}};
    bool periodic[3] = {false, false, false};

    auto options = VesinOptions();
    options.cutoff = 1.0;
    options.skin = 0.6;
    options.full = false;
    options.sorted = false;
    options.algorithm = VesinCellList;
    options.return_shifts = true;
    options.return_distances = false;
    options.return_vectors = false;

    double high_output_points[][3] = {
        {0.0, 0.0, 0.0},
        {0.9, 0.0, 0.0},
        {1.8, 0.0, 0.0},
        {2.7, 0.0, 0.0},
    };
    double low_output_points[][3] = {
        {0.0, 0.0, 0.0},
        {1.2, 0.0, 0.0},
        {1.5, 0.0, 0.0},
        {2.7, 0.0, 0.0},
    };

    auto state = vesin::VerletState();
    vesin::verlet_set_options(state, options);
    vesin::verlet_rebuild(state, high_output_points, 4, box, periodic);

    REQUIRE(state.n_pairs == 3);
    REQUIRE_FALSE(vesin::verlet_needs_rebuild(state, low_output_points, 4, box, periodic));

    auto neighbors = VesinNeighborList();
    vesin::verlet_recompute(state, high_output_points, box, options, neighbors);
    REQUIRE(neighbors.length == 3);
    REQUIRE(state.output_capacity >= 3);

    const auto high_output_capacity = state.output_capacity;

    vesin::verlet_recompute(state, low_output_points, box, options, neighbors);
    REQUIRE(neighbors.length == 1);
    CHECK(state.output_capacity == high_output_capacity);

    vesin::verlet_recompute(state, high_output_points, box, options, neighbors);
    REQUIRE(neighbors.length == 3);
    CHECK(state.output_capacity == high_output_capacity);

    neighbors.device = {VesinCPU, 0};
    vesin_free(&neighbors);
}
