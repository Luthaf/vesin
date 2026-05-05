#include <catch2/catch_test_macros.hpp>

#include "../src/verlet.hpp"

static vesin::BoundingBox make_box(const double (*points)[3], size_t n_points, const double matrix[3][3], const bool periodic[3]) {
    auto box_matrix = vesin::Matrix{{{
        {{matrix[0][0], matrix[0][1], matrix[0][2]}},
        {{matrix[1][0], matrix[1][1], matrix[1][2]}},
        {{matrix[2][0], matrix[2][1], matrix[2][2]}},
    }}};

    auto box = vesin::BoundingBox(box_matrix, periodic);
    box.make_bounding_for(points, n_points);
    return box;
}

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
        {1.19, 0.0, 0.0},
        {1.51, 0.0, 0.0},
        {2.7, 0.0, 0.0},
    };

    auto state = vesin::cpu::VerletState();
    state.set_options(options);

    auto high_output_box = make_box(high_output_points, 4, box, periodic);
    state.rebuild(reinterpret_cast<const vesin::Vector*>(high_output_points), 4, high_output_box);

    REQUIRE(state.candidate_count() == 3);
    auto low_output_box = make_box(low_output_points, 4, box, periodic);
    REQUIRE_FALSE(state.needs_rebuild(reinterpret_cast<const vesin::Vector*>(low_output_points), 4, low_output_box));

    auto neighbors = VesinNeighborList();
    state.recompute(reinterpret_cast<const vesin::Vector*>(high_output_points), high_output_box, options, neighbors);
    REQUIRE(neighbors.length == 3);
    REQUIRE(state.output_capacity >= 3);

    const auto high_output_capacity = state.output_capacity;

    state.recompute(reinterpret_cast<const vesin::Vector*>(low_output_points), low_output_box, options, neighbors);
    REQUIRE(neighbors.length == 1);
    CHECK(state.output_capacity == high_output_capacity);

    state.recompute(reinterpret_cast<const vesin::Vector*>(high_output_points), high_output_box, options, neighbors);
    REQUIRE(neighbors.length == 3);
    CHECK(state.output_capacity == high_output_capacity);

    neighbors.device = {VesinCPU, 0};
    vesin_free(&neighbors);
}
