#include <catch2/catch_test_macros.hpp>

#include "../src/cpu_cell_list.hpp"
#include "../src/verlet.hpp"

TEST_CASE("Verlet filter keeps allocation capacity for smaller output") {
    auto options = VesinOptions();
    options.cutoff = 1.0;
    options.skin = 0.6;
    options.full = false;
    options.sorted = false;
    options.algorithm = VesinCellList;
    options.return_shifts = true;
    options.return_distances = false;
    options.return_vectors = false;

    vesin::Matrix box_matrix;
    bool periodic[3] = {false, false, false};

    vesin::Vector lot_of_pairs_points[] = {
        {0.0, 0.0, 0.0},
        {0.9, 0.0, 0.0},
        {1.8, 0.0, 0.0},
        {2.7, 0.0, 0.0},
    };
    vesin::Vector few_pairs_points[] = {
        {0.0, 0.0, 0.0},
        {1.19, 0.0, 0.0},
        {1.51, 0.0, 0.0},
        {2.7, 0.0, 0.0},
    };

    auto state = vesin::cpu::VerletList();
    state.set_options(options);
    size_t output_capacity = 0;

    auto lot_of_pairs_box = vesin::BoundingBox(box_matrix, periodic);
    lot_of_pairs_box.make_bounding_for(lot_of_pairs_points, 4);
    state.rebuild(lot_of_pairs_points, 4, lot_of_pairs_box);

    REQUIRE(state.candidate_count() == 3);
    auto few_pairs_box = vesin::BoundingBox(box_matrix, periodic);
    few_pairs_box.make_bounding_for(few_pairs_points, 4);
    REQUIRE_FALSE(state.needs_rebuild(few_pairs_points, 4, few_pairs_box));

    auto neighbors = VesinNeighborList();
    neighbors.device = {VesinCPU, 0};
    state.filter(lot_of_pairs_points, lot_of_pairs_box, options, neighbors, output_capacity);
    REQUIRE(neighbors.length == 3);
    REQUIRE(output_capacity >= 3);

    auto initial_capacity = output_capacity;

    state.filter(few_pairs_points, few_pairs_box, options, neighbors, output_capacity);
    REQUIRE(neighbors.length == 1);
    CHECK(output_capacity == initial_capacity);

    state.filter(lot_of_pairs_points, lot_of_pairs_box, options, neighbors, output_capacity);
    REQUIRE(neighbors.length == 3);
    CHECK(output_capacity == initial_capacity);

    vesin_free(&neighbors);
}

TEST_CASE("Periodic systems use minimum-image distance for rebuild") {
    auto options = VesinOptions();
    options.cutoff = 0.25;
    options.skin = 0.2;
    options.full = false;
    options.sorted = false;
    options.algorithm = VesinCellList;
    options.return_shifts = true;
    options.return_distances = false;
    options.return_vectors = false;

    auto box_matrix = vesin::Matrix({{
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    }});
    bool periodic[3] = {true, true, true};

    vesin::Vector initial_points[] = {
        {0.12, 0.11, 0.31},
        {0.95, 0.11, 0.31},
    };
    auto ref_box = vesin::BoundingBox(box_matrix, periodic);
    ref_box.make_bounding_for(initial_points, 2);

    auto state = vesin::cpu::VerletList();
    state.set_options(options);
    state.rebuild(initial_points, 2, ref_box);

    // this is within the skin distance (0.12 -> 0.95 is 0.17 with periodic
    // boundaries) and should not trigger a rebuild
    vesin::Vector small_displacement_points[] = {
        {0.12, 0.11, 0.31},
        {1.02, 0.11, 0.31},
    };
    auto small_displacement_box = vesin::BoundingBox(box_matrix, periodic);
    small_displacement_box.make_bounding_for(small_displacement_points, 2);

    REQUIRE_FALSE(state.needs_rebuild(small_displacement_points, 2, small_displacement_box));

    // this is above the skin distance (0.95 -> 1.11 > 0.2) and should trigger a rebuild
    vesin::Vector large_displacement_points[] = {
        {0.95, 0.11, 0.31},
        {1.11, 0.11, 0.31},
    };
    auto large_displacement_box = vesin::BoundingBox(box_matrix, periodic);
    large_displacement_box.make_bounding_for(large_displacement_points, 2);
    REQUIRE(state.needs_rebuild(large_displacement_points, 2, large_displacement_box));
}
