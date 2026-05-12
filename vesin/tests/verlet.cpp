#include <catch2/catch_test_macros.hpp>

#include "../src/cpu_cell_list.hpp"
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
    size_t output_capacity = 0;

    auto high_output_box = make_box(high_output_points, 4, box, periodic);
    state.rebuild(reinterpret_cast<const vesin::Vector*>(high_output_points), 4, high_output_box);

    REQUIRE(state.candidate_count() == 3);
    auto low_output_box = make_box(low_output_points, 4, box, periodic);
    REQUIRE_FALSE(state.needs_rebuild(reinterpret_cast<const vesin::Vector*>(low_output_points), 4, low_output_box));

    auto neighbors = VesinNeighborList();
    state.recompute(reinterpret_cast<const vesin::Vector*>(high_output_points), high_output_box, options, neighbors, output_capacity);
    REQUIRE(neighbors.length == 3);
    REQUIRE(output_capacity >= 3);

    const auto high_output_capacity = output_capacity;

    state.recompute(reinterpret_cast<const vesin::Vector*>(low_output_points), low_output_box, options, neighbors, output_capacity);
    REQUIRE(neighbors.length == 1);
    CHECK(output_capacity == high_output_capacity);

    state.recompute(reinterpret_cast<const vesin::Vector*>(high_output_points), high_output_box, options, neighbors, output_capacity);
    REQUIRE(neighbors.length == 3);
    CHECK(output_capacity == high_output_capacity);

    neighbors.device = {VesinCPU, 0};
    vesin_free(&neighbors);
}

TEST_CASE("Periodic wrapped coordinates use minimum-image distance for rebuild") {
    double box_matrix[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    bool periodic[3] = {true, true, true};

    auto options = VesinOptions();
    options.cutoff = 0.25;
    options.skin = 0.2;
    options.full = false;
    options.sorted = false;
    options.algorithm = VesinCellList;
    options.return_shifts = true;
    options.return_distances = false;
    options.return_vectors = false;

    // Atom B starts near one boundary; after a small wrapped displacement across the
    // boundary, physical motion is 0.07 but wrapped coordinates jump by 0.93.
    double wrapped_pair_ref[][3] = {
        {0.12, 0.11, 0.31},
        {0.95, 0.11, 0.31},
    };
    auto ref_box = make_box(wrapped_pair_ref, 2, box_matrix, periodic);

    auto state = vesin::cpu::VerletState();
    state.set_options(options);
    state.rebuild(reinterpret_cast<const vesin::Vector*>(wrapped_pair_ref), 2, ref_box);

    double wrapped_pair_small_shift[][3] = {
        {0.12, 0.11, 0.31},
        {1.02, 0.11, 0.31},
    };
    auto small_shift_box = make_box(wrapped_pair_small_shift, 2, box_matrix, periodic);
    REQUIRE_FALSE(state.needs_rebuild(
        reinterpret_cast<const vesin::Vector*>(wrapped_pair_small_shift),
        2,
        small_shift_box
    ));

    double wrapped_pair_large_shift[][3] = {
        {0.12, 0.11, 0.31},
        {1.11, 0.11, 0.31},
    };
    auto large_shift_box = make_box(wrapped_pair_large_shift, 2, box_matrix, periodic);
    REQUIRE(state.needs_rebuild(
        reinterpret_cast<const vesin::Vector*>(wrapped_pair_large_shift),
        2,
        large_shift_box
    ));
}

TEST_CASE("Non-periodic displacement validation is origin independent") {
    double box_matrix[3][3] = {{0.0}};
    bool periodic[3] = {false, false, false};

    auto options = VesinOptions();
    options.cutoff = 1.0;
    options.skin = 0.4;
    options.full = false;
    options.sorted = false;
    options.algorithm = VesinCellList;
    options.return_shifts = true;
    options.return_distances = false;
    options.return_vectors = false;

    double shifted_points[][3] = {
        {10.0, 20.0, -30.0},
        {10.7, 20.0, -30.0},
    };

    auto box = make_box(shifted_points, 2, box_matrix, periodic);
    auto state = vesin::cpu::VerletState();
    state.set_options(options);
    state.rebuild(reinterpret_cast<const vesin::Vector*>(shifted_points), 2, box);

    REQUIRE_FALSE(state.needs_rebuild(
        reinterpret_cast<const vesin::Vector*>(shifted_points),
        2,
        box
    ));
}

TEST_CASE("CPU extra data persists across stateless and Verlet calls") {
    double box_matrix[3][3] = {{0.0}};
    bool periodic[3] = {false, false, false};

    auto options = VesinOptions();
    options.cutoff = 1.0;
    options.skin = 0.0;
    options.full = false;
    options.sorted = false;
    options.algorithm = VesinCellList;
    options.return_shifts = true;
    options.return_distances = false;
    options.return_vectors = false;

    double points[][3] = {
        {0.0, 0.0, 0.0},
        {0.9, 0.0, 0.0},
        {1.8, 0.0, 0.0},
        {2.7, 0.0, 0.0},
    };

    auto neighbors = VesinNeighborList();

    vesin::cpu::neighbors(
        reinterpret_cast<const vesin::Vector*>(points),
        4,
        make_box(points, 4, box_matrix, periodic),
        options,
        neighbors
    );
    REQUIRE(neighbors.opaque != nullptr);
    auto* opaque = neighbors.opaque;
    auto* extra = static_cast<vesin::cpu::ExtraDataCpu*>(neighbors.opaque);
    REQUIRE(extra->capacity >= neighbors.length);
    REQUIRE(extra->verlet_state == nullptr);

    options.skin = 0.6;
    vesin::cpu::neighbors(
        reinterpret_cast<const vesin::Vector*>(points),
        4,
        make_box(points, 4, box_matrix, periodic),
        options,
        neighbors
    );
    CHECK(neighbors.opaque == opaque);
    CHECK(extra->capacity >= neighbors.length);
    CHECK(extra->verlet_state != nullptr);

    options.skin = 0.0;
    vesin::cpu::neighbors(
        reinterpret_cast<const vesin::Vector*>(points),
        4,
        make_box(points, 4, box_matrix, periodic),
        options,
        neighbors
    );
    CHECK(neighbors.opaque == opaque);
    CHECK(extra->capacity >= neighbors.length);
    CHECK(extra->verlet_state == nullptr);

    neighbors.device = {VesinCPU, 0};
    vesin_free(&neighbors);
}
