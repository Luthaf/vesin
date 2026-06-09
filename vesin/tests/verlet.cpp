#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>

#include "../src/cluster.hpp"
#include "../src/cpu_cell_list.hpp"
#include "../src/verlet.hpp"

using namespace Catch::Matchers;

#define CHECK_APPROX_EQUAL(a, b) CHECK_THAT(a, WithinULP(b, 4));

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

static vesin::BoundingBox make_box(const std::vector<vesin::Vector>& points, const double matrix[3][3], const bool periodic[3]) {
    auto box_matrix = vesin::Matrix{{{
        {{matrix[0][0], matrix[0][1], matrix[0][2]}},
        {{matrix[1][0], matrix[1][1], matrix[1][2]}},
        {{matrix[2][0], matrix[2][1], matrix[2][2]}},
    }}};

    auto box = vesin::BoundingBox(box_matrix, periodic);
    box.make_bounding_for(reinterpret_cast<const double (*)[3]>(points.data()), points.size());
    return box;
}

static std::vector<vesin::Vector> lattice_points(size_t edge, double spacing) {
    auto points = std::vector<vesin::Vector>();
    points.reserve(edge * edge * edge);

    for (size_t z = 0; z < edge; z++) {
        for (size_t y = 0; y < edge; y++) {
            for (size_t x = 0; x < edge; x++) {
                points.push_back(vesin::Vector{
                    spacing * static_cast<double>(x),
                    spacing * static_cast<double>(y),
                    spacing * static_cast<double>(z),
                });
            }
        }
    }

    return points;
}

static std::vector<vesin::Vector> displaced_points(const std::vector<vesin::Vector>& points) {
    auto displaced = points;

    for (size_t i = 0; i < displaced.size(); i++) {
        displaced[i][0] += (static_cast<double>(i % 3) - 1.0) * 0.01;
        displaced[i][1] += (static_cast<double>((i / 3) % 3) - 1.0) * 0.01;
        displaced[i][2] += (static_cast<double>((i / 9) % 3) - 1.0) * 0.01;
    }

    return displaced;
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

TEST_CASE("Verlet cache invalidates when candidate algorithm changes") {
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

    double points[][3] = {
        {0.0, 0.0, 0.0},
        {0.9, 0.0, 0.0},
        {1.8, 0.0, 0.0},
        {2.7, 0.0, 0.0},
    };

    auto state = vesin::cpu::VerletState();
    state.set_options(options);
    auto box_state = make_box(points, 4, box, periodic);
    state.rebuild(reinterpret_cast<const vesin::Vector*>(points), 4, box_state);

    REQUIRE(state.candidate_count() == 3);

    options.algorithm = VesinAutoAlgorithm;
    state.set_options(options);
    CHECK(state.candidate_count() == 0);
}

TEST_CASE("Auto Verlet cache stores cluster candidates below atom-pair count") {
    double box_matrix[3][3] = {{0.0}};
    bool periodic[3] = {false, false, false};

    auto points = lattice_points(8, 0.9);
    REQUIRE(points.size() >= vesin::CLUSTER_PAIR_THRESHOLD);

    auto options = VesinOptions();
    options.cutoff = 1.0;
    options.skin = 0.35;
    options.full = false;
    options.sorted = false;
    options.algorithm = VesinCellList;
    options.return_shifts = true;
    options.return_distances = false;
    options.return_vectors = false;

    auto box = make_box(points, box_matrix, periodic);

    auto cell_state = vesin::cpu::VerletState();
    cell_state.set_options(options);
    cell_state.rebuild(points.data(), points.size(), box);
    auto atom_candidate_count = cell_state.candidate_count();
    REQUIRE(atom_candidate_count > 0);

    options.algorithm = VesinAutoAlgorithm;
    auto auto_state = vesin::cpu::VerletState();
    auto_state.set_options(options);
    auto_state.rebuild(points.data(), points.size(), box);

    CHECK(auto_state.candidates.length > 0);
    CHECK(auto_state.simd_candidate_count() == auto_state.candidates.length);
    CHECK(auto_state.candidate_count() < atom_candidate_count);
}

TEST_CASE("Auto Verlet cluster cache matches exact cell-list output after small displacements") {
    double box_matrix[3][3] = {{0.0}};
    bool periodic[3] = {false, false, false};

    auto reference_points = lattice_points(8, 0.9);
    auto current_points = displaced_points(reference_points);

    auto options = VesinOptions();
    options.cutoff = 1.0;
    options.skin = 0.35;
    options.full = false;
    options.sorted = true;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    auto reference_box = make_box(reference_points, box_matrix, periodic);
    auto current_box = make_box(current_points, box_matrix, periodic);

    auto auto_state = vesin::cpu::VerletState();
    auto_state.set_options(options);
    auto_state.rebuild(reference_points.data(), reference_points.size(), reference_box);
    REQUIRE_FALSE(auto_state.needs_rebuild(current_points.data(), current_points.size(), current_box));

    auto actual = VesinNeighborList();
    size_t actual_capacity = 0;
    auto_state.recompute(current_points.data(), current_box, options, actual, actual_capacity);

    auto exact_options = options;
    exact_options.skin = 0.0;
    exact_options.algorithm = VesinCellList;

    auto expected = VesinNeighborList();
    size_t expected_capacity = 0;
    vesin::cpu::stateless_neighbors(
        current_points.data(),
        current_points.size(),
        make_box(current_points, box_matrix, periodic),
        exact_options,
        expected,
        expected_capacity
    );

    REQUIRE(actual.length == expected.length);
    for (size_t k = 0; k < actual.length; k++) {
        CHECK(actual.pairs[k][0] == expected.pairs[k][0]);
        CHECK(actual.pairs[k][1] == expected.pairs[k][1]);
        CHECK(actual.shifts[k][0] == expected.shifts[k][0]);
        CHECK(actual.shifts[k][1] == expected.shifts[k][1]);
        CHECK(actual.shifts[k][2] == expected.shifts[k][2]);
        CHECK_APPROX_EQUAL(actual.distances[k], expected.distances[k]);
        CHECK_APPROX_EQUAL(actual.vectors[k][0], expected.vectors[k][0]);
        CHECK_APPROX_EQUAL(actual.vectors[k][1], expected.vectors[k][1]);
        CHECK_APPROX_EQUAL(actual.vectors[k][2], expected.vectors[k][2]);
    }

    actual.device = {VesinCPU, 0};
    expected.device = {VesinCPU, 0};
    vesin_free(&actual);
    vesin_free(&expected);
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
