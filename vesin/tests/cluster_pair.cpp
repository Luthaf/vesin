#include <array>
#include <cmath>
#include <cstdio>
#include <set>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace Catch::Matchers;

#include <vesin.h>

/// Helper: build a simple cubic lattice with n^3 atoms
static std::vector<std::array<double, 3>> cubic_lattice(int n, double spacing) {
    std::vector<std::array<double, 3>> points;
    for (int ix = 0; ix < n; ix++) {
        for (int iy = 0; iy < n; iy++) {
            for (int iz = 0; iz < n; iz++) {
                points.push_back({
                    ix * spacing,
                    iy * spacing,
                    iz * spacing,
                });
            }
        }
    }
    return points;
}

/// Helper: collect (i, j, shift) tuples into a set for comparison
using PairSet = std::set<std::tuple<size_t, size_t, int32_t, int32_t, int32_t>>;

static PairSet collect_pairs(const VesinNeighborList& nl) {
    PairSet result;
    for (size_t k = 0; k < nl.length; k++) {
        result.emplace(
            nl.pairs[k][0], nl.pairs[k][1],
            nl.shifts[k][0], nl.shifts[k][1], nl.shifts[k][2]
        );
    }
    return result;
}

/// Compute a neighbor list forcing cell-list algorithm
static VesinNeighborList compute_with_algorithm(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    bool periodic[3],
    double cutoff,
    bool full_list,
    VesinAlgorithm algorithm
) {
    auto options = VesinOptions();
    options.cutoff = cutoff;
    options.full = full_list;
    options.sorted = false;
    options.algorithm = algorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    VesinNeighborList neighbors;
    const char* error_message = nullptr;
    auto status = vesin_neighbors(
        points, n_points, box, periodic,
        {VesinCPU, 0}, options, &neighbors, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);
    if (error_message != nullptr) {
        FAIL("Error: " << error_message);
    }
    return neighbors;
}

TEST_CASE("Cluster-pair: correctness vs cell-list on 4x4x4 lattice") {
    // 4^3 = 64 atoms, below cluster-pair threshold (256) so Auto
    // uses cell-list. Verifies both paths agree.
    auto points = cubic_lattice(4, 1.5);
    REQUIRE(points.size() == 64);

    double box_len = 4 * 1.5;
    double box[3][3] = {{box_len, 0, 0}, {0, box_len, 0}, {0, 0, box_len}};
    bool periodic[3] = {true, true, true};
    double cutoff = 2.5;

    // Cell-list (forced)
    auto cell_list_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinCellList
    );

    // Auto (should use cluster-pair for N=64)
    auto auto_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinAutoAlgorithm
    );

    auto cl_pairs = collect_pairs(cell_list_nl);
    auto auto_pairs = collect_pairs(auto_nl);

    CHECK(cl_pairs.size() == auto_pairs.size());
    CHECK(cl_pairs == auto_pairs);

    vesin_free(&cell_list_nl);
    vesin_free(&auto_nl);
}

TEST_CASE("Cluster-pair: full list on 4x4x4 lattice") {
    auto points = cubic_lattice(4, 1.5);
    double box_len = 4 * 1.5;
    double box[3][3] = {{box_len, 0, 0}, {0, box_len, 0}, {0, 0, box_len}};
    bool periodic[3] = {true, true, true};
    double cutoff = 2.5;

    auto cell_list_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, true, VesinCellList
    );

    auto auto_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, true, VesinAutoAlgorithm
    );

    auto cl_pairs = collect_pairs(cell_list_nl);
    auto auto_pairs = collect_pairs(auto_nl);

    CHECK(cl_pairs.size() == auto_pairs.size());
    CHECK(cl_pairs == auto_pairs);

    vesin_free(&cell_list_nl);
    vesin_free(&auto_nl);
}

TEST_CASE("Cluster-pair: non-periodic system") {
    auto points = cubic_lattice(5, 1.2); // 125 atoms
    double box[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    bool periodic[3] = {false, false, false};
    double cutoff = 2.0;

    auto cell_list_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinCellList
    );

    auto auto_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinAutoAlgorithm
    );

    auto cl_pairs = collect_pairs(cell_list_nl);
    auto auto_pairs = collect_pairs(auto_nl);

    CHECK(cl_pairs.size() == auto_pairs.size());
    CHECK(cl_pairs == auto_pairs);

    vesin_free(&cell_list_nl);
    vesin_free(&auto_nl);
}

TEST_CASE("Cluster-pair: distances match cell-list") {
    auto points = cubic_lattice(4, 1.5);
    double box_len = 4 * 1.5;
    double box[3][3] = {{box_len, 0, 0}, {0, box_len, 0}, {0, 0, box_len}};
    bool periodic[3] = {true, true, true};
    double cutoff = 2.5;

    // Both sorted so we can compare element-wise
    auto options = VesinOptions();
    options.cutoff = cutoff;
    options.full = false;
    options.sorted = true;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    VesinNeighborList cl_nl;
    const char* error_message = nullptr;

    options.algorithm = VesinCellList;
    auto status = vesin_neighbors(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic,
        {VesinCPU, 0}, options, &cl_nl, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);

    VesinNeighborList auto_nl;
    options.algorithm = VesinAutoAlgorithm;
    status = vesin_neighbors(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic,
        {VesinCPU, 0}, options, &auto_nl, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);

    REQUIRE(cl_nl.length == auto_nl.length);

    for (size_t k = 0; k < cl_nl.length; k++) {
        CHECK(cl_nl.pairs[k][0] == auto_nl.pairs[k][0]);
        CHECK(cl_nl.pairs[k][1] == auto_nl.pairs[k][1]);
        CHECK(cl_nl.shifts[k][0] == auto_nl.shifts[k][0]);
        CHECK(cl_nl.shifts[k][1] == auto_nl.shifts[k][1]);
        CHECK(cl_nl.shifts[k][2] == auto_nl.shifts[k][2]);
        CHECK_THAT(cl_nl.distances[k], WithinULP(auto_nl.distances[k], 4));
    }

    vesin_free(&cl_nl);
    vesin_free(&auto_nl);
}

TEST_CASE("Cluster-pair: triclinic box") {
    // Triclinic box with 64 atoms -- exercises the shifted BB test
    // for periodic images where cell_shift != (0,0,0).
    auto points = cubic_lattice(4, 1.2); // 64 atoms
    REQUIRE(points.size() == 64);

    // Triclinic box: off-diagonal elements create non-trivial periodic shifts
    double box[3][3] = {{4.8, 0.0, 0.0}, {1.2, 4.8, 0.0}, {0.8, 0.6, 4.8}};
    bool periodic[3] = {true, true, true};
    double cutoff = 2.0;

    auto cell_list_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinCellList
    );

    auto auto_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinAutoAlgorithm
    );

    auto cl_pairs = collect_pairs(cell_list_nl);
    auto auto_pairs = collect_pairs(auto_nl);

    CHECK(cl_pairs.size() == auto_pairs.size());
    CHECK(cl_pairs == auto_pairs);

    vesin_free(&cell_list_nl);
    vesin_free(&auto_nl);
}

TEST_CASE("Cluster-pair: triclinic box full list") {
    // Same triclinic geometry but with full list
    auto points = cubic_lattice(4, 1.2);
    double box[3][3] = {{4.8, 0.0, 0.0}, {1.2, 4.8, 0.0}, {0.8, 0.6, 4.8}};
    bool periodic[3] = {true, true, true};
    double cutoff = 2.0;

    auto cell_list_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, true, VesinCellList
    );

    auto auto_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, true, VesinAutoAlgorithm
    );

    auto cl_pairs = collect_pairs(cell_list_nl);
    auto auto_pairs = collect_pairs(auto_nl);

    CHECK(cl_pairs.size() == auto_pairs.size());
    CHECK(cl_pairs == auto_pairs);

    vesin_free(&cell_list_nl);
    vesin_free(&auto_nl);
}

TEST_CASE("Cluster-pair: large periodic system with BB rejection") {
    // 125 atoms in periodic box. Many pairs come from periodic images,
    // so this exercises the shifted BB distance test under load.
    auto points = cubic_lattice(5, 1.0); // 125 atoms
    double box_len = 5.0;
    double box[3][3] = {{box_len, 0, 0}, {0, box_len, 0}, {0, 0, box_len}};
    bool periodic[3] = {true, true, true};
    double cutoff = 1.8;

    auto cell_list_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinCellList
    );

    auto auto_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinAutoAlgorithm
    );

    auto cl_pairs = collect_pairs(cell_list_nl);
    auto auto_pairs = collect_pairs(auto_nl);

    CHECK(cl_pairs.size() > 0);
    CHECK(cl_pairs.size() == auto_pairs.size());
    CHECK(cl_pairs == auto_pairs);

    vesin_free(&cell_list_nl);
    vesin_free(&auto_nl);
}

TEST_CASE("Cluster-pair: larger system 5x5x5") {
    auto points = cubic_lattice(5, 1.2); // 125 atoms
    double box_len = 5 * 1.2;
    double box[3][3] = {{box_len, 0, 0}, {0, box_len, 0}, {0, 0, box_len}};
    bool periodic[3] = {true, true, true};
    double cutoff = 2.0;

    auto cell_list_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinCellList
    );

    auto auto_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinAutoAlgorithm
    );

    auto cl_pairs = collect_pairs(cell_list_nl);
    auto auto_pairs = collect_pairs(auto_nl);

    CHECK(cl_pairs.size() == auto_pairs.size());
    CHECK(cl_pairs == auto_pairs);

    vesin_free(&cell_list_nl);
    vesin_free(&auto_nl);
}

// --- Tests with N >= 256 that exercise the cluster-pair SIMD path ---

TEST_CASE("Cluster-pair SIMD: 7x7x7 periodic half list") {
    // 7^3 = 343 atoms -> above threshold (256), Auto uses cluster-pair
    auto points = cubic_lattice(7, 1.2);
    REQUIRE(points.size() == 343);

    double box_len = 7 * 1.2;
    double box[3][3] = {{box_len, 0, 0}, {0, box_len, 0}, {0, 0, box_len}};
    bool periodic[3] = {true, true, true};
    double cutoff = 2.0;

    auto cell_list_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinCellList
    );

    auto auto_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinAutoAlgorithm
    );

    auto cl_pairs = collect_pairs(cell_list_nl);
    auto auto_pairs = collect_pairs(auto_nl);

    CHECK(cl_pairs.size() > 0);
    CHECK(cl_pairs.size() == auto_pairs.size());
    CHECK(cl_pairs == auto_pairs);

    vesin_free(&cell_list_nl);
    vesin_free(&auto_nl);
}

TEST_CASE("Cluster-pair SIMD: 7x7x7 periodic full list") {
    auto points = cubic_lattice(7, 1.2);
    double box_len = 7 * 1.2;
    double box[3][3] = {{box_len, 0, 0}, {0, box_len, 0}, {0, 0, box_len}};
    bool periodic[3] = {true, true, true};
    double cutoff = 2.0;

    auto cell_list_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, true, VesinCellList
    );

    auto auto_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, true, VesinAutoAlgorithm
    );

    auto cl_pairs = collect_pairs(cell_list_nl);
    auto auto_pairs = collect_pairs(auto_nl);

    CHECK(cl_pairs.size() > 0);
    CHECK(cl_pairs.size() == auto_pairs.size());
    CHECK(cl_pairs == auto_pairs);

    vesin_free(&cell_list_nl);
    vesin_free(&auto_nl);
}

TEST_CASE("Cluster-pair SIMD: 7x7x7 non-periodic") {
    auto points = cubic_lattice(7, 1.2); // 343 atoms
    double box[3][3] = {{0, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    bool periodic[3] = {false, false, false};
    double cutoff = 2.0;

    auto cell_list_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinCellList
    );

    auto auto_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinAutoAlgorithm
    );

    auto cl_pairs = collect_pairs(cell_list_nl);
    auto auto_pairs = collect_pairs(auto_nl);

    CHECK(cl_pairs.size() > 0);
    CHECK(cl_pairs.size() == auto_pairs.size());
    CHECK(cl_pairs == auto_pairs);

    vesin_free(&cell_list_nl);
    vesin_free(&auto_nl);
}

TEST_CASE("Cluster-pair SIMD: triclinic 7x7x7") {
    auto points = cubic_lattice(7, 1.2); // 343 atoms
    // Triclinic box
    double box[3][3] = {{8.4, 0.0, 0.0}, {2.1, 8.4, 0.0}, {1.4, 1.05, 8.4}};
    bool periodic[3] = {true, true, true};
    double cutoff = 2.0;

    auto cell_list_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinCellList
    );

    auto auto_nl = compute_with_algorithm(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic, cutoff, false, VesinAutoAlgorithm
    );

    auto cl_pairs = collect_pairs(cell_list_nl);
    auto auto_pairs = collect_pairs(auto_nl);

    CHECK(cl_pairs.size() > 0);
    CHECK(cl_pairs.size() == auto_pairs.size());
    CHECK(cl_pairs == auto_pairs);

    vesin_free(&cell_list_nl);
    vesin_free(&auto_nl);
}

TEST_CASE("Cluster-pair SIMD: distances match cell-list 7x7x7") {
    auto points = cubic_lattice(7, 1.2); // 343 atoms
    double box_len = 7 * 1.2;
    double box[3][3] = {{box_len, 0, 0}, {0, box_len, 0}, {0, 0, box_len}};
    bool periodic[3] = {true, true, true};
    double cutoff = 2.0;

    auto options = VesinOptions();
    options.cutoff = cutoff;
    options.full = false;
    options.sorted = true;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;

    VesinNeighborList cl_nl;
    const char* error_message = nullptr;

    options.algorithm = VesinCellList;
    auto status = vesin_neighbors(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic,
        {VesinCPU, 0}, options, &cl_nl, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);

    VesinNeighborList auto_nl;
    options.algorithm = VesinAutoAlgorithm;
    status = vesin_neighbors(
        reinterpret_cast<const double(*)[3]>(points.data()),
        points.size(), box, periodic,
        {VesinCPU, 0}, options, &auto_nl, &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);

    REQUIRE(cl_nl.length == auto_nl.length);

    for (size_t k = 0; k < cl_nl.length; k++) {
        CHECK(cl_nl.pairs[k][0] == auto_nl.pairs[k][0]);
        CHECK(cl_nl.pairs[k][1] == auto_nl.pairs[k][1]);
        CHECK(cl_nl.shifts[k][0] == auto_nl.shifts[k][0]);
        CHECK(cl_nl.shifts[k][1] == auto_nl.shifts[k][1]);
        CHECK(cl_nl.shifts[k][2] == auto_nl.shifts[k][2]);
        CHECK_THAT(cl_nl.distances[k], WithinULP(auto_nl.distances[k], 4));
    }

    vesin_free(&cl_nl);
    vesin_free(&auto_nl);
}
