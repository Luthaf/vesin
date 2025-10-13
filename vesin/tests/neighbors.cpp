#include <array>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace Catch::Matchers;

#include <vesin.h>

#define CHECK_APPROX_EQUAL(a, b) CHECK_THAT(a, WithinULP(b, 4));

static void check_neighbors(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    bool periodic[3],
    double cutoff,
    bool full_list,
    std::vector<std::array<size_t, 2>> expected_pairs,
    std::vector<std::array<int32_t, 3>> expected_shifts,
    std::vector<double> expected_distances,
    std::vector<std::array<double, 3>> expected_vectors
) {
    auto options = VesinOptions();
    options.cutoff = cutoff;
    options.full = full_list;
    options.return_shifts = !expected_shifts.empty();
    options.return_distances = !expected_distances.empty();
    options.return_vectors = !expected_vectors.empty();

    VesinNeighborList neighbors;

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

    if (!expected_pairs.empty()) {
        REQUIRE(neighbors.length == expected_pairs.size());
        for (size_t i = 0; i < neighbors.length; i++) {
            CHECK(neighbors.pairs[i][0] == expected_pairs[i][0]);
            CHECK(neighbors.pairs[i][1] == expected_pairs[i][1]);
        }
    }

    if (!expected_shifts.empty()) {
        REQUIRE(neighbors.length == expected_shifts.size());
        for (size_t i = 0; i < neighbors.length; i++) {
            CHECK(neighbors.shifts[i][0] == expected_shifts[i][0]);
            CHECK(neighbors.shifts[i][1] == expected_shifts[i][1]);
            CHECK(neighbors.shifts[i][2] == expected_shifts[i][2]);
        }
    }

    if (!expected_distances.empty()) {
        REQUIRE(neighbors.length == expected_distances.size());
        for (size_t i = 0; i < neighbors.length; i++) {
            CHECK_APPROX_EQUAL(neighbors.distances[i], expected_distances[i]);
        }
    }

    if (!expected_vectors.empty()) {
        REQUIRE(neighbors.length == expected_vectors.size());
        for (size_t i = 0; i < neighbors.length; i++) {
            CHECK_APPROX_EQUAL(neighbors.vectors[i][0], expected_vectors[i][0]);
            CHECK_APPROX_EQUAL(neighbors.vectors[i][1], expected_vectors[i][1]);
            CHECK_APPROX_EQUAL(neighbors.vectors[i][2], expected_vectors[i][2]);
        }
    }

    vesin_free(&neighbors);
}

TEST_CASE("Non-periodic") {
    double points[][3] = {
        {0.134, 1.282, 1.701},
        {-0.273, 1.026, -1.471},
        {1.922, -0.124, 1.900},
        {1.400, -0.464, 0.480},
        {0.149, 1.865, 0.635},
    };

    double box[3][3] = {{0}};

    // reference computed with ASE
    auto expected_pairs = std::vector<std::array<size_t, 2>>{
        {0, 1},
        {0, 2},
        {0, 3},
        {0, 4},
        {1, 3},
        {1, 4},
        {2, 3},
        {2, 4},
        {3, 4},
    };

    auto expected_distances = std::vector<double>{
        3.2082345612501593,
        2.283282943482914,
        2.4783286706972505,
        1.215100818862369,
        2.9707625283755013,
        2.3059143522689647,
        1.550639867925496,
        2.9495550511899244,
        2.6482573515427084,
    };

    bool periodic[3] = {false, false, false};
    check_neighbors(
        points,
        /*n_points=*/5,
        box,
        periodic,
        /*cutoff=*/3.42,
        /*full_list=*/false,
        expected_pairs,
        {},
        expected_distances,
        {}
    );
}

TEST_CASE("Mixed periodic boundaries") {
    bool periodic[3] = {true, false, false};
    double cutoff = 0.5;

    // To understand this test, only focus on the first and second axis (x and y coords)
    // 1) Since only the first periodic boundary is enabled,
    //    the first and second point are only 0.2 away.
    // 2) Notice that the first and third point are 0.8 away.
    //    However, if periodicity is enabled on the second axis, then
    //    they would only be 0.1 away and would be considered neighbors.

    double points[][3] = {
        {0.1, 0.0, 0.0},
        {0.9, 0.0, 0.0},
        {0.1, 0.9, 0.0},
    };

    double box[3][3] = {
        {1.0, 0.0, 0.0},
        {0.0, 1.0, 0.0},
        {0.0, 0.0, 1.0},
    };

    // reference computed with ASE
    auto expected_pairs = std::vector<std::array<size_t, 2>>{
        {0, 1},
    };

    auto expected_distances = std::vector<double>{0.2};

    check_neighbors(
        points,
        /*n_points=*/3,
        box,
        /*periodic=*/periodic,
        /*cutoff=*/cutoff,
        /*full_list=*/false,
        expected_pairs,
        {},
        expected_distances,
        {}
    );
}

TEST_CASE("FCC unit cell") {
    double points[][3] = {
        {0.0, 0.0, 0.0},
    };

    double box[3][3] = {
        {0.0, 1.5, 1.5},
        {1.5, 0.0, 1.5},
        {1.5, 1.5, 0.0},
    };

    auto expected_vectors = std::vector<std::array<double, 3>>{
        {1.5, 0.0, -1.5},
        {1.5, -1.5, 0.0},
        {0.0, 1.5, -1.5},
        {1.5, 1.5, 0.0},
        {1.5, 0.0, 1.5},
        {0.0, 1.5, 1.5},
    };

    auto expected_shifts = std::vector<std::array<int32_t, 3>>{
        {-1, 0, 1},
        {-1, 1, 0},
        {0, -1, 1},
        {0, 0, 1},
        {0, 1, 0},
        {1, 0, 0},
    };

    auto expected_pairs = std::vector<std::array<size_t, 2>>(6, {0, 0});
    auto expected_distances = std::vector<double>(6, 2.1213203435596424);

    bool periodic[3] = {true, true, true};
    check_neighbors(
        points,
        /*n_points=*/1,
        box,
        periodic,
        /*cutoff=*/3.0,
        /*full_list=*/false,
        expected_pairs,
        expected_shifts,
        expected_distances,
        expected_vectors
    );
}

TEST_CASE("Large box, small cutoff") {
    double points[][3] = {
        {0.0, 0.0, 0.0},
        {0.0, 2.0, 0.0},
        {0.0, 0.0, 2.0},
        // points outside the box natural boundaries
        {-6.0, 0.0, 0.0},
        {-6.0, -2.0, 0.0},
        {-6.0, 0.0, -2.0},
    };

    double box[3][3] = {
        {54.0, 0.0, 0.0},
        {0.0, 54.0, 0.0},
        {0.0, 0.0, 54.0},
    };

    auto expected_pairs = std::vector<std::array<size_t, 2>>{
        {0, 1},
        {0, 2},
        {3, 4},
        {3, 5},
    };
    auto expected_shifts = std::vector<std::array<int32_t, 3>>(4, {0, 0, 0});
    auto expected_distances = std::vector<double>(4, 2.0);

    bool periodic[3] = {true, true, true};
    check_neighbors(
        points,
        /*n_points=*/6,
        box,
        periodic,
        /*cutoff=*/2.1,
        /*full_list=*/false,
        expected_pairs,
        expected_shifts,
        expected_distances,
        {}
    );
}

TEST_CASE("Cutoff larger than the box size") {
    double points[][3] = {
        {0.0, 0.0, 0.0},
    };

    double box[3][3] = {
        {0.5, 0.0, 0.0},
        {0.0, 0.5, 0.0},
        {0.0, 0.0, 0.5},
    };

    auto expected_pairs = std::vector<std::array<size_t, 2>>(3, {0, 0});
    auto expected_distances = std::vector<double>(3, 0.5);
    auto expected_shifts = std::vector<std::array<int32_t, 3>>{
        {0, 0, 1},
        {0, 1, 0},
        {1, 0, 0},
    };
    auto expected_vectors = std::vector<std::array<double, 3>>{
        {0.0, 0.0, 0.5},
        {0.0, 0.5, 0.0},
        {0.5, 0.0, 0.0},
    };

    bool periodic[3] = {true, true, true};
    check_neighbors(
        points,
        /*n_points=*/1,
        box,
        periodic,
        /*cutoff=*/0.6,
        /*full_list=*/false,
        expected_pairs,
        expected_shifts,
        expected_distances,
        expected_vectors
    );
}

TEST_CASE("Slanted box") {
    double points[][3] = {
        {1.42, 0.0, 0.0},
        {2.84, 0.0, 0.0},
        {3.55, -1.22975607, 0.0},
        {4.97, -1.22975607, 0.0},
    };

    double box[3][3] = {
        {4.26, -2.45951215, 0.0},
        {2.13, 1.22975607, 0.0},
        {0.0, 0.0, 50.0},
    };

    auto options = VesinOptions();
    options.cutoff = 6.4;
    options.full = false;
    options.return_shifts = true;
    options.return_distances = false;
    options.return_vectors = false;

    VesinNeighborList neighbors;

    const char* error_message = nullptr;
    bool periodic[3] = {true, true, true};
    auto status = vesin_neighbors(
        points,
        /*n_points=*/4,
        box,
        periodic,
        {VesinDeviceKind::VesinCPU, 0},
        options,
        &neighbors,
        &error_message
    );
    REQUIRE(status == EXIT_SUCCESS);

    REQUIRE(neighbors.length == 90);
    auto previously_missing = std::vector<std::array<int32_t, 3>>{
        {-2, 0, 0},
        {-2, 1, 0},
        {-2, 2, 0},
    };

    for (const auto& missing : previously_missing) {
        bool found = false;
        for (size_t i = 0; i < neighbors.length; i++) {
            auto* pair = neighbors.pairs[i];
            if (pair[0] != 0 || pair[1] != 3) {
                continue;
            }

            auto* shift = neighbors.shifts[i];
            if (shift[0] == missing[0] && shift[1] == missing[1] && shift[2] == missing[2]) {
                found = true;
                break;
            }
        }
        REQUIRE(found);
    }

    vesin_free(&neighbors);
}

// TODO: tests for full NL
