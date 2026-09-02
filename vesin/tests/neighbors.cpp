#include <algorithm>
#include <array>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace Catch::Matchers;

#include <vesin.h>

#define CHECK_APPROX_EQUAL(a, b) CHECK_THAT(a, WithinULP(b, 4));

/// A single neighbor list entry, used to compare actual and expected
/// half-list output independently of pair ordering (see `canonicalize`).
struct Pair {
    size_t i = 0;
    size_t j = 0;
    std::array<int32_t, 3> shift = {0, 0, 0};
    double distance = 0.0;
    std::array<double, 3> vector = {0.0, 0.0, 0.0};

    /// Canonicalize a half-list pair so that swapping which point is `i` vs `j`
    /// (and correspondingly negating the shift/vector) does not affect comparison.
    void canonicalize() {
        bool negate = false;
        if (this->i > this->j) {
            std::swap(this->i, this->j);
            negate = true;
        } else if (this->i == this->j) {
            auto total = this->shift[0] + this->shift[1] + this->shift[2];
            if (total < 0 || (total == 0 && (this->shift[2] < 0 || (this->shift[2] == 0 && this->shift[1] < 0)))) {
                negate = true;
            }
        }

        if (negate) {
            for (size_t d = 0; d < 3; d++) {
                this->shift[d] = -this->shift[d];
                this->vector[d] = -this->vector[d];
            }
        }
    }
};



static bool by_canonical_key(const Pair& a, const Pair& b) {
    if (a.i != b.i) {
        return a.i < b.i;
    }
    if (a.j != b.j) {
        return a.j < b.j;
    }
    return a.shift < b.shift;
}

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
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
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

    auto n = neighbors.length;
    if (!expected_pairs.empty()) {
        REQUIRE(n == expected_pairs.size());
    }
    if (!expected_shifts.empty()) {
        REQUIRE(n == expected_shifts.size());
    }
    if (!expected_distances.empty()) {
        REQUIRE(n == expected_distances.size());
    }
    if (!expected_vectors.empty()) {
        REQUIRE(n == expected_vectors.size());
    }

    if (full_list || expected_pairs.empty()) {
        // Full lists (and calls that don't check pairs at all) have no
        // pair-ordering ambiguity: compare positionally, as before.
        for (size_t i = 0; i < n; i++) {
            if (!expected_pairs.empty()) {
                CHECK(neighbors.pairs[i][0] == expected_pairs[i][0]);
                CHECK(neighbors.pairs[i][1] == expected_pairs[i][1]);
            }

            if (!expected_shifts.empty()) {
                CHECK(neighbors.shifts[i][0] == expected_shifts[i][0]);
                CHECK(neighbors.shifts[i][1] == expected_shifts[i][1]);
                CHECK(neighbors.shifts[i][2] == expected_shifts[i][2]);
            }

            if (!expected_distances.empty()) {
                CHECK_APPROX_EQUAL(neighbors.distances[i], expected_distances[i]);
            }

            if (!expected_vectors.empty()) {
                CHECK_APPROX_EQUAL(neighbors.vectors[i][0], expected_vectors[i][0]);
                CHECK_APPROX_EQUAL(neighbors.vectors[i][1], expected_vectors[i][1]);
                CHECK_APPROX_EQUAL(neighbors.vectors[i][2], expected_vectors[i][2]);
            }
        }
    } else {
        // Half lists: which point ends up as `i` vs `j` for a given pair is
        // not guaranteed to be stable (see `canonicalize`), so compare the
        // canonicalized records as an order-independent collection.
        auto actual = std::vector<Pair>(n);
        for (size_t i = 0; i < n; i++) {
            actual[i].i = neighbors.pairs[i][0];
            actual[i].j = neighbors.pairs[i][1];

            if (!expected_shifts.empty()) {
                actual[i].shift = std::array<int32_t, 3>{
                    neighbors.shifts[i][0],
                    neighbors.shifts[i][1],
                    neighbors.shifts[i][2]
                };
            }

            if (!expected_distances.empty()) {
                actual[i].distance = neighbors.distances[i];
            }

            if (!expected_vectors.empty()) {
                actual[i].vector = std::array<double, 3>{
                    neighbors.vectors[i][0],
                    neighbors.vectors[i][1],
                    neighbors.vectors[i][2]
                };
            }

            actual[i].canonicalize();
        }

        auto expected = std::vector<Pair>(n);
        for (size_t i = 0; i < n; i++) {
            expected[i].i = expected_pairs[i][0];
            expected[i].j = expected_pairs[i][1];

            if (!expected_shifts.empty()) {
                expected[i].shift = expected_shifts[i];
            }

            if (!expected_distances.empty()) {
                expected[i].distance = expected_distances[i];
            }

            if (!expected_vectors.empty()) {
                expected[i].vector = expected_vectors[i];
            }

            expected[i].canonicalize();
        }

        std::sort(actual.begin(), actual.end(), by_canonical_key);
        std::sort(expected.begin(), expected.end(), by_canonical_key);

        for (size_t i = 0; i < n; i++) {
            CHECK(actual[i].i == expected[i].i);
            CHECK(actual[i].j == expected[i].j);

            if (!expected_shifts.empty()) {
                CHECK(actual[i].shift[0] == expected[i].shift[0]);
                CHECK(actual[i].shift[1] == expected[i].shift[1]);
                CHECK(actual[i].shift[2] == expected[i].shift[2]);
            }

            if (!expected_distances.empty()) {
                CHECK_APPROX_EQUAL(actual[i].distance, expected[i].distance);
            }

            if (!expected_vectors.empty()) {
                CHECK_APPROX_EQUAL(actual[i].vector[0], expected[i].vector[0]);
                CHECK_APPROX_EQUAL(actual[i].vector[1], expected[i].vector[1]);
                CHECK_APPROX_EQUAL(actual[i].vector[2], expected[i].vector[2]);
            }
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
    bool periodic[3] = {false, false, false};

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

TEST_CASE("FCC unit cell") {
    double points[][3] = {
        {0.0, 0.0, 0.0},
    };

    double box[3][3] = {
        {0.0, 1.5, 1.5},
        {1.5, 0.0, 1.5},
        {1.5, 1.5, 0.0},
    };
    bool periodic[3] = {true, true, true};

    auto expected_vectors = std::vector<std::array<double, 3>>{
        {0.0, 1.5, 1.5},
        {1.5, -1.5, 0.0},
        {1.5, 0.0, 1.5},
        {0.0, 1.5, -1.5},
        {1.5, 0.0, -1.5},
        {1.5, 1.5, 0.0},
    };

    auto expected_shifts = std::vector<std::array<int32_t, 3>>{
        {1, 0, 0},
        {-1, 1, 0},
        {0, 1, 0},
        {0, -1, 1},
        {-1, 0, 1},
        {0, 0, 1},
    };

    auto expected_pairs = std::vector<std::array<size_t, 2>>(6, {0, 0});
    auto expected_distances = std::vector<double>(6, 2.1213203435596424);

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
    bool periodic[3] = {true, true, true};

    auto expected_pairs = std::vector<std::array<size_t, 2>>{
        {0, 1},
        {0, 2},
        {3, 5},
        {3, 4},
    };
    auto expected_shifts = std::vector<std::array<int32_t, 3>>(4, {0, 0, 0});
    auto expected_distances = std::vector<double>(4, 2.0);

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
    bool periodic[3] = {true, true, true};

    auto expected_pairs = std::vector<std::array<size_t, 2>>(3, {0, 0});
    auto expected_distances = std::vector<double>(3, 0.5);
    auto expected_shifts = std::vector<std::array<int32_t, 3>>{
        {1, 0, 0},
        {0, 1, 0},
        {0, 0, 1},
    };
    auto expected_vectors = std::vector<std::array<double, 3>>{
        {0.5, 0.0, 0.0},
        {0.0, 0.5, 0.0},
        {0.0, 0.0, 0.5},
    };

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
    bool periodic[3] = {true, true, true};

    auto options = VesinOptions();
    options.cutoff = 6.4;
    options.full = false;
    options.sorted = false;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = false;
    options.return_vectors = false;

    VesinNeighborList neighbors;

    const char* error_message = nullptr;
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
            auto pair = Pair{
                neighbors.pairs[i][0],
                neighbors.pairs[i][1],
                {neighbors.shifts[i][0], neighbors.shifts[i][1], neighbors.shifts[i][2]},
                0.0,
                {0.0, 0.0, 0.0},
            };
            pair.canonicalize();

            if (pair.i != 0 || pair.j != 3) {
                continue;
            }

            if (pair.shift[0] == missing[0] && pair.shift[1] == missing[1] && pair.shift[2] == missing[2]) {
                found = true;
                break;
            }
        }
        REQUIRE(found);
    }

    vesin_free(&neighbors);
}

// TODO: tests for full NL
