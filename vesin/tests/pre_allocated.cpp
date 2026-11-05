#include <cstdint>
#include <cstring>

#include <array>
#include <string>
#include <vector>

#include <catch2/catch_test_macros.hpp>

#include <vesin.h>

/// Simple cubic lattice of 4x4x4 points in a periodic box. With a cutoff of
/// 1.5, each point has 18 neighbors.
struct System {
    System() {
        for (size_t x = 0; x < 4; x++) {
            for (size_t y = 0; y < 4; y++) {
                for (size_t z = 0; z < 4; z++) {
                    points.push_back({static_cast<double>(x), static_cast<double>(y), static_cast<double>(z)});
                }
            }
        }
    }

    std::vector<std::array<double, 3>> points;
    double box[3][3] = {
        {4.0, 0.0, 0.0},
        {0.0, 4.0, 0.0},
        {0.0, 0.0, 4.0},
    };
    bool periodic[3] = {true, true, true};

    /// Run the calculation, returning the error message or an empty string
    std::string compute(VesinNeighborList& neighbors, VesinOptions options) {
        const char* error_message = nullptr;
        auto status = vesin_neighbors(
            reinterpret_cast<const double (*)[3]>(points.data()),
            points.size(),
            box,
            periodic,
            {VesinCPU, 0},
            options,
            &neighbors,
            &error_message
        );

        if (status != EXIT_SUCCESS) {
            REQUIRE(error_message != nullptr);
            return error_message;
        }
        return "";
    }
};

/// Copy of the data in a `VesinNeighborList`
struct Results {
    std::vector<std::array<size_t, 2>> pairs;
    std::vector<std::array<int32_t, 3>> shifts;
    std::vector<double> distances;
    std::vector<std::array<double, 3>> vectors;

    Results(const VesinNeighborList& neighbors, VesinOptions options) {
        for (size_t i = 0; i < neighbors.length; i++) {
            pairs.push_back({neighbors.pairs[i][0], neighbors.pairs[i][1]});
            if (options.return_shifts) {
                shifts.push_back({neighbors.shifts[i][0], neighbors.shifts[i][1], neighbors.shifts[i][2]});
            }
            if (options.return_distances) {
                distances.push_back(neighbors.distances[i]);
            }
            if (options.return_vectors) {
                vectors.push_back({neighbors.vectors[i][0], neighbors.vectors[i][1], neighbors.vectors[i][2]});
            }
        }
    }

    bool operator==(const Results& other) const {
        return pairs == other.pairs && shifts == other.shifts &&
               distances == other.distances && vectors == other.vectors;
    }
};

/// Output arrays owned by the test
struct Arrays {
    explicit Arrays(size_t capacity):
        pairs(capacity),
        shifts(capacity),
        distances(capacity),
        vectors(capacity) {}

    /// Set all the arrays in `neighbors`, with the given capacity
    void set(VesinNeighborList& neighbors, size_t capacity) {
        neighbors.pairs = reinterpret_cast<size_t (*)[2]>(pairs.data());
        neighbors.shifts = reinterpret_cast<int32_t (*)[3]>(shifts.data());
        neighbors.distances = distances.data();
        neighbors.vectors = reinterpret_cast<double (*)[3]>(vectors.data());
        neighbors.length = capacity;
    }

    std::vector<std::array<size_t, 2>> pairs;
    std::vector<std::array<int32_t, 3>> shifts;
    std::vector<double> distances;
    std::vector<std::array<double, 3>> vectors;
};

static VesinOptions default_options() {
    auto options = VesinOptions();
    options.cutoff = 1.5;
    options.full = true;
    options.sorted = true;
    options.algorithm = VesinAutoAlgorithm;
    options.return_shifts = true;
    options.return_distances = true;
    options.return_vectors = true;
    return options;
}

/// Compute the reference results with vesin-allocated outputs
static Results reference(System& system, VesinOptions options) {
    options.pre_allocated_outputs = false;

    auto neighbors = VesinNeighborList();
    REQUIRE(system.compute(neighbors, options).empty());
    auto results = Results(neighbors, options);
    vesin_free(&neighbors);
    return results;
}

TEST_CASE("Pre-allocated output arrays") {
    auto system = System();
    auto n_points = system.points.size();

    for (double skin : {0.0, 0.3}) {
        for (size_t padded_to : {0, 20}) {
            auto options = default_options();
            options.skin = skin;
            options.padded_to = padded_to;
            options.pre_allocated_outputs = true;

            auto expected = reference(system, options);
            auto n_pairs = expected.pairs.size();
            REQUIRE(n_pairs == (padded_to > 0 ? n_points * padded_to : n_points * 18));

            // a bit more room than needed
            auto capacity = n_pairs + 10;
            auto arrays = Arrays(capacity);
            auto neighbors = VesinNeighborList();

            // vesin writes the pairs in the arrays
            arrays.set(neighbors, capacity);
            REQUIRE(system.compute(neighbors, options).empty());
            CHECK(neighbors.pairs == reinterpret_cast<size_t (*)[2]>(arrays.pairs.data()));
            CHECK(neighbors.length == n_pairs);
            CHECK(Results(neighbors, options) == expected);

            // calling again with the same arrays works
            std::memset(arrays.distances.data(), 0, arrays.distances.size() * sizeof(double));
            arrays.set(neighbors, capacity);
            REQUIRE(system.compute(neighbors, options).empty());
            CHECK(Results(neighbors, options) == expected);

            // exactly the right size
            arrays.set(neighbors, n_pairs);
            REQUIRE(system.compute(neighbors, options).empty());
            CHECK(Results(neighbors, options) == expected);

            // too small
            arrays.set(neighbors, n_pairs - 1);
            CHECK(system.compute(neighbors, options) == (
                "the pre-allocated output arrays can hold " + std::to_string(n_pairs - 1) +
                " pairs, but this calculation needs " + std::to_string(n_pairs) + " pairs"
            ));

            // one of the requested arrays is missing
            arrays.set(neighbors, capacity);
            neighbors.vectors = nullptr;
            CHECK(system.compute(neighbors, options) == "`options.pre_allocated_outputs` is set, but `neighbors.vectors` is NULL");

            // arrays which are not requested do not need to be set
            options.return_vectors = false;
            REQUIRE(system.compute(neighbors, options).empty());
            CHECK(Results(neighbors, options) == reference(system, options));

            // vesin never releases pre-allocated arrays (this would crash or be
            // reported by sanitizers)
            vesin_free(&neighbors);
        }
    }
}

TEST_CASE("Switching between pre-allocated and vesin-allocated outputs") {
    auto system = System();
    auto options = default_options();
    auto n_pairs = system.points.size() * 18;
    auto arrays = Arrays(n_pairs);

    auto neighbors = VesinNeighborList();
    REQUIRE(system.compute(neighbors, options).empty());

    SECTION("Output arrays can not be modified without pre-allocated outputs") {
        auto* pairs = neighbors.pairs;
        auto expected_error = std::string(
            "`neighbors.pairs` is not the array allocated by vesin in the last calculation. "
            "Output arrays can only be set by the caller with `options.pre_allocated_outputs`, "
            "and must be set back to NULL before calculations without it"
        );

        neighbors.pairs = nullptr;
        CHECK(system.compute(neighbors, options) == expected_error);

        neighbors.pairs = reinterpret_cast<size_t (*)[2]>(arrays.pairs.data());
        CHECK(system.compute(neighbors, options) == expected_error);

        neighbors.pairs = pairs;
        REQUIRE(system.compute(neighbors, options).empty());
        CHECK(neighbors.pairs == pairs);
    }

    SECTION("Arrays allocated by vesin can not be used as pre-allocated arrays") {
        options.pre_allocated_outputs = true;
        neighbors.length = n_pairs;
        CHECK(system.compute(neighbors, options) == "`neighbors.pairs` was allocated by vesin, it can not be used as a pre-allocated output array");
    }

    SECTION("Switching back and forth") {
        options.pre_allocated_outputs = true;
        arrays.set(neighbors, n_pairs);
        REQUIRE(system.compute(neighbors, options).empty());
        CHECK(neighbors.pairs == reinterpret_cast<size_t (*)[2]>(arrays.pairs.data()));

        // the pre-allocated arrays must be removed before going back to
        // vesin-allocated outputs
        options.pre_allocated_outputs = false;
        CHECK(system.compute(neighbors, options) == (
            "`neighbors.pairs` is not the array allocated by vesin in the last calculation. "
            "Output arrays can only be set by the caller with `options.pre_allocated_outputs`, "
            "and must be set back to NULL before calculations without it"
        ));

        neighbors.pairs = nullptr;
        neighbors.shifts = nullptr;
        neighbors.distances = nullptr;
        neighbors.vectors = nullptr;
        REQUIRE(system.compute(neighbors, options).empty());
        CHECK(neighbors.pairs != nullptr);
        CHECK(neighbors.pairs != reinterpret_cast<size_t (*)[2]>(arrays.pairs.data()));
        CHECK(Results(neighbors, options) == reference(system, options));
    }

    vesin_free(&neighbors);
}
