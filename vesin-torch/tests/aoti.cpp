#include <torch/csrc/inductor/aoti_package/model_package_loader.h>
#undef CHECK // torch defines a CHECK macro, which conflicts with Catch2's

#include <catch2/catch_test_macros.hpp>

#include "utils.hpp"

TEST_CASE("AOTInductor") {
    load_vesin_torch();

    SECTION("energy") {
        auto model = torch::inductor::AOTIModelPackageLoader(VESIN_TORCH_TESTS_MODELS_DIR "/energy.pt2");

        for (const auto& ref: references()) {
            // run multiple times to re-use the state of the neighbor list
            for (int32_t repeat = 0; repeat < 2; repeat++) {
                auto outputs = model.run({ref.points, ref.box});
                REQUIRE(outputs.size() == 1);
                CHECK(torch::allclose(outputs[0], ref.energy));
            }
        }
    }

    SECTION("energy and forces") {
        auto model = torch::inductor::AOTIModelPackageLoader(VESIN_TORCH_TESTS_MODELS_DIR "/energy-forces.pt2");

        for (const auto& ref: references()) {
            for (int32_t repeat = 0; repeat < 2; repeat++) {
                auto outputs = model.run({ref.points, ref.box});
                REQUIRE(outputs.size() == 2);
                CHECK(torch::allclose(outputs[0], ref.energy));
                CHECK(torch::allclose(outputs[1], ref.forces));
            }
        }
    }
}
