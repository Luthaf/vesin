#include <torch/script.h>
#include <torch/torch.h>
#undef CHECK // torch defines a CHECK macro, which conflicts with Catch2's

#include <catch2/catch_test_macros.hpp>

#include "utils.hpp"

TEST_CASE("TorchScript energy and forces") {
    load_vesin_torch();
    auto model = torch::jit::load(VESIN_TORCH_TESTS_MODELS_DIR "/energy.pt");

    for (const auto& ref: references()) {
        for (int32_t repeat = 0; repeat < 2; repeat++) {
            auto points = ref.points.clone().requires_grad_(true);
            auto energy = model.forward({points, ref.box}).toTensor();
            auto forces = -torch::autograd::grad({energy}, {points})[0];

            CHECK(torch::allclose(energy, ref.energy));
            CHECK(torch::allclose(forces, ref.forces));
        }
    }
}
