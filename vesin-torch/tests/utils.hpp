#ifndef VESIN_TORCH_TESTS_UTILS_HPP
#define VESIN_TORCH_TESTS_UTILS_HPP

#include <cstdint>
#include <string>
#include <vector>

#include <torch/script.h>
#include <torch/torch.h>

#include <catch2/catch_test_macros.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#ifdef _WIN32
/// Convert an UTF-8 string to UTF-16, to use with the W variants of the
/// Windows API
inline std::wstring utf8_to_wide(const std::string& value) {
    auto size = MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, nullptr, 0);
    REQUIRE(size > 0);

    auto result = std::wstring(static_cast<size_t>(size), L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.c_str(), -1, &result[0], size);
    // remove the NULL terminator
    result.resize(static_cast<size_t>(size - 1));
    return result;
}
#endif

/// Load the vesin_torch library, registering the operators and classes it
/// defines. This must happen before loading the models.
inline void load_vesin_torch() {
    static bool loaded = [](){
#ifdef _WIN32
        return LoadLibraryW(utf8_to_wide(VESIN_TORCH_LIBRARY).c_str()) != nullptr;
#else
        return dlopen(VESIN_TORCH_LIBRARY, RTLD_NOW) != nullptr;
#endif
    }();
    REQUIRE(loaded);
}


/// Inputs and reference outputs computed in Python by `generate-models.py`
struct Reference {
    torch::Tensor points;
    torch::Tensor box;
    torch::Tensor energy;
    torch::Tensor forces;
};

inline std::vector<Reference> references() {
    auto reference_model = torch::jit::load(VESIN_TORCH_TESTS_MODELS_DIR "/reference.pt");
    auto n_references = reference_model.attr("n_references").toInt();
    REQUIRE(n_references > 0);

    auto systems = std::vector<Reference>();
    for (int64_t i = 0; i < n_references; i++) {
        auto suffix = "_" + std::to_string(i);
        systems.push_back(Reference{
            reference_model.attr("points" + suffix).toTensor(),
            reference_model.attr("box" + suffix).toTensor(),
            reference_model.attr("energy" + suffix).toTensor(),
            reference_model.attr("forces" + suffix).toTensor(),
        });
    }

    return systems;
}

#endif
