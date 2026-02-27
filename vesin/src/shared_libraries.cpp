#include <cstdlib>

#include <mutex>
#include <string>
#include <vector>

#include "vesin.h"

#if defined(__linux__)

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <dlfcn.h>
#include <link.h>

static int phdr_callback(struct dl_phdr_info* info, size_t, void* data) {
    auto* result = reinterpret_cast<std::vector<std::string>*>(data);
    result->emplace_back(info->dlpi_name);
    return 0;
}

static void list_libraries_impl(std::vector<std::string>& result) {
    dl_iterate_phdr(phdr_callback, &result);
}

#elif defined(__APPLE__)

#include <dlfcn.h>
#include <mach-o/dyld.h>

static void list_libraries_impl(std::vector<std::string>& result) {
    auto count = _dyld_image_count();
    for (uint32_t i = 0; i < count; i++) {
        result.emplace_back(_dyld_get_image_name(i));
    }
}

#elif defined(_WIN32)

// clang-format off
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <Windows.h>
#include <TlHelp32.h>
// clang-format on

#include <cstring>

static void list_libraries_impl(std::vector<std::string>& result) {
    auto handle = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, 0);

    if (handle == INVALID_HANDLE_VALUE) {
        auto error = GetLastError();
        throw std::runtime_error(
            "failed to get a process snapshot, error code " + std::to_string(error)
        );
    }

    MODULEENTRY32 module;
    std::memset(&module, 0, sizeof(MODULEENTRY32));
    module.dwSize = sizeof(MODULEENTRY32);

    if (!Module32First(handle, &module)) {
        auto error = GetLastError();
        CloseHandle(handle);
        throw std::runtime_error(
            "failed to get the first module in process, error code " + std::to_string(error)
        );
    }

    do {
        result.emplace_back(module.szExePath);
    } while (Module32Next(handle, &module));

    CloseHandle(handle);
}

#else

#error "Unsupported OS, please add it to this file!"

#endif

static std::mutex MUTEX;
static std::vector<std::string> LOADED_LIBRARIES;

extern "C" void VESIN_API vesin_list_libraries(const char* libraries[], size_t* libraries_count) {
    // prevent simultaneous calls to this function
    auto guard = std::lock_guard<std::mutex>(MUTEX);

    if (libraries_count == nullptr || libraries == nullptr) {
        return;
    }

    LOADED_LIBRARIES.clear();
    list_libraries_impl(LOADED_LIBRARIES);

    size_t copy_count = std::min(*libraries_count, LOADED_LIBRARIES.size());
    for (size_t i = 0; i < copy_count; ++i) {
        libraries[i] = LOADED_LIBRARIES[i].c_str();
    }

    // Always return the total available count so the caller knows if they need a bigger buffer
    *libraries_count = LOADED_LIBRARIES.size();
}
