#include <cstring>
#include <string>

#include "vesin.h"

#include "cpu_cell_list.hpp"

thread_local std::string LAST_ERROR;

extern "C" int vesin_neighbors(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    bool periodic,
    VesinDevice device,
    VesinOptions options,
    VesinNeighborList* neighbors,
    const char** error_message
) {
    if (error_message == nullptr) {
        return EXIT_FAILURE;
    }

    if (points == nullptr) {
        *error_message = "`points` can not be a NULL pointer";
        return EXIT_FAILURE;
    }

    if (box == nullptr) {
        *error_message = "`cell` can not be a NULL pointer";
        return EXIT_FAILURE;
    }

    if (neighbors == nullptr) {
        *error_message = "`neighbors` can not be a NULL pointer";
        return EXIT_FAILURE;
    }

    if (neighbors->device != VesinUnknownDevice && neighbors->device != device) {
        *error_message = "`neighbors` device and data `device` do not match, free the neighbors first";
        return EXIT_FAILURE;
    }

    if (device == VesinUnknownDevice) {
        *error_message = "got an unknown device to use when running simulation";
        return EXIT_FAILURE;
    }

    if (neighbors->device == VesinUnknownDevice) {
        // initialize the device
        neighbors->device = device;
    } else if (neighbors->device != device) {
        *error_message = "`neighbors.device` and `device` do not match, free the neighbors first";
        return EXIT_FAILURE;
    }

    try {
        if (device == VesinCPU) {
            auto matrix = vesin::Matrix{{{
                {{box[0][0], box[0][1], box[0][2]}},
                {{box[1][0], box[1][1], box[1][2]}},
                {{box[2][0], box[2][1], box[2][2]}},
            }}};

            vesin::cpu::neighbors(
                reinterpret_cast<const vesin::Vector*>(points),
                n_points,
                vesin::BoundingBox(matrix, periodic),
                options,
                *neighbors
            );
        } else {
            throw std::runtime_error("unknown device " + std::to_string(device));
        }
    } catch (const std::bad_alloc&) {
        LAST_ERROR = "failed to allocate memory";
        *error_message = LAST_ERROR.c_str();
        return EXIT_FAILURE;
    } catch (const std::exception& e) {
        LAST_ERROR = e.what();
        *error_message = LAST_ERROR.c_str();
        return EXIT_FAILURE;
    } catch (...) {
        *error_message = "fatal error: unknown type thrown as exception";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


extern "C" void vesin_free(VesinNeighborList* neighbors) {
    if (neighbors == nullptr) {
        return;
    }

    if (neighbors->device == VesinUnknownDevice) {
        // nothing to do
    } else if (neighbors->device == VesinCPU) {
        vesin::cpu::free_neighbors(*neighbors);
    }

    std::memset(neighbors, 0, sizeof(VesinNeighborList));
}
