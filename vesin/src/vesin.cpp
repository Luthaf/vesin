#include <cstring>
#include <iostream>
#include <string>

#include "cpu_cell_list.hpp"
#include "vesin.h"
#include "vesin_cuda.hpp"
#include "verlet.hpp"

// used to store dynamically allocated error messages before giving a pointer
// to them back to the user
thread_local std::string LAST_ERROR;

static void free_cpu_verlet_state(VesinNeighborList* neighbors) {
    if (neighbors->device.type == VesinCPU && neighbors->opaque != nullptr) {
        delete static_cast<vesin::VerletState*>(neighbors->opaque);
        neighbors->opaque = nullptr;
    }
}

extern "C" int vesin_neighbors(
    const double (*points)[3],
    size_t n_points,
    const double box[3][3],
    const bool periodic[3],
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

    if (!std::isfinite(options.cutoff) || options.cutoff <= 0) {
        *error_message = "cutoff must be a finite, positive number";
        return EXIT_FAILURE;
    }

    if (options.cutoff <= 1e-6) {
        *error_message = "cutoff is too small";
        return EXIT_FAILURE;
    }

    if (!std::isfinite(options.skin) || options.skin < 0.0) {
        *error_message = "skin must be a finite, non-negative number";
        return EXIT_FAILURE;
    }

    if (neighbors->device.type != VesinUnknownDevice && neighbors->device.type != device.type) {
        *error_message = "`neighbors` device and data `device` do not match, free the neighbors first";
        return EXIT_FAILURE;
    }

    if (device.type == VesinUnknownDevice) {
        *error_message = "got an unknown device type";
        return EXIT_FAILURE;
    }

    if (neighbors->device.type == VesinUnknownDevice) {
        // initialize the device
        neighbors->device = device;
    } else if (neighbors->device.type != device.type) {
        *error_message = "`neighbors.device` and `device` do not match, free the neighbors first";
        return EXIT_FAILURE;
    }

    try {
        if (options.skin > 0.0) {
            if (device.type != VesinCPU) {
                *error_message = "Verlet caching with skin > 0 is only supported on CPU";
                return EXIT_FAILURE;
            }

            if (options.algorithm != VesinAutoAlgorithm && options.algorithm != VesinCellList) {
                *error_message = "only VesinAutoAlgorithm and VesinCellList are supported on CPU";
                return EXIT_FAILURE;
            }

            if (neighbors->opaque == nullptr) {
                neighbors->opaque = new vesin::VerletState();
            }

            auto* state = static_cast<vesin::VerletState*>(neighbors->opaque);
            vesin::verlet_set_options(*state, options);

            if (vesin::verlet_needs_rebuild(*state, points, n_points, box, periodic)) {
                vesin::verlet_rebuild(*state, points, n_points, box, periodic);
            } else {
                state->did_rebuild_flag = false;
            }

            vesin::verlet_recompute(*state, points, box, options, *neighbors);
            return EXIT_SUCCESS;
        }

        if (device.type == VesinCPU) {
            free_cpu_verlet_state(neighbors);

            auto matrix = vesin::Matrix{{{
                {{box[0][0], box[0][1], box[0][2]}},
                {{box[1][0], box[1][1], box[1][2]}},
                {{box[2][0], box[2][1], box[2][2]}},
            }}};

            auto box = vesin::BoundingBox(matrix, periodic);
            box.make_bounding_for(points, n_points);

            vesin::cpu::neighbors(
                reinterpret_cast<const vesin::Vector*>(points),
                n_points,
                std::move(box),
                options,
                *neighbors
            );
        } else if (device.type == VesinCUDA) {
            vesin::cuda::neighbors(
                points,
                n_points,
                box,
                periodic,
                options,
                *neighbors
            );
        } else {
            throw std::runtime_error("unknown device " + std::to_string(device.type));
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

    try {
        if (neighbors->device.type == VesinUnknownDevice) {
            // nothing to do
        } else if (neighbors->device.type == VesinCPU) {
            free_cpu_verlet_state(neighbors);
            vesin::cpu::free_neighbors(*neighbors);
        } else if (neighbors->device.type == VesinCUDA) {
            vesin::cuda::free_neighbors(*neighbors);
        } else {
            throw std::runtime_error("unknown device " + std::to_string(neighbors->device.type) + " when freeing memory");
        }
    } catch (const std::exception& e) {
        std::cerr << "error in vesin_free: " << e.what() << std::endl;
        return;
    } catch (...) {
        std::cerr << "fatal error in vesin_free, unknown type thrown as exception" << std::endl;
        return;
    }

    std::memset(neighbors, 0, sizeof(VesinNeighborList));
}
