#include <cstdio>
#include <cstring>
#include <string>

#include "cpu_cell_list.hpp"
#include "vesin.h"
#include "vesin_cuda.hpp"
#include "verlet.hpp"

// used to store dynamically allocated error messages before giving a pointer
// to them back to the user
thread_local std::string LAST_ERROR;

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

    // Validate Verlet options
    if (options.skin < 0) {
        *error_message = "skin must be non-negative";
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

    // Check if Verlet caching is enabled
    bool use_verlet = options.skin > 0;

    try {
        if (use_verlet) {
            // Verlet caching path -- CPU only in this PR
            if (device.type != VesinCPU) {
                LAST_ERROR = "Verlet caching (skin > 0) is only supported on CPU";
                *error_message = LAST_ERROR.c_str();
                return EXIT_FAILURE;
            }

            // Create VerletState if not present
            if (neighbors->opaque == nullptr || !neighbors->verlet_mode) {
                auto* state = new vesin::VerletState();
                state->cutoff = options.cutoff;
                state->skin = options.skin;
                state->half_skin_sq = (options.skin / 2.0) * (options.skin / 2.0);
                state->full_list = options.full;
                state->n_points = 0;
                state->n_pairs = 0;
                state->did_rebuild_flag = false;
                state->has_cache = false;
                std::memset(state->ref_box, 0, sizeof(state->ref_box));
                for (int d = 0; d < 3; d++) {
                    state->ref_periodic[d] = false;
                }
                neighbors->opaque = static_cast<void*>(state);
                neighbors->verlet_mode = true;
            }

            auto* state = static_cast<vesin::VerletState*>(neighbors->opaque);

            // CPU path
            bool needs_rebuild = vesin::verlet_needs_rebuild(
                *state, points, n_points, box, periodic
            );

            if (needs_rebuild) {
                vesin::verlet_rebuild(*state, points, n_points, box, periodic);
            } else {
                state->did_rebuild_flag = false;
            }

            vesin::verlet_recompute(*state, points, box, options, *neighbors);

        } else {
            // Stateless path (original behavior)
            if (device.type == VesinCPU) {
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
        // Free VerletState if present (only when verlet_mode is true)
        if (neighbors->verlet_mode && neighbors->opaque != nullptr) {
            auto* state = static_cast<vesin::VerletState*>(neighbors->opaque);
            delete state;
            neighbors->opaque = nullptr;
        }

        if (neighbors->device.type == VesinUnknownDevice) {
            // nothing to do
        } else if (neighbors->device.type == VesinCPU) {
            vesin::cpu::free_neighbors(*neighbors);
        } else if (neighbors->device.type == VesinCUDA) {
            vesin::cuda::free_neighbors(*neighbors);
        } else {
            throw std::runtime_error("unknown device " + std::to_string(neighbors->device.type) + " when freeing memory");
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error in vesin_free: %s\n", e.what());
        return;
    } catch (...) {
        std::fprintf(stderr, "fatal error in vesin_free, unknown type thrown as exception\n");
        return;
    }

    std::memset(neighbors, 0, sizeof(VesinNeighborList));
}
