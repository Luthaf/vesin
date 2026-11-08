#ifndef VESIN_CUDA_DEVICE_GUARD_HPP
#define VESIN_CUDA_DEVICE_GUARD_HPP

#include <cstdint>

#include <gpulite/gpulite.hpp>

namespace vesin::cuda {

/// Guard that sets the current device to `device_id` on construction and
/// restores the previous device on destruction. vesin allocates memory on the
/// current device, so this must be set before any allocation.
class DeviceGuard {
public:
    explicit DeviceGuard(int32_t device_id) {
        GPULITE_CUDART_CALL(cudaGetDevice(&this->previous_));
        if (this->previous_ != device_id) {
            GPULITE_CUDART_CALL(cudaSetDevice(device_id));
        } else {
            this->previous_ = -1;
        }
    }

    ~DeviceGuard() {
        if (this->previous_ >= 0) {
            gpulite::CUDART::instance().cudaSetDevice(this->previous_);
        }
    }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;

private:
    int32_t previous_ = 0;
};

} // namespace vesin::cuda

#endif // VESIN_CUDA_DEVICE_GUARD_HPP
