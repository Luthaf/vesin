#ifndef VESIN_CUDA_UTILS_HPP
#define VESIN_CUDA_UTILS_HPP

#include <cstdint>

#include <gpulite/gpulite.hpp>

namespace vesin::cuda {

/// Guard that sets the current device to `device_id` on construction and
/// restores the previous device on destruction.
///
/// The CUDA runtime allocates memory on the current device, and resolves the
/// default streams (`NULL`, `cudaStreamLegacy` and `cudaStreamPerThread`) to
/// the ones of the current device, so this must be active whenever vesin works
/// with data on a given device.
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
            // ignore errors, we can not throw from a destructor
            gpulite::CUDART::instance().cudaSetDevice(this->previous_);
        }
    }

    DeviceGuard(const DeviceGuard&) = delete;
    DeviceGuard& operator=(const DeviceGuard&) = delete;
    DeviceGuard(DeviceGuard&&) = delete;
    DeviceGuard& operator=(DeviceGuard&&) = delete;

private:
    int32_t previous_ = -1;
};

} // namespace vesin::cuda

#endif // VESIN_CUDA_UTILS_HPP
