# Minimal CUDA manylinux image for ARM64 wheel building
FROM quay.io/pypa/manylinux_2_28_aarch64

# Install only essential CUDA components for ARM64
ARG CUDA_VER="12-8"
ARG ARCH="aarch64"

# Install CUDA repository and minimal packages for ARM64
RUN yum install -y yum-utils && \
    yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/sbsa/cuda-rhel8.repo && \
    yum -y install --setopt=install_weak_deps=False \
    cuda-cudart-${CUDA_VER}.${ARCH} \
    cuda-cudart-devel-${CUDA_VER}.${ARCH} \
    cuda-nvcc-${CUDA_VER}.${ARCH} \
    cuda-crt-${CUDA_VER}.${ARCH} \
    cuda-nvtx-${CUDA_VER}.${ARCH} && \
    yum clean all && \
    rm -rf /var/cache/yum/* /tmp/* /var/tmp/*

# Add CUDA lib path
RUN echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/999_nvidia_cuda.conf

# Set minimal CUDA environment
ENV CUDA_HOME=/usr/local/cuda \
    CUDA_ROOT=/usr/local/cuda \
    PATH="/usr/local/cuda/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

CMD ["/bin/bash"]