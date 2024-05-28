#pragma once

#include <cuda_runtime.h>
#include <tl/expected.hpp>

#ifdef _WIN64
#include <Windows.h>
#endif

namespace SLCuda {

/*
 * RAII-style wrappers for CUDA objects
 */

class CudaStream {
    cudaStream_t stream;

    CudaStream(cudaStream_t stream);
public:
    static tl::expected<CudaStream, cudaError_t> create();

    tl::expected<uint64_t, cudaError_t> getId() const;
    cudaStream_t handle() const;
};

class CudaExternalMemory {
    cudaExternalMemoryHandleType handleType;
    cudaExternalMemory_t raw;
    size_t size;
    void* data;

    CudaExternalMemory(size_t size, cudaExternalMemoryHandleType handleType, cudaExternalMemory_t raw, void* data);
public:
    static tl::expected<CudaExternalMemory, cudaError_t> create(HANDLE handle, size_t size, cudaExternalMemoryHandleType handleType);

    void* getDataPointer() const;
    size_t getSize() const;
};

class CudaExternalSemaphore {
    cudaExternalSemaphoreHandleType handleType;
    cudaExternalSemaphore_t raw;

    CudaExternalSemaphore(cudaExternalSemaphore_t raw, cudaExternalSemaphoreHandleType handleType);
public:
    static tl::expected<CudaExternalSemaphore, cudaError_t> create(HANDLE handle, cudaExternalSemaphoreHandleType handleType);

    tl::expected<void, cudaError_t> wait(CudaStream& stream, uint64_t waitValue);
    tl::expected<void, cudaError_t> signal(CudaStream& stream, uint64_t signalValue);
};

}