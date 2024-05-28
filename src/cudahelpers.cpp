#include <cudahelpers.hpp>
#include <cuda_runtime.h>
#include <iostream>

#include <cuda.h>

namespace SLCuda {

CudaStream::CudaStream(cudaStream_t stream)
    : stream(stream) {}

tl::expected<uint64_t, cudaError_t> CudaStream::getId() const {
    uint64_t id;
    cudaError_t err;
    return ((err = cudaStreamGetId(stream, &id)) != cudaSuccess) ? id : err;
}

cudaStream_t CudaStream::handle() const {
    return stream;
}

tl::expected<CudaStream, cudaError_t> CudaStream::create() {
    cudaStream_t stream;
    cudaError_t err;
    if ((err = cudaStreamCreate(&stream)) != cudaSuccess) {
        return tl::make_unexpected(err);
    }

    return CudaStream(std::move(stream));
}

CudaExternalMemory::CudaExternalMemory(size_t size, cudaExternalMemoryHandleType handleType, cudaExternalMemory_t raw, void* data)
    : size(size), handleType(handleType), raw(raw), data(data) {}

void* CudaExternalMemory::getDataPointer() const {
    return data;
}

size_t CudaExternalMemory::getSize() const {
    return size;
}


#ifdef _WIN64
tl::expected<CudaExternalMemory, cudaError_t> CudaExternalMemory::create(HANDLE handle, size_t size, cudaExternalMemoryHandleType handleType)
{
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {};
    externalMemoryHandleDesc.type = handleType;
    externalMemoryHandleDesc.handle.win32.handle = handle;
    externalMemoryHandleDesc.size = size;

    cudaExternalMemory_t raw = nullptr;
    cudaError_t err;

    if ((err = cudaImportExternalMemory(&raw, &externalMemoryHandleDesc)) != cudaSuccess)
        return tl::unexpected(err);

    cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
    externalMemBufferDesc.offset = 0;
    externalMemBufferDesc.size = size;
    externalMemBufferDesc.flags = 0;

    void* dataPtr = nullptr;

    if ((err = cudaExternalMemoryGetMappedBuffer(&dataPtr, raw, &externalMemBufferDesc)) != cudaSuccess)
        return tl::unexpected(err);

    return CudaExternalMemory(size, handleType, raw, dataPtr);
}
#endif

CudaExternalSemaphore::CudaExternalSemaphore(cudaExternalSemaphore_t raw, cudaExternalSemaphoreHandleType handleType)
    : raw(raw), handleType(handleType) {}


tl::expected<CudaExternalSemaphore, cudaError_t> CudaExternalSemaphore::create(HANDLE handle, cudaExternalSemaphoreHandleType handleType) {
    cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {};
	externalSemaphoreHandleDesc.type = handleType;
	externalSemaphoreHandleDesc.flags = 0;
	externalSemaphoreHandleDesc.handle.win32.handle = handle;

    cudaExternalSemaphore_t raw;
    cudaError_t err;

    if ((err = cudaImportExternalSemaphore(&raw, &externalSemaphoreHandleDesc)) != cudaSuccess) 
        return tl::unexpected(err);

    return CudaExternalSemaphore(raw, handleType);
}

tl::expected<void, cudaError_t> CudaExternalSemaphore::wait(CudaStream& stream, uint64_t waitValue) {
    cudaExternalSemaphoreWaitParams extSemaphoreWaitParams = {};
    extSemaphoreWaitParams.flags = 0;
    extSemaphoreWaitParams.params.fence.value = waitValue;

    cudaError_t err;

    if ((err = cudaWaitExternalSemaphoresAsync(&raw, &extSemaphoreWaitParams, 1, stream.handle())) != cudaSuccess)
        return tl::unexpected(err);

    return {};
}

tl::expected<void, cudaError_t> CudaExternalSemaphore::signal(CudaStream& stream, uint64_t signalValue) {
    cudaExternalSemaphoreSignalParams extSemaphoreSignalParams = {};
    extSemaphoreSignalParams.flags = 0;
    extSemaphoreSignalParams.params.fence.value = signalValue;

    cudaError_t err;
    if ((err = cudaSignalExternalSemaphoresAsync(&raw, &extSemaphoreSignalParams, 1, stream.handle())) != cudaSuccess)
        return tl::unexpected(err);

    return {};
}

}