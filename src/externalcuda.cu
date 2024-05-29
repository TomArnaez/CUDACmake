#include <format>

#include <core.hpp>
#include <cudahelpers.hpp>
#include <externalcuda.hpp>

namespace SLCuda {

ExternalCuda::ExternalCuda(
    uint32_t width,
    uint32_t height,
    CudaExternalMemory inputExternalMemory,
    CudaExternalMemory outputExternalMemory,
    CudaExternalSemaphore timelineSemaphore
) :
    width(width),
    height(height),
    inputExternalMemory(inputExternalMemory),
    outputExternalMemory(outputExternalMemory),
    timelineSemaphore(timelineSemaphore),
    core(Core::create(width, height).value()) {}

tl::expected<ExternalCuda, std::string> ExternalCuda::create(ExternalCudaCreateInfo createInfo) {
    CudaExternalMemory inputExternalMemory = CudaExternalMemory::create(
        createInfo.inputBufferMemoryHandle,
        createInfo.bufferMemorySize,
        cudaExternalMemoryHandleTypeOpaqueWin32
    ).map_error([](cudaError_t err) {
        std::cout << 2 << std::endl;
        return tl::unexpected(std::string("Failed to import input external memory with error: ") + cudaGetErrorString(err));
    }).value();

    CudaExternalMemory outputExternalMemory = CudaExternalMemory::create(
        createInfo.outputBufferMemoryHandle,
        createInfo.bufferMemorySize,
        cudaExternalMemoryHandleTypeOpaqueWin32
    ).map_error([](cudaError_t err) {
        return tl::unexpected(std::string("Failed to import output external memory with error: ") + cudaGetErrorString(err));
    }).value();

    CudaExternalSemaphore timelineSemaphore = CudaExternalSemaphore::create(
        createInfo.timelineSemaphoreHandle,
        cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32
    ).map_error([](cudaError_t err) {
        return tl::unexpected(std::string("Failed to import external semaphore with error: ") + cudaGetErrorString(err));
    }).value();

    return ExternalCuda(
        createInfo.width,
        createInfo.height,
        inputExternalMemory,
        outputExternalMemory,
        timelineSemaphore
    );
}

tl::expected<void, std::string> ExternalCuda::run(uint64_t waitValue, uint64_t signalValue) {
    timelineSemaphore.wait(core.getStream(), waitValue)
        .map_error([](cudaError_t err) {
            return tl::unexpected(std::string("Got err whilst waiting: {}") + cudaGetErrorString(err));
        });

    cudaMemcpyAsync(
        (thrust::raw_pointer_cast(core.getBuffer().data())),
        inputExternalMemory.getDataPointer(),
        inputExternalMemory.getSize(),
        cudaMemcpyDeviceToDevice,
        core.getStream().handle()
    );

    core.run();

    cudaMemcpyAsync(
        outputExternalMemory.getDataPointer(),
        (thrust::raw_pointer_cast(core.getBuffer().data())),
        inputExternalMemory.getSize(),
        cudaMemcpyDeviceToDevice,
        core.getStream().handle()
    );

    timelineSemaphore.signal(core.getStream(), signalValue)
        .map_error([](cudaError_t err) {
            return tl::unexpected(std::string("Got err whilst signalling: {}") + cudaGetErrorString(err));
        });

    return {};
}

void ExternalCuda::setDarkCorrection(std::span<unsigned short> darkMap, unsigned short offset) {
    core.addCorrection(std::make_shared<DarkCorrection>(darkMap, offset));
}

void ExternalCuda::setGainCorrection(std::span<unsigned short> gainMap) {
    core.addCorrection(std::make_shared<GainCorrection>(gainMap));
}

void ExternalCuda::setDefectCorrection(std::span<unsigned short> defectMap) {
    core.addCorrection(std::make_shared<DefectCorrection>(defectMap, width, height));
}

void ExternalCuda::setHistogramEquailisation(bool enable) {
    core.addCorrection(std::make_shared<HistogramEquilisation>(width, height));
}

}
