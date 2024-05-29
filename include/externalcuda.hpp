#pragma once

#include <memory>
#include <cstdint>
#include <wtypes.h>
#include <span>
#include <string>

#include <core.hpp>
#include <tl/expected.hpp>

namespace SLCuda {

class Core;

struct ExternalCudaCreateInfo {
	uint32_t width;
	uint32_t height;
	HANDLE timelineSemaphoreHandle;
    HANDLE inputBufferMemoryHandle;
	HANDLE outputBufferMemoryHandle;
    size_t bufferMemorySize;
};

class ExternalCuda {
    Core core;
    uint32_t width;
    uint32_t height;
    CudaExternalMemory inputExternalMemory;
    CudaExternalMemory outputExternalMemory;
    CudaExternalSemaphore timelineSemaphore;

    ExternalCuda(
        uint32_t width,
        uint32_t height,
        CudaExternalMemory inputExternalBuffer,
        CudaExternalMemory outputExternalBuffer,
        CudaExternalSemaphore timelineSemaphore
    );
public:
    __declspec(dllexport) static tl::expected<ExternalCuda, std::string> create(ExternalCudaCreateInfo createInfo);

    __declspec(dllexport) void setDarkCorrection(std::span<unsigned short> darkMap, unsigned short offset);
	__declspec(dllexport) void setGainCorrection(std::span<unsigned short> gainMap);
	__declspec(dllexport) void setDefectCorrection(std::span<unsigned short> defectMap);
	__declspec(dllexport)void setHistogramEquailisation(bool enable);

    __declspec(dllexport) tl::expected<void, std::string> run(uint64_t waitValue, uint64_t signalValue);
};

}