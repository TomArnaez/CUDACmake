#include <memory>
#include <cstdint>
#include <wtypes.h>
#include <span>
#include <string>

#include <cudahelpers.hpp>

namespace SLCuda {

struct ExternalCudaCreateInfo {
	uint32_t width;
	uint32_t height;
	HANDLE timelineSemaphoreHandle;
    HANDLE inputBufferMemoryHandle;
	HANDLE outputBufferMemoryHandle;
    size_t bufferMemorySize;
};

class Core;

class ExternalCuda {
    std::shared_ptr<Core> core;
    uint32_t width;
    uint32_t height;
    CudaExternalMemory inputExternalBuffer;
    CudaExternalMemory outputExternalBuffer;
    CudaExternalSemaphore timelineSemaphore;

    ExternalCuda(
        uint32_t width,
        uint32_t height,
        CudaExternalMemory inputExternalBuffer,
        CudaExternalMemory outputExternalBuffer,
        CudaExternalSemaphore timelineSemaphore
    );

public:
    // static tl::expected<ExternalCuda, std::string> create(ExternalCudaCreateInfo createInfo);

    // void setDarkCorrection(std::span<unsigned short> darkMap, unsigned short offset);
	// void setGainCorrection(std::span<unsigned short> gainMap);
	// void setDefectCorrection(std::span<unsigned short> defectMap);
	// void setHistogramEquailisation(bool enable);

    void run();
};

}