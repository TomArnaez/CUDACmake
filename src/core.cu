#include <core.hpp>
#include <cudahelpers.hpp>

namespace SLCuda {

Core::Core(uint32_t width, uint32_t height, CudaStream cudaStream)
    : 
        width(width), 
        height(height), 
        deviceBuffer(width * height),
        stream(std::move(cudaStream)) {}

tl::expected<Core, cudaError_t> Core::create(uint32_t width, uint32_t height) {
    CudaStream stream = CudaStream::create().map_error([](cudaError_t err) {
        return tl::unexpected(err);
    }).value();

    return Core(width, height, stream);
}

void Core::addCorrection(std::shared_ptr<ICorrection> correction) {
    corrections[static_cast<size_t>(correction->type())] = { correction, true };
}

void Core::enableCorrection(CorrectionType type, bool enable) {
    corrections[static_cast<size_t>(type)].enabled = enable;
}

void Core::run() {
    for (auto &entry : corrections)
        if (entry.enabled && entry.correction)
            entry.correction->run(deviceBuffer);
}

CudaStream& Core::getStream() {
    return stream;
}

thrust::device_vector<uint16_t>& Core::getBuffer() {
    return deviceBuffer;
}

}