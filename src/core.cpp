#include <core.hpp>

namespace SLCuda {

Core::Core(uint32_t width, uint32_t height, CudaStream steam)
    : width(width), height(height), stream(stream) {}

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

void Core::run(thrust::device_vector<unsigned short>& input) {
    for (auto &entry : corrections)
        if (entry.enabled && entry.correction)
            entry.correction->run(input);
}

CudaStream& Core::getStream() {
    return stream;
}

}