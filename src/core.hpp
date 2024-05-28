#pragma once

#include <memory>
#include <thrust/device_vector.h>
#include <vector>

#include <corrections.hpp>
#include <cudahelpers.hpp>

namespace SLCuda {

struct CorrectionEntry {
    std::shared_ptr<ICorrection> correction;
    bool enabled;
};

class Core {
    uint32_t width;
    uint32_t height;
    CudaStream stream;
    std::array<CorrectionEntry, static_cast<size_t>(CorrectionType::Count)> corrections;
    Core(uint32_t width, uint32_t height, CudaStream steam);
public:
    static tl::expected<Core, cudaError_t> create(uint32_t width, uint32_t height);
    
    void addCorrection(std::shared_ptr<ICorrection> correction);
    void enableCorrection(CorrectionType type, bool enable);
    void run(thrust::device_vector<unsigned short>& input);

    CudaStream& getStream();
};

}