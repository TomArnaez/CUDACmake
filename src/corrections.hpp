#pragma once

#include <span>
#include <thrust/device_vector.h>

namespace SLCuda {

enum class CorrectionType {
    DarkCorrection,
    GainCorrection,
    DefectCorrection,
    HistogramEq,
    Count
};

class ICorrection {
public:
    virtual void run(thrust::device_vector<unsigned short>& input) = 0;
	virtual CorrectionType type() const = 0;
};

};