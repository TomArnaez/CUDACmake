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

class DarkCorrection: public ICorrection {
private:
	thrust::device_vector<unsigned short> darkMap;
	unsigned short offset;
public:
    DarkCorrection(std::span<unsigned short> darkMap, unsigned short offset);
    DarkCorrection(thrust::device_vector<unsigned short> darkMap, unsigned short offset);
	void run(thrust::device_vector<unsigned short>& input) override;
	CorrectionType type() const override { return CorrectionType::DarkCorrection; }
};

class GainCorrection: public ICorrection {
private:
	thrust::device_vector<double> normedGainMap;
public:
    GainCorrection(std::span<unsigned short> gainMap);
	GainCorrection(thrust::device_vector<unsigned short> gainMap);
    void run(thrust::device_vector<unsigned short>& input);
    void normaliseGainMap(thrust::device_vector<unsigned short> gainMap);
	CorrectionType type() const override { return CorrectionType::GainCorrection; }
};

class DefectCorrection : public ICorrection {
private:
	uint32_t width;
	uint32_t height;
	thrust::device_vector<unsigned short> defectMap;
public:
    DefectCorrection(std::span<unsigned short> defectMap, uint32_t width, uint32_t height);
	DefectCorrection(thrust::device_vector<unsigned short> defectMap, uint32_t width, uint32_t height);
	void run(thrust::device_vector<unsigned short>& input) override;
	CorrectionType type() const override { return CorrectionType::DefectCorrection; }
};

class HistogramEquilisation : public ICorrection {
private:
	uint32_t width;
	uint32_t height;
	int numBins;
	void* tempStorage = nullptr;
	size_t tempStorageBytes = 0;
	thrust::device_vector<int> histogram;
	thrust::device_vector<float> normedHistogram;
	thrust::device_vector<unsigned short> LUT;
public:
	HistogramEquilisation(uint32_t width, uint32_t height, int numBins = 16384);
	~HistogramEquilisation();
	void run(thrust::device_vector<unsigned short>& input) override;
	CorrectionType type() const override { return CorrectionType::HistogramEq; }
};

};