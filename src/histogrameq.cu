
#include <corrections.hpp>
#include <thrust/gather.h>
#include <cub/device/device_histogram.cuh>

namespace SLCuda {
    
constexpr uint32_t HISTOGRAM_EQ_RANGE = 256;

HistogramEquilisation::HistogramEquilisation(uint32_t width, uint32_t height, int numBins)
	: width(width), height(height), numBins(numBins), histogram(numBins), normedHistogram(numBins), LUT(numBins) {
	cub::DeviceHistogram::HistogramEven(
		tempStorage, tempStorageBytes,
		static_cast<unsigned short*>(nullptr), thrust::raw_pointer_cast(histogram.data()), numBins,
		0, numBins - 1, width * height
	);

	cudaMalloc(&tempStorage, tempStorageBytes);
}

void HistogramEquilisation::run(thrust::device_vector<unsigned short>& input) {
	uint32_t totalPixels = width * height;
	// TODO: Bug where the sum of the histogram doesn't equal number of pixels
	cub::DeviceHistogram::HistogramEven(
		tempStorage, tempStorageBytes,
		thrust::raw_pointer_cast(input.data()), thrust::raw_pointer_cast(histogram.data()), numBins,
		0, numBins - 1, totalPixels);

	thrust::inclusive_scan(histogram.begin(), histogram.end(), histogram.begin());

	thrust::transform(histogram.begin(), histogram.end(), normedHistogram.begin(),
		[totalPixels] __device__(unsigned int x) -> float {
		return static_cast<float>(x) / totalPixels;
	});

	thrust::transform(normedHistogram.begin(), normedHistogram.end(), LUT.begin(),
		[] __device__(float x) -> unsigned short {
		return static_cast<unsigned short>(HISTOGRAM_EQ_RANGE * x);
	});

	thrust::gather(input.begin(), input.end(), LUT.begin(), input.begin());
}

HistogramEquilisation::~HistogramEquilisation() {
	if (tempStorage)
		cudaFree(tempStorage);
}

}