#include <gtest/gtest.h>
#include <opencv2/imgcodecs.hpp>

#include <vulkanhelpers.hpp>
#include <externalcuda.hpp>

const std::string TEST_IMAGES_DIR = "C:\\dev\\data\\Test Images\\";
const std::string DARK_IMAGE_PATH = TEST_IMAGES_DIR + "AVG_Dark_2802_2400.tif";
const std::string GAIN_IMAGE_PATH = TEST_IMAGES_DIR + "AVG_Gain_2802_2400.tif";
const std::string DEFECT_IMAGE_PATH = TEST_IMAGES_DIR + "DefectMap.tif";
const std::string PCB_IMAGE_PATH = TEST_IMAGES_DIR + "AVG_PCB_2802_2400.tif";

TEST(VulkanAppTest, InitializeVulkan) {
    cv::Mat darkMap = cv::imread(DARK_IMAGE_PATH, cv::IMREAD_UNCHANGED);
    cv::Mat gainMap = cv::imread(GAIN_IMAGE_PATH, cv::IMREAD_UNCHANGED);
    cv::Mat defectMap = cv::imread(DEFECT_IMAGE_PATH, cv::IMREAD_UNCHANGED);
    cv::Mat PCBImage = cv::imread(PCB_IMAGE_PATH, cv::IMREAD_UNCHANGED);

    const uint32_t width = PCBImage.cols;
    const uint32_t height = PCBImage.rows;
    const uint32_t numPixels = PCBImage.rows * PCBImage.cols;
    const size_t imageSizeInBytes = numPixels * sizeof(unsigned short);

    std::span<unsigned short> inputSpan(reinterpret_cast<unsigned short*>(PCBImage.ptr()), numPixels);
    std::span<unsigned short> darkMapSpan(reinterpret_cast<unsigned short*>(darkMap.ptr()), numPixels);
    std::span<unsigned short> gainMapSpan(reinterpret_cast<unsigned short*>(gainMap.ptr()), numPixels);
    std::span<unsigned short> defectMapSpan(reinterpret_cast<unsigned short*>(defectMap.ptr()), numPixels);

    vk::Instance instance = initializeVulkan();

    vk::PhysicalDevice physicalDevice;
    vk::Device device = createDevice(instance, physicalDevice);

    vk::Queue queue = device.getQueue(0, 0);
    vk::CommandPoolCreateInfo poolInfo({}, 0);
    vk::CommandPool commandPool = device.createCommandPool(poolInfo);

    vk::Semaphore timelineSemaphore = createExportableTimelineSemaphore(device, getDefaultSemaphoreHandleType(), 0);
    HANDLE timelineSemaphoreHandle = getWin32HandleFromSemaphore(device, timelineSemaphore, getDefaultSemaphoreHandleType());

    auto [srcBuffer, srcMemory] = createBuffer(
        device,
        physicalDevice,
        imageSizeInBytes,
        vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    auto [finalBuffer, finalMemory] = createBuffer(
        device,
        physicalDevice,
        imageSizeInBytes,
        vk::BufferUsageFlagBits::eTransferDst,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
    );

    auto [inputBuffer, inputMemory] = createExternalBuffer(
        device,
        physicalDevice,
        imageSizeInBytes,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32
    );
    HANDLE inputMemoryHandle = getWin32HandleFromMemory(device, inputMemory, getDefaultMemHandleType());

    auto [outputBuffer, outputMemory] = createExternalBuffer(
        device,
        physicalDevice,
        imageSizeInBytes,
        vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eTransferSrc,
        vk::MemoryPropertyFlagBits::eDeviceLocal,
        vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32
    );
    HANDLE outputMemoryHandle = getWin32HandleFromMemory(device, outputMemory, getDefaultMemHandleType());

    SLCuda::ExternalCudaCreateInfo createInfo {
        width,
        height,
        timelineSemaphoreHandle,
        inputMemoryHandle,
        outputMemoryHandle,
        imageSizeInBytes
    };

    SLCuda::ExternalCuda cuda = SLCuda::ExternalCuda::create(createInfo)
        .map_error([](std::string err) {
            std::cout << "Got Error: " << err << std::endl;
            return -1;
        }).value();

    cuda.setDarkCorrection(darkMapSpan, 300);
    cuda.setGainCorrection(gainMapSpan);
    cuda.setDefectCorrection(defectMapSpan);
    cuda.setHistogramEquailisation(true);

    void* data = device.mapMemory(srcMemory, 0, imageSizeInBytes);
    memcpy(data, PCBImage.data, imageSizeInBytes);
    device.unmapMemory(srcMemory);

    vk::CommandBuffer copySrcToInputCmdBuffer = createCopyCommandBuffer(
        device,
        commandPool,
        srcBuffer,
        inputBuffer,
        imageSizeInBytes
    );

    vk::CommandBuffer copyOutputToFinalCmdBuffer = createCopyCommandBuffer(
        device,
        commandPool,
        outputBuffer,
        finalBuffer,
        imageSizeInBytes
    );

    auto start = std::chrono::high_resolution_clock::now();

    submitCommandBufferWithSignalOnly(
        device,
        queue,
        copySrcToInputCmdBuffer,
        timelineSemaphore,
        1
    );

    cuda.run(1, 2)
        .map_error([](std::string err) {
            std::cout << "Got err whilst running: " << err << std::endl;
        });

    submitCommandBufferWithTimelineSemaphore(
        device,
        commandPool,
        queue,
        copyOutputToFinalCmdBuffer,
        timelineSemaphore,
        2,
        3
    );

    waitForTimelineSemaphore(device, timelineSemaphore, 3);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;

    void* finalData = device.mapMemory(finalMemory, 0, imageSizeInBytes);
    cv::Mat outputImage(PCBImage.rows, PCBImage.cols, PCBImage.type(), finalData);
    cv::imwrite("finalimg.tiff", outputImage);
    device.unmapMemory(finalMemory);
}