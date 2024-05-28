#include <gtest/gtest.h>
#include <externalcuda.hpp>

#define VK_USE_PLATFORM_WIN32_KHR
#define NOMINMAX
#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1

#include <vulkanhelpers.hpp>

const std::string TEST_IMAGES_DIR = "C:\\dev\\data\\Test Images\\";
const std::string DARK_IMAGE_PATH = TEST_IMAGES_DIR + "AVG_Dark_2802_2400.tif";
const std::string GAIN_IMAGE_PATH = TEST_IMAGES_DIR + "AVG_Gain_2802_2400.tif";
const std::string DEFECT_IMAGE_PATH = TEST_IMAGES_DIR + "DefectMap.tif";
const std::string PCB_IMAGE_PATH = TEST_IMAGES_DIR + "AVG_PCB_2802_2400.tif";

TEST(VulkanAppTest, InitializeVulkan) {

    VULKAN_HPP_DEFAULT_DISPATCHER.init();
    vk::DynamicLoader dl;
    VULKAN_HPP_DEFAULT_DISPATCHER.init(dl);
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(getInstanceProcAddr);

    // Create a Vulkan instance with the required extensions
    vk::ApplicationInfo appInfo("External Semaphore Example", 1, "No Engine", 1, VK_API_VERSION_1_2);

    std::vector<const char*> instanceExtensions = { };
    vk::InstanceCreateInfo instanceCreateInfo({}, &appInfo, 0, nullptr, instanceExtensions.size(), instanceExtensions.data());
    vk::Instance instance = vk::createInstance(instanceCreateInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);

    std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
    vk::PhysicalDevice physicalDevice = physicalDevices[0];

    vk::PhysicalDeviceProperties deviceProperties = physicalDevice.getProperties();
    std::cout << "Device Name: " << deviceProperties.deviceName << std::endl;

    std::vector<const char*> deviceExtensions = {
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME,
        VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME,
    };

    float queuePriority = 1.0f;
    vk::DeviceQueueCreateInfo queueCreateInfo({}, 0, 1, &queuePriority);
    vk::DeviceCreateInfo deviceCreateInfo({}, 1, &queueCreateInfo, 0, nullptr, deviceExtensions.size(), deviceExtensions.data());
    vk::Device device = physicalDevice.createDevice(deviceCreateInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(device);

    vk::Queue queue = device.getQueue(0, 0);
    vk::CommandPoolCreateInfo poolInfo({}, 0);
    vk::CommandPool commandPool = device.createCommandPool(poolInfo);

    vk::Semaphore timelineSemaphore = createExportableTimelineSemaphore(device, getDefaultSemaphoreHandleType(), 0);
}