#include <tuple>

#define VULKAN_HPP_DISPATCH_LOADER_DYNAMIC 1
#define VULKAN_HPP_TYPESAFE_CONVERSION
#define VK_USE_PLATFORM_WIN32_KHR
#define NOMINMAX

#include <vulkan/vulkan.hpp>
#include <iostream>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

vk::Instance initializeVulkan() {
    VULKAN_HPP_DEFAULT_DISPATCHER.init();
    vk::DynamicLoader dl;
    VULKAN_HPP_DEFAULT_DISPATCHER.init(dl);
    PFN_vkGetInstanceProcAddr getInstanceProcAddr = dl.getProcAddress<PFN_vkGetInstanceProcAddr>("vkGetInstanceProcAddr");
    VULKAN_HPP_DEFAULT_DISPATCHER.init(getInstanceProcAddr);

    vk::ApplicationInfo appInfo("External Semaphore Example", 1, "No Engine", 1, VK_API_VERSION_1_2);
    std::vector<const char*> instanceExtensions = { };
    vk::InstanceCreateInfo instanceCreateInfo({}, &appInfo, 0, nullptr, instanceExtensions.size(), instanceExtensions.data());

    vk::Instance instance = vk::createInstance(instanceCreateInfo);
    VULKAN_HPP_DEFAULT_DISPATCHER.init(instance);
    return instance;
}

vk::Device createDevice(const vk::Instance& instance, vk::PhysicalDevice& physicalDevice) {
    std::vector<vk::PhysicalDevice> physicalDevices = instance.enumeratePhysicalDevices();
    physicalDevice = physicalDevices[1];

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
    return device;
}

uint32_t findMemoryType(const vk::PhysicalDevice& physDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = physDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

vk::Semaphore createExportableTimelineSemaphore(
    const vk::Device& device,
    vk::ExternalSemaphoreHandleTypeFlags handleType,
    uint64_t initialValue
    ) {
    vk::SemaphoreTypeCreateInfo timelineCreateInfo(
        vk::SemaphoreType::eTimeline, 
        initialValue
    );
    vk::ExportSemaphoreCreateInfo exportSemaphoreCreateInfo(
        handleType,
        &timelineCreateInfo
    );
    vk::SemaphoreCreateInfo semaphoreCreateInfo(
        vk::SemaphoreCreateFlags(),
        &exportSemaphoreCreateInfo
    );
    return device.createSemaphore(semaphoreCreateInfo);
}

HANDLE getWin32HandleFromSemaphore(
    const vk::Device& device,
    const vk::Semaphore& semaphore,
    vk::ExternalSemaphoreHandleTypeFlagBits handleType
    ) {
    vk::SemaphoreGetWin32HandleInfoKHR semaphoreGetWin32HandleInfo(semaphore, handleType);
    HANDLE win32Handle = device.getSemaphoreWin32HandleKHR(semaphoreGetWin32HandleInfo);
    return win32Handle;
}

HANDLE getWin32HandleFromMemory(const vk::Device& device, const vk::DeviceMemory& memory, vk::ExternalMemoryHandleTypeFlagBits handleType) {
    vk::MemoryGetWin32HandleInfoKHR memoryGetWin32HandleInfo(memory, handleType);
    return device.getMemoryWin32HandleKHR(memoryGetWin32HandleInfo);
}

vk::ExternalMemoryHandleTypeFlagBits getDefaultMemHandleType() {
    return vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32;
}

vk::ExternalSemaphoreHandleTypeFlagBits getDefaultSemaphoreHandleType() {
    return vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueWin32;
}

std::pair<vk::Buffer, vk::DeviceMemory> createBuffer(
    const vk::Device& device,
    const vk::PhysicalDevice& physDevice, 
    vk::DeviceSize deviceSize,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties
    ) {
    vk::BufferCreateInfo bufferInfo({}, deviceSize, usage, vk::SharingMode::eExclusive);
    vk::Buffer buffer = device.createBuffer(bufferInfo);

    vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);
    vk::MemoryAllocateInfo allocInfo(memRequirements.size, findMemoryType(physDevice, memRequirements.memoryTypeBits, properties));
    vk::DeviceMemory memory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(buffer, memory, 0);

    return std::make_pair(buffer, memory);
}

std::pair<vk::Buffer, vk::DeviceMemory> createExternalBuffer(
    const vk::Device& device,
    const vk::PhysicalDevice& physDevice, 
    vk::DeviceSize deviceSize,
    vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties,
    vk::ExternalMemoryHandleTypeFlagsKHR extMemHandleType
    ) {
    vk::ExternalMemoryBufferCreateInfo externalMemoryBufferInfo(extMemHandleType);
    vk::BufferCreateInfo bufferInfo({}, deviceSize, usage, vk::SharingMode::eExclusive);
    bufferInfo.pNext = &externalMemoryBufferInfo;

    vk::Buffer buffer = device.createBuffer(bufferInfo);

    vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);

    vk::ExportMemoryAllocateInfoKHR vulkanExportMemoryAllocateInfoKHR(extMemHandleType);

    vk::MemoryAllocateInfo allocInfo(memRequirements.size, findMemoryType(physDevice, memRequirements.memoryTypeBits, properties), &vulkanExportMemoryAllocateInfoKHR);
    vk::DeviceMemory memory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(buffer, memory, 0);

    return std::make_pair(buffer, memory);
}

vk::CommandBuffer createCopyCommandBuffer(
    const vk::Device& device,
    vk::CommandPool& commandPool,
    vk::Buffer srcBuffer,
    vk::Buffer dstBuffer,
    vk::DeviceSize size
    ) {
    vk::CommandBufferAllocateInfo allocInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1);
    vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(allocInfo).front();

    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    commandBuffer.begin(beginInfo);

    vk::BufferCopy copyRegion(0, 0, size);
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

    commandBuffer.end();
    return commandBuffer;
}

void submitCommandBufferWithFence(
    const vk::Device& device,
    vk::CommandPool& commandPool,
    vk::Queue& queue,
    vk::CommandBuffer& commandBuffer
    ) {
    vk::FenceCreateInfo fenceInfo;
    vk::Fence fence = device.createFence(fenceInfo);

    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);

    queue.submit(submitInfo, fence);

    device.waitForFences(fence, VK_TRUE, UINT64_MAX);

    device.destroyFence(fence);
    device.freeCommandBuffers(commandPool, commandBuffer);
}

void submitCommandBufferWithSignalOnly(
    const vk::Device& device,
    vk::Queue& queue,
    vk::CommandBuffer& commandBuffer,
    const vk::Semaphore& timelineSemaphore, 
    uint64_t signalValue
    ) {
    vk::TimelineSemaphoreSubmitInfo timelineSemaphoreInfo;
    timelineSemaphoreInfo.signalSemaphoreValueCount = 1;
    timelineSemaphoreInfo.pSignalSemaphoreValues = &signalValue;

    vk::SubmitInfo submitInfo;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &timelineSemaphore;
    submitInfo.pNext = &timelineSemaphoreInfo;

    queue.submit(submitInfo, nullptr);
}

void submitCommandBufferWithTimelineSemaphore(
    const vk::Device& device,
    vk::CommandPool& commandPool,
    vk::Queue& queue,
    vk::CommandBuffer& commandBuffer,
    const vk::Semaphore& timelineSemaphore, 
    uint64_t waitValue,
    uint64_t signalValue
    ) {
    vk::TimelineSemaphoreSubmitInfo timelineSemaphoreInfo;
    timelineSemaphoreInfo.waitSemaphoreValueCount = 1;
    timelineSemaphoreInfo.pWaitSemaphoreValues = &waitValue;
    timelineSemaphoreInfo.signalSemaphoreValueCount = 1;
    timelineSemaphoreInfo.pSignalSemaphoreValues = &signalValue;

    vk::SubmitInfo submitInfo;
    submitInfo.pWaitDstStageMask = nullptr;
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &timelineSemaphore;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &commandBuffer;
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &timelineSemaphore;
    submitInfo.pNext = &timelineSemaphoreInfo;

    queue.submit(submitInfo, nullptr);
}

void waitForTimelineSemaphore(
    const vk::Device& device,
    const vk::Semaphore& timelineSemaphore,
    uint64_t waitValue
    ) {
    vk::SemaphoreWaitInfo waitInfo;
    waitInfo.semaphoreCount = 1;
    waitInfo.pSemaphores = &timelineSemaphore;
    waitInfo.pValues = &waitValue;
    device.waitSemaphores(waitInfo, UINT64_MAX);
}