#include <tuple>
#include <vulkan/vulkan.hpp>

VULKAN_HPP_DEFAULT_DISPATCH_LOADER_DYNAMIC_STORAGE

uint32_t findMemoryType(const vk::PhysicalDevice& physDevice, uint32_t typeFilter, vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties = physDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable memory type!");
}

vk::Semaphore createExportableTimelineSemaphore(const vk::Device& device, vk::ExternalSemaphoreHandleTypeFlags handleType, uint64_t initialValue) {
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

HANDLE getWin32HandleFromSemaphore(const vk::Device& device, const vk::Semaphore& semaphore, vk::ExternalSemaphoreHandleTypeFlagBits handleType) {
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

std::pair<vk::Buffer, vk::DeviceMemory> createBuffer(const vk::Device& device, const vk::PhysicalDevice& physDevice, 
    vk::DeviceSize deviceSize, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties) {

    vk::BufferCreateInfo bufferInfo({}, deviceSize, usage, vk::SharingMode::eExclusive);
    vk::Buffer buffer = device.createBuffer(bufferInfo);

    vk::MemoryRequirements memRequirements = device.getBufferMemoryRequirements(buffer);
    vk::MemoryAllocateInfo allocInfo(memRequirements.size, findMemoryType(physDevice, memRequirements.memoryTypeBits, properties));
    vk::DeviceMemory memory = device.allocateMemory(allocInfo);
    device.bindBufferMemory(buffer, memory, 0);

    return std::make_pair(buffer, memory);
}

std::pair<vk::Buffer, vk::DeviceMemory> createExternalBuffer(const vk::Device& device, const vk::PhysicalDevice& physDevice, 
    vk::DeviceSize deviceSize, vk::BufferUsageFlags usage,
    vk::MemoryPropertyFlags properties,
    vk::ExternalMemoryHandleTypeFlagsKHR extMemHandleType) {

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

void copyBuffer(const vk::Device& device, vk::Queue& queue, vk::CommandPool& commandPool, vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) {
    vk::CommandBufferAllocateInfo allocInfo(commandPool, vk::CommandBufferLevel::ePrimary, 1);
    vk::CommandBuffer commandBuffer = device.allocateCommandBuffers(allocInfo).front();

    vk::CommandBufferBeginInfo beginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
    commandBuffer.begin(beginInfo);

    vk::BufferCopy copyRegion(0, 0, size);
    commandBuffer.copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

    commandBuffer.end();

    vk::SubmitInfo submitInfo(0, nullptr, nullptr, 1, &commandBuffer);
    queue.submit(1, &submitInfo, nullptr);
    queue.waitIdle();

    device.freeCommandBuffers(commandPool, commandBuffer);
}