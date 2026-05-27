#include <vulkan/vulkan.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#define VK_CHECK(expr)                                                        \
    do {                                                                      \
        VkResult vk_check_status = (expr);                                    \
        if (vk_check_status != VK_SUCCESS) {                                  \
            throw std::runtime_error(std::string("Vulkan error at ") +        \
                                     #expr + ": " +                           \
                                     std::to_string(vk_check_status));        \
        }                                                                     \
    } while (false)

struct VulkanBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
};

std::vector<uint32_t> read_spirv_file(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) {
        throw std::runtime_error("failed to open SPIR-V file: " + path);
    }

    std::streamsize file_size = file.tellg();
    if (file_size <= 0 || file_size % 4 != 0) {
        throw std::runtime_error("invalid SPIR-V file size");
    }

    file.seekg(0, std::ios::beg);

    std::vector<uint32_t> words(static_cast<size_t>(file_size) / 4);
    if (!file.read(reinterpret_cast<char *>(words.data()), file_size)) {
        throw std::runtime_error("failed to read SPIR-V file");
    }

    return words;
}

uint32_t find_memory_type(VkPhysicalDevice physical_device,
                          uint32_t type_bits,
                          VkMemoryPropertyFlags required_flags) {
    VkPhysicalDeviceMemoryProperties memory_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);

    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; ++i) {
        bool type_matches = (type_bits & (1u << i)) != 0;
        bool flags_match =
            (memory_properties.memoryTypes[i].propertyFlags & required_flags) ==
            required_flags;

        if (type_matches && flags_match) {
            return i;
        }
    }

    throw std::runtime_error("failed to find suitable memory type");
}

uint32_t find_compute_queue_family(VkPhysicalDevice physical_device) {
    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                             &queue_family_count,
                                             nullptr);

    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_device,
                                             &queue_family_count,
                                             queue_families.data());

    for (uint32_t i = 0; i < queue_family_count; ++i) {
        if (queue_families[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            return i;
        }
    }

    throw std::runtime_error("failed to find compute queue family");
}

VkPhysicalDevice pick_physical_device(VkInstance instance) {
    uint32_t physical_device_count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance,
                                        &physical_device_count,
                                        nullptr));

    if (physical_device_count == 0) {
        throw std::runtime_error("no Vulkan physical devices found");
    }

    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance,
                                        &physical_device_count,
                                        physical_devices.data()));

    return physical_devices[0];
}

VulkanBuffer create_buffer(VkPhysicalDevice physical_device,
                           VkDevice device,
                           VkDeviceSize size,
                           VkBufferUsageFlags usage,
                           VkMemoryPropertyFlags memory_flags) {
    VulkanBuffer result;
    result.size = size;

    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(device, &buffer_info, nullptr, &result.buffer));

    VkMemoryRequirements memory_requirements;
    vkGetBufferMemoryRequirements(device,
                                  result.buffer,
                                  &memory_requirements);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = memory_requirements.size;
    alloc_info.memoryTypeIndex =
        find_memory_type(physical_device,
                         memory_requirements.memoryTypeBits,
                         memory_flags);

    VK_CHECK(vkAllocateMemory(device, &alloc_info, nullptr, &result.memory));
    VK_CHECK(vkBindBufferMemory(device, result.buffer, result.memory, 0));

    return result;
}

void upload_f32_buffer(VkDevice device,
                       const VulkanBuffer &buffer,
                       const std::vector<float> &data) {
    void *mapped = nullptr;
    VK_CHECK(vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &mapped));
    std::memcpy(mapped, data.data(), data.size() * sizeof(float));
    vkUnmapMemory(device, buffer.memory);
}

std::vector<float> download_f32_buffer(VkDevice device,
                                       const VulkanBuffer &buffer,
                                       size_t element_count) {
    std::vector<float> result(element_count);

    void *mapped = nullptr;
    VK_CHECK(vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &mapped));
    std::memcpy(result.data(), mapped, element_count * sizeof(float));
    vkUnmapMemory(device, buffer.memory);

    return result;
}

int main(int argc, char **argv) {
    try {
        if (argc < 2) {
            std::cerr << "usage: " << argv[0] << " shader.spv [n]\n";
            return 1;
        }

        std::string spirv_path = argv[1];
        uint32_t n = 64;

        if (argc >= 3) {
            n = static_cast<uint32_t>(std::stoul(argv[2]));
        }

        if (n == 0) {
            throw std::runtime_error("n must be greater than zero");
        }

        std::vector<uint32_t> spirv_code = read_spirv_file(spirv_path);

        VkApplicationInfo app_info{};
        app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        app_info.pApplicationName = "sp_tiles_vulkan_host";
        app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.pEngineName = "none";
        app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        app_info.apiVersion = VK_API_VERSION_1_0;

        VkInstanceCreateInfo instance_info{};
        instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        instance_info.pApplicationInfo = &app_info;

        VkInstance instance = VK_NULL_HANDLE;
        VK_CHECK(vkCreateInstance(&instance_info, nullptr, &instance));

        VkPhysicalDevice physical_device = pick_physical_device(instance);
        uint32_t queue_family_index = find_compute_queue_family(physical_device);

        float queue_priority = 1.0f;

        VkDeviceQueueCreateInfo queue_info{};
        queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        queue_info.queueFamilyIndex = queue_family_index;
        queue_info.queueCount = 1;
        queue_info.pQueuePriorities = &queue_priority;

        VkDeviceCreateInfo device_info{};
        device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        device_info.queueCreateInfoCount = 1;
        device_info.pQueueCreateInfos = &queue_info;

        VkDevice device = VK_NULL_HANDLE;
        VK_CHECK(vkCreateDevice(physical_device, &device_info, nullptr, &device));

        VkQueue queue = VK_NULL_HANDLE;
        vkGetDeviceQueue(device, queue_family_index, 0, &queue);

        VkDeviceSize buffer_size = sizeof(float) * n;
        VkMemoryPropertyFlags host_memory_flags =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
            VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

        VulkanBuffer buffer_a = create_buffer(
            physical_device,
            device,
            buffer_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            host_memory_flags);

        VulkanBuffer buffer_b = create_buffer(
            physical_device,
            device,
            buffer_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            host_memory_flags);

        VulkanBuffer buffer_c = create_buffer(
            physical_device,
            device,
            buffer_size,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            host_memory_flags);

        std::vector<float> a(n);
        std::vector<float> b(n);
        std::vector<float> c(n, 0.0f);

        for (uint32_t i = 0; i < n; ++i) {
            a[i] = static_cast<float>(i);
            b[i] = static_cast<float>(1000 + i);
        }

        upload_f32_buffer(device, buffer_a, a);
        upload_f32_buffer(device, buffer_b, b);
        upload_f32_buffer(device, buffer_c, c);

        VkShaderModuleCreateInfo shader_module_info{};
        shader_module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shader_module_info.codeSize = spirv_code.size() * sizeof(uint32_t);
        shader_module_info.pCode = spirv_code.data();

        VkShaderModule shader_module = VK_NULL_HANDLE;
        VK_CHECK(vkCreateShaderModule(device,
                                      &shader_module_info,
                                      nullptr,
                                      &shader_module));

        VkDescriptorSetLayoutBinding bindings[3]{};

        for (uint32_t i = 0; i < 3; ++i) {
            bindings[i].binding = i;
            bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
            bindings[i].pImmutableSamplers = nullptr;
        }

        VkDescriptorSetLayoutCreateInfo descriptor_set_layout_info{};
        descriptor_set_layout_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        descriptor_set_layout_info.bindingCount = 3;
        descriptor_set_layout_info.pBindings = bindings;

        VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
        VK_CHECK(vkCreateDescriptorSetLayout(device,
                                            &descriptor_set_layout_info,
                                            nullptr,
                                            &descriptor_set_layout));

        VkPipelineLayoutCreateInfo pipeline_layout_info{};
        pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipeline_layout_info.setLayoutCount = 1;
        pipeline_layout_info.pSetLayouts = &descriptor_set_layout;

        VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
        VK_CHECK(vkCreatePipelineLayout(device,
                                        &pipeline_layout_info,
                                        nullptr,
                                        &pipeline_layout));

        VkPipelineShaderStageCreateInfo shader_stage_info{};
        shader_stage_info.sType =
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shader_stage_info.module = shader_module;
        shader_stage_info.pName = "vector_add";

        VkComputePipelineCreateInfo pipeline_info{};
        pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipeline_info.stage = shader_stage_info;
        pipeline_info.layout = pipeline_layout;

        VkPipeline pipeline = VK_NULL_HANDLE;
        VK_CHECK(vkCreateComputePipelines(device,
                                          VK_NULL_HANDLE,
                                          1,
                                          &pipeline_info,
                                          nullptr,
                                          &pipeline));

        VkDescriptorPoolSize pool_size{};
        pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        pool_size.descriptorCount = 3;

        VkDescriptorPoolCreateInfo descriptor_pool_info{};
        descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        descriptor_pool_info.maxSets = 1;
        descriptor_pool_info.poolSizeCount = 1;
        descriptor_pool_info.pPoolSizes = &pool_size;

        VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
        VK_CHECK(vkCreateDescriptorPool(device,
                                        &descriptor_pool_info,
                                        nullptr,
                                        &descriptor_pool));

        VkDescriptorSetAllocateInfo descriptor_set_alloc_info{};
        descriptor_set_alloc_info.sType =
            VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        descriptor_set_alloc_info.descriptorPool = descriptor_pool;
        descriptor_set_alloc_info.descriptorSetCount = 1;
        descriptor_set_alloc_info.pSetLayouts = &descriptor_set_layout;

        VkDescriptorSet descriptor_set = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateDescriptorSets(device,
                                          &descriptor_set_alloc_info,
                                          &descriptor_set));

        VkDescriptorBufferInfo buffer_infos[3]{};
        buffer_infos[0].buffer = buffer_a.buffer;
        buffer_infos[0].offset = 0;
        buffer_infos[0].range = buffer_size;

        buffer_infos[1].buffer = buffer_b.buffer;
        buffer_infos[1].offset = 0;
        buffer_infos[1].range = buffer_size;

        buffer_infos[2].buffer = buffer_c.buffer;
        buffer_infos[2].offset = 0;
        buffer_infos[2].range = buffer_size;

        VkWriteDescriptorSet descriptor_writes[3]{};

        for (uint32_t i = 0; i < 3; ++i) {
            descriptor_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            descriptor_writes[i].dstSet = descriptor_set;
            descriptor_writes[i].dstBinding = i;
            descriptor_writes[i].dstArrayElement = 0;
            descriptor_writes[i].descriptorCount = 1;
            descriptor_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            descriptor_writes[i].pBufferInfo = &buffer_infos[i];
        }

        vkUpdateDescriptorSets(device, 3, descriptor_writes, 0, nullptr);

        VkCommandPoolCreateInfo command_pool_info{};
        command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        command_pool_info.queueFamilyIndex = queue_family_index;
        command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

        VkCommandPool command_pool = VK_NULL_HANDLE;
        VK_CHECK(vkCreateCommandPool(device,
                                     &command_pool_info,
                                     nullptr,
                                     &command_pool));

        VkCommandBufferAllocateInfo command_buffer_alloc_info{};
        command_buffer_alloc_info.sType =
            VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        command_buffer_alloc_info.commandPool = command_pool;
        command_buffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        command_buffer_alloc_info.commandBufferCount = 1;

        VkCommandBuffer command_buffer = VK_NULL_HANDLE;
        VK_CHECK(vkAllocateCommandBuffers(device,
                                          &command_buffer_alloc_info,
                                          &command_buffer));

        VkCommandBufferBeginInfo begin_info{};
        begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

        VK_CHECK(vkBeginCommandBuffer(command_buffer, &begin_info));

        vkCmdBindPipeline(command_buffer,
                          VK_PIPELINE_BIND_POINT_COMPUTE,
                          pipeline);

        vkCmdBindDescriptorSets(command_buffer,
                                VK_PIPELINE_BIND_POINT_COMPUTE,
                                pipeline_layout,
                                0,
                                1,
                                &descriptor_set,
                                0,
                                nullptr);

        uint32_t local_size_x = 16;
        uint32_t group_count_x = (n + local_size_x - 1) / local_size_x;

        vkCmdDispatch(command_buffer, group_count_x, 1, 1);

        VK_CHECK(vkEndCommandBuffer(command_buffer));

        VkSubmitInfo submit_info{};
        submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submit_info.commandBufferCount = 1;
        submit_info.pCommandBuffers = &command_buffer;

        VkFenceCreateInfo fence_info{};
        fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

        VkFence fence = VK_NULL_HANDLE;
        VK_CHECK(vkCreateFence(device, &fence_info, nullptr, &fence));

        VK_CHECK(vkQueueSubmit(queue, 1, &submit_info, fence));
        VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        std::vector<float> output = download_f32_buffer(device, buffer_c, n);

        for (uint32_t i = 0; i < n; ++i) {
            std::cout << i << ": " << a[i] << " + " << b[i]
                      << " = " << output[i] << "\n";
        }

        vkDestroyFence(device, fence, nullptr);
        vkDestroyCommandPool(device, command_pool, nullptr);
        vkDestroyDescriptorPool(device, descriptor_pool, nullptr);
        vkDestroyPipeline(device, pipeline, nullptr);
        vkDestroyPipelineLayout(device, pipeline_layout, nullptr);
        vkDestroyDescriptorSetLayout(device, descriptor_set_layout, nullptr);
        vkDestroyShaderModule(device, shader_module, nullptr);

        vkDestroyBuffer(device, buffer_a.buffer, nullptr);
        vkFreeMemory(device, buffer_a.memory, nullptr);

        vkDestroyBuffer(device, buffer_b.buffer, nullptr);
        vkFreeMemory(device, buffer_b.memory, nullptr);

        vkDestroyBuffer(device, buffer_c.buffer, nullptr);
        vkFreeMemory(device, buffer_c.memory, nullptr);

        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);

        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}