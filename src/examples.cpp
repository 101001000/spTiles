#include <vulkan/vulkan.h>

#include <cmath>
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

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t queue_family_index = 0;
};

struct BufferBinding {
    VulkanBuffer *buffer = nullptr;
    VkDeviceSize range = 0;
};

struct ComputeRun {
    std::string spirv_path;
    std::string entry_name;
    std::vector<BufferBinding> bindings;
    uint32_t dispatch_x = 1;
    uint32_t dispatch_y = 1;
    uint32_t dispatch_z = 1;
};

static uint32_t cdiv_u32(uint32_t lhs, uint32_t rhs) {
    return (lhs + rhs - 1) / rhs;
}

static std::vector<uint32_t> read_spirv_file(const std::string &path) {
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

static uint32_t find_memory_type(VkPhysicalDevice physical_device,
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

static uint32_t find_compute_queue_family(VkPhysicalDevice physical_device) {
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

static VkPhysicalDevice pick_physical_device(VkInstance instance) {
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

static VulkanContext create_vulkan_context() {
    VulkanContext context;

    VkApplicationInfo app_info{};
    app_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app_info.pApplicationName = "sp_tiles_vulkan_eval";
    app_info.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.pEngineName = "none";
    app_info.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    app_info.apiVersion = VK_API_VERSION_1_0;

    VkInstanceCreateInfo instance_info{};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app_info;

    VK_CHECK(vkCreateInstance(&instance_info, nullptr, &context.instance));

    context.physical_device = pick_physical_device(context.instance);
    context.queue_family_index = find_compute_queue_family(context.physical_device);

    float queue_priority = 1.0f;

    VkDeviceQueueCreateInfo queue_info{};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = context.queue_family_index;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &queue_priority;

    VkDeviceCreateInfo device_info{};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;

    VK_CHECK(vkCreateDevice(context.physical_device,
                            &device_info,
                            nullptr,
                            &context.device));

    vkGetDeviceQueue(context.device, context.queue_family_index, 0, &context.queue);

    return context;
}

static void destroy_vulkan_context(VulkanContext &context) {
    if (context.device != VK_NULL_HANDLE) {
        vkDestroyDevice(context.device, nullptr);
    }

    if (context.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(context.instance, nullptr);
    }
}

static VulkanBuffer create_buffer(VulkanContext &context,
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

    VK_CHECK(vkCreateBuffer(context.device, &buffer_info, nullptr, &result.buffer));

    VkMemoryRequirements memory_requirements;
    vkGetBufferMemoryRequirements(context.device,
                                  result.buffer,
                                  &memory_requirements);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = memory_requirements.size;
    alloc_info.memoryTypeIndex =
        find_memory_type(context.physical_device,
                         memory_requirements.memoryTypeBits,
                         memory_flags);

    VK_CHECK(vkAllocateMemory(context.device, &alloc_info, nullptr, &result.memory));
    VK_CHECK(vkBindBufferMemory(context.device, result.buffer, result.memory, 0));

    return result;
}

static void destroy_buffer(VkDevice device, VulkanBuffer &buffer) {
    if (buffer.buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device, buffer.buffer, nullptr);
    }

    if (buffer.memory != VK_NULL_HANDLE) {
        vkFreeMemory(device, buffer.memory, nullptr);
    }

    buffer = {};
}

static VulkanBuffer create_host_buffer(VulkanContext &context, VkDeviceSize size) {
    return create_buffer(context,
                         size,
                         VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
}

static void upload_f32_buffer(VkDevice device,
                              const VulkanBuffer &buffer,
                              const std::vector<float> &data) {
    void *mapped = nullptr;
    VK_CHECK(vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &mapped));
    std::memcpy(mapped, data.data(), data.size() * sizeof(float));
    vkUnmapMemory(device, buffer.memory);
}

static void upload_i32_buffer(VkDevice device,
                              const VulkanBuffer &buffer,
                              int32_t value) {
    void *mapped = nullptr;
    VK_CHECK(vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &mapped));
    std::memcpy(mapped, &value, sizeof(value));
    vkUnmapMemory(device, buffer.memory);
}

static std::vector<float> download_f32_buffer(VkDevice device,
                                              const VulkanBuffer &buffer,
                                              size_t element_count) {
    std::vector<float> result(element_count);

    void *mapped = nullptr;
    VK_CHECK(vkMapMemory(device, buffer.memory, 0, buffer.size, 0, &mapped));
    std::memcpy(result.data(), mapped, element_count * sizeof(float));
    vkUnmapMemory(device, buffer.memory);

    return result;
}

static void run_compute(VulkanContext &context, const ComputeRun &run) {
    std::vector<uint32_t> spirv_code = read_spirv_file(run.spirv_path);

    VkShaderModuleCreateInfo shader_module_info{};
    shader_module_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_module_info.codeSize = spirv_code.size() * sizeof(uint32_t);
    shader_module_info.pCode = spirv_code.data();

    VkShaderModule shader_module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(context.device,
                                  &shader_module_info,
                                  nullptr,
                                  &shader_module));

    std::vector<VkDescriptorSetLayoutBinding> layout_bindings(run.bindings.size());
    for (uint32_t i = 0; i < layout_bindings.size(); ++i) {
        layout_bindings[i].binding = i;
        layout_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layout_bindings[i].descriptorCount = 1;
        layout_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo descriptor_set_layout_info{};
    descriptor_set_layout_info.sType =
        VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_set_layout_info.bindingCount =
        static_cast<uint32_t>(layout_bindings.size());
    descriptor_set_layout_info.pBindings = layout_bindings.data();

    VkDescriptorSetLayout descriptor_set_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorSetLayout(context.device,
                                         &descriptor_set_layout_info,
                                         nullptr,
                                         &descriptor_set_layout));

    VkPipelineLayoutCreateInfo pipeline_layout_info{};
    pipeline_layout_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_info.setLayoutCount = 1;
    pipeline_layout_info.pSetLayouts = &descriptor_set_layout;

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    VK_CHECK(vkCreatePipelineLayout(context.device,
                                    &pipeline_layout_info,
                                    nullptr,
                                    &pipeline_layout));

    VkPipelineShaderStageCreateInfo shader_stage_info{};
    shader_stage_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shader_stage_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    shader_stage_info.module = shader_module;
    shader_stage_info.pName = run.entry_name.c_str();

    VkComputePipelineCreateInfo pipeline_info{};
    pipeline_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_info.stage = shader_stage_info;
    pipeline_info.layout = pipeline_layout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    VK_CHECK(vkCreateComputePipelines(context.device,
                                      VK_NULL_HANDLE,
                                      1,
                                      &pipeline_info,
                                      nullptr,
                                      &pipeline));

    VkDescriptorPoolSize pool_size{};
    pool_size.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    pool_size.descriptorCount = static_cast<uint32_t>(run.bindings.size());

    VkDescriptorPoolCreateInfo descriptor_pool_info{};
    descriptor_pool_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descriptor_pool_info.maxSets = 1;
    descriptor_pool_info.poolSizeCount = 1;
    descriptor_pool_info.pPoolSizes = &pool_size;

    VkDescriptorPool descriptor_pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(context.device,
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
    VK_CHECK(vkAllocateDescriptorSets(context.device,
                                      &descriptor_set_alloc_info,
                                      &descriptor_set));

    std::vector<VkDescriptorBufferInfo> buffer_infos(run.bindings.size());
    std::vector<VkWriteDescriptorSet> descriptor_writes(run.bindings.size());

    for (uint32_t i = 0; i < run.bindings.size(); ++i) {
        buffer_infos[i].buffer = run.bindings[i].buffer->buffer;
        buffer_infos[i].offset = 0;
        buffer_infos[i].range = run.bindings[i].range;

        descriptor_writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        descriptor_writes[i].dstSet = descriptor_set;
        descriptor_writes[i].dstBinding = i;
        descriptor_writes[i].descriptorCount = 1;
        descriptor_writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descriptor_writes[i].pBufferInfo = &buffer_infos[i];
    }

    vkUpdateDescriptorSets(context.device,
                           static_cast<uint32_t>(descriptor_writes.size()),
                           descriptor_writes.data(),
                           0,
                           nullptr);

    VkCommandPoolCreateInfo command_pool_info{};
    command_pool_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_info.queueFamilyIndex = context.queue_family_index;
    command_pool_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateCommandPool(context.device,
                                 &command_pool_info,
                                 nullptr,
                                 &command_pool));

    VkCommandBufferAllocateInfo command_buffer_alloc_info{};
    command_buffer_alloc_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_alloc_info.commandPool = command_pool;
    command_buffer_alloc_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_alloc_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(context.device,
                                      &command_buffer_alloc_info,
                                      &command_buffer));

    VkCommandBufferBeginInfo begin_info{};
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;

    VK_CHECK(vkBeginCommandBuffer(command_buffer, &begin_info));

    vkCmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(command_buffer,
                            VK_PIPELINE_BIND_POINT_COMPUTE,
                            pipeline_layout,
                            0,
                            1,
                            &descriptor_set,
                            0,
                            nullptr);

    vkCmdDispatch(command_buffer, run.dispatch_x, run.dispatch_y, run.dispatch_z);

    VK_CHECK(vkEndCommandBuffer(command_buffer));

    VkSubmitInfo submit_info{};
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;

    VkFenceCreateInfo fence_info{};
    fence_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    VkFence fence = VK_NULL_HANDLE;
    VK_CHECK(vkCreateFence(context.device, &fence_info, nullptr, &fence));

    VK_CHECK(vkQueueSubmit(context.queue, 1, &submit_info, fence));
    VK_CHECK(vkWaitForFences(context.device, 1, &fence, VK_TRUE, UINT64_MAX));

    vkDestroyFence(context.device, fence, nullptr);
    vkDestroyCommandPool(context.device, command_pool, nullptr);
    vkDestroyDescriptorPool(context.device, descriptor_pool, nullptr);
    vkDestroyPipeline(context.device, pipeline, nullptr);
    vkDestroyPipelineLayout(context.device, pipeline_layout, nullptr);
    vkDestroyDescriptorSetLayout(context.device, descriptor_set_layout, nullptr);
    vkDestroyShaderModule(context.device, shader_module, nullptr);
}

static bool almost_equal(float lhs, float rhs) {
    return std::fabs(lhs - rhs) <= 1.0e-3f;
}

constexpr uint32_t local_size_x = 1024;

static void run_vector_add_test(VulkanContext &context,
                                const std::string &spirv_path,
                                uint32_t size_x) {
    uint32_t element_count = size_x;
    VkDeviceSize f32_buffer_size = sizeof(float) * element_count;
    VkDeviceSize i32_buffer_size = sizeof(int32_t);

    VulkanBuffer buffer_a = create_host_buffer(context, f32_buffer_size);
    VulkanBuffer buffer_a_size = create_host_buffer(context, i32_buffer_size);
    VulkanBuffer buffer_a_stride = create_host_buffer(context, i32_buffer_size);
    VulkanBuffer buffer_b = create_host_buffer(context, f32_buffer_size);
    VulkanBuffer buffer_b_size = create_host_buffer(context, i32_buffer_size);
    VulkanBuffer buffer_b_stride = create_host_buffer(context, i32_buffer_size);
    VulkanBuffer buffer_c = create_host_buffer(context, f32_buffer_size);
    VulkanBuffer buffer_c_size = create_host_buffer(context, i32_buffer_size);
    VulkanBuffer buffer_c_stride = create_host_buffer(context, i32_buffer_size);

    std::vector<float> a(element_count);
    std::vector<float> b(element_count);
    std::vector<float> c(element_count, 0.0f);

    for (uint32_t i = 0; i < element_count; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(1000 + i);
    }

    upload_f32_buffer(context.device, buffer_a, a);
    upload_i32_buffer(context.device, buffer_a_size, static_cast<int32_t>(size_x));
    upload_i32_buffer(context.device, buffer_a_stride, 1);
    upload_f32_buffer(context.device, buffer_b, b);
    upload_i32_buffer(context.device, buffer_b_size, static_cast<int32_t>(size_x));
    upload_i32_buffer(context.device, buffer_b_stride, 1);
    upload_f32_buffer(context.device, buffer_c, c);
    upload_i32_buffer(context.device, buffer_c_size, static_cast<int32_t>(size_x));
    upload_i32_buffer(context.device, buffer_c_stride, 1);

    ComputeRun run;
    run.spirv_path = spirv_path;
    run.entry_name = "vector_add";
    run.bindings = {
        {&buffer_a, f32_buffer_size},
        {&buffer_a_size, i32_buffer_size},
        {&buffer_a_stride, i32_buffer_size},
        {&buffer_b, f32_buffer_size},
        {&buffer_b_size, i32_buffer_size},
        {&buffer_b_stride, i32_buffer_size},
        {&buffer_c, f32_buffer_size},
        {&buffer_c_size, i32_buffer_size},
        {&buffer_c_stride, i32_buffer_size},
    };
    run.dispatch_x = cdiv_u32(element_count, local_size_x);

    run_compute(context, run);

    std::vector<float> output =
        download_f32_buffer(context.device, buffer_c, element_count);

    for (uint32_t i = 0; i < element_count; ++i) {
        float expected = a[i] + b[i];
        if (!almost_equal(output[i], expected)) {
            throw std::runtime_error("vector_add mismatch at " + std::to_string(i));
        }
    }

    std::cout << "vector_add OK: " << element_count << " elements\n";

    destroy_buffer(context.device, buffer_a);
    destroy_buffer(context.device, buffer_a_size);
    destroy_buffer(context.device, buffer_a_stride);
    destroy_buffer(context.device, buffer_b);
    destroy_buffer(context.device, buffer_b_size);
    destroy_buffer(context.device, buffer_b_stride);
    destroy_buffer(context.device, buffer_c);
    destroy_buffer(context.device, buffer_c_size);
    destroy_buffer(context.device, buffer_c_stride);
}

static std::vector<float> reference_matmul(const std::vector<float> &a,
                                           const std::vector<float> &b,
                                           uint32_t m,
                                           uint32_t n,
                                           uint32_t k) {
    std::vector<float> c(m * n, 0.0f);

    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            float sum = 0.0f;
            for (uint32_t kk = 0; kk < k; ++kk) {
                sum += a[row * k + kk] * b[kk * n + col];
            }
            c[row * n + col] = sum;
        }
    }

    return c;
}

static void run_matmul_test(VulkanContext &context,
                            const std::string &spirv_path,
                            uint32_t m,
                            uint32_t n,
                            uint32_t k) {

    uint32_t a_count = m * k;
    uint32_t b_count = k * n;
    uint32_t c_count = m * n;

    VkDeviceSize a_buffer_size = sizeof(float) * a_count;
    VkDeviceSize b_buffer_size = sizeof(float) * b_count;
    VkDeviceSize c_buffer_size = sizeof(float) * c_count;
    VkDeviceSize i32_buffer_size = sizeof(int32_t);

    VulkanBuffer buffer_a = create_host_buffer(context, a_buffer_size);
    VulkanBuffer buffer_a_dim0 = create_host_buffer(context, i32_buffer_size);
    VulkanBuffer buffer_a_dim1 = create_host_buffer(context, i32_buffer_size);
    VulkanBuffer buffer_a_stride0 = create_host_buffer(context, i32_buffer_size);
    VulkanBuffer buffer_a_stride1 = create_host_buffer(context, i32_buffer_size);

    VulkanBuffer buffer_b = create_host_buffer(context, b_buffer_size);
    VulkanBuffer buffer_b_dim0 = create_host_buffer(context, i32_buffer_size);
    VulkanBuffer buffer_b_dim1 = create_host_buffer(context, i32_buffer_size);
    VulkanBuffer buffer_b_stride0 = create_host_buffer(context, i32_buffer_size);
    VulkanBuffer buffer_b_stride1 = create_host_buffer(context, i32_buffer_size);

    VulkanBuffer buffer_c = create_host_buffer(context, c_buffer_size);
    VulkanBuffer buffer_c_dim0 = create_host_buffer(context, i32_buffer_size);
    VulkanBuffer buffer_c_dim1 = create_host_buffer(context, i32_buffer_size);
    VulkanBuffer buffer_c_stride0 = create_host_buffer(context, i32_buffer_size);
    VulkanBuffer buffer_c_stride1 = create_host_buffer(context, i32_buffer_size);

    std::vector<float> a(a_count);
    std::vector<float> b(b_count);
    std::vector<float> c(c_count, 0.0f);

    for (uint32_t i = 0; i < a_count; ++i) {
        a[i] = static_cast<float>(static_cast<int32_t>(i % 17) - 8) * 0.25f;
    }

    for (uint32_t i = 0; i < b_count; ++i) {
        b[i] = static_cast<float>(static_cast<int32_t>(i % 13) - 6) * 0.5f;
    }

    upload_f32_buffer(context.device, buffer_a, a);
    upload_i32_buffer(context.device, buffer_a_dim0, static_cast<int32_t>(m));
    upload_i32_buffer(context.device, buffer_a_dim1, static_cast<int32_t>(k));
    upload_i32_buffer(context.device, buffer_a_stride0, static_cast<int32_t>(k));
    upload_i32_buffer(context.device, buffer_a_stride1, 1);

    upload_f32_buffer(context.device, buffer_b, b);
    upload_i32_buffer(context.device, buffer_b_dim0, static_cast<int32_t>(k));
    upload_i32_buffer(context.device, buffer_b_dim1, static_cast<int32_t>(n));
    upload_i32_buffer(context.device, buffer_b_stride0, static_cast<int32_t>(n));
    upload_i32_buffer(context.device, buffer_b_stride1, 1);

    upload_f32_buffer(context.device, buffer_c, c);
    upload_i32_buffer(context.device, buffer_c_dim0, static_cast<int32_t>(m));
    upload_i32_buffer(context.device, buffer_c_dim1, static_cast<int32_t>(n));
    upload_i32_buffer(context.device, buffer_c_stride0, static_cast<int32_t>(n));
    upload_i32_buffer(context.device, buffer_c_stride1, 1);


    ComputeRun run;
    run.spirv_path = spirv_path;
    run.entry_name = "matmul_fp32";
    run.bindings = {
        {&buffer_a, a_buffer_size},
        {&buffer_a_dim0, i32_buffer_size},
        {&buffer_a_dim1, i32_buffer_size},
        {&buffer_a_stride0, i32_buffer_size},
        {&buffer_a_stride1, i32_buffer_size},

        {&buffer_b, b_buffer_size},
        {&buffer_b_dim0, i32_buffer_size},
        {&buffer_b_dim1, i32_buffer_size},
        {&buffer_b_stride0, i32_buffer_size},
        {&buffer_b_stride1, i32_buffer_size},

        {&buffer_c, c_buffer_size},
        {&buffer_c_dim0, i32_buffer_size},
        {&buffer_c_dim1, i32_buffer_size},
        {&buffer_c_stride0, i32_buffer_size},
        {&buffer_c_stride1, i32_buffer_size},
    };

    auto ceil_div = [](uint32_t x, uint32_t y) {
        return (x + y - 1) / y;
    };

    constexpr uint32_t tile_m = 32;
    constexpr uint32_t tile_n = 32;
    constexpr uint32_t tile_size = tile_m * tile_n;

    uint32_t output_tile_count = ceil_div(m, tile_m) * ceil_div(n, tile_n);
    uint32_t invocation_count = output_tile_count * tile_size;

    run.dispatch_x = ceil_div(invocation_count, local_size_x);
    run.dispatch_y = 1;
    run.dispatch_z = 1;

    run_compute(context, run);

    std::vector<float> output =
        download_f32_buffer(context.device, buffer_c, c_count);

    std::vector<float> expected = reference_matmul(a, b, m, n, k);

    for (uint32_t i = 0; i < c_count; ++i) {
        if (!almost_equal(output[i], expected[i])) {
            throw std::runtime_error("matmul mismatch at " + std::to_string(i) +
                                     ": got " + std::to_string(output[i]) +
                                     ", expected " + std::to_string(expected[i]));
        }
    }

    std::cout << "matmul OK: " << m << "x" << k << " @ "
              << k << "x" << n << "\n";

    destroy_buffer(context.device, buffer_a);
    destroy_buffer(context.device, buffer_a_dim0);
    destroy_buffer(context.device, buffer_a_dim1);
    destroy_buffer(context.device, buffer_a_stride0);
    destroy_buffer(context.device, buffer_a_stride1);

    destroy_buffer(context.device, buffer_b);
    destroy_buffer(context.device, buffer_b_dim0);
    destroy_buffer(context.device, buffer_b_dim1);
    destroy_buffer(context.device, buffer_b_stride0);
    destroy_buffer(context.device, buffer_b_stride1);

    destroy_buffer(context.device, buffer_c);
    destroy_buffer(context.device, buffer_c_dim0);
    destroy_buffer(context.device, buffer_c_dim1);
    destroy_buffer(context.device, buffer_c_stride0);
    destroy_buffer(context.device, buffer_c_stride1);
}

int main(int argc, char **argv) {
    try {
        if (argc < 3) {
            std::cerr << "usage:\n"
                      << "  " << argv[0] << " vector_add shader.spv [size_x]\n"
                      << "  " << argv[0] << " matmul shader.spv [m] [n] [k]\n";
            return 1;
        }

        std::string test_name = argv[1];
        std::string spirv_path = argv[2];

        VulkanContext context = create_vulkan_context();

        if (test_name == "vector_add") {
            uint32_t size_x = argc >= 4
                                  ? static_cast<uint32_t>(std::stoul(argv[3]))
                                  : 1024;
            run_vector_add_test(context, spirv_path, size_x);
        } else if (test_name == "matmul") {
            uint32_t m = argc >= 4 ? static_cast<uint32_t>(std::stoul(argv[3])) : 64;
            uint32_t n = argc >= 5 ? static_cast<uint32_t>(std::stoul(argv[4])) : 64;
            uint32_t k = argc >= 6 ? static_cast<uint32_t>(std::stoul(argv[5])) : 64;
            run_matmul_test(context, spirv_path, m, n, k);
        } else {
            throw std::runtime_error("unknown test: " + test_name);
        }

        destroy_vulkan_context(context);
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << "\n";
        return 1;
    }
}