#include "stdafx.h"

#define VK_CHECK(x) { \
	VkResult res = (x); \
	if (res != VK_SUCCESS) { \
		throw std::exception("failed: " #x); \
	} \
}

struct vec4 {
	float r, g, b, a;
};

constexpr size_t WORKGROUP_SIZE = 16;
constexpr size_t WIDTH = 256;
constexpr size_t HEIGHT = 256;
constexpr size_t BUFFER_SIZE = sizeof(vec4) * WIDTH * HEIGHT;

class AOBench {
public:
	AOBench() :
		instance(VK_NULL_HANDLE),
		device(VK_NULL_HANDLE),
		buffer(VK_NULL_HANDLE),
		memory(VK_NULL_HANDLE),
		descriptorPool(VK_NULL_HANDLE),
		descriptorSetLayout(VK_NULL_HANDLE),
		shaderModule(VK_NULL_HANDLE),
		pipelineLayout(VK_NULL_HANDLE),
		pipeline(VK_NULL_HANDLE),
		commandPool(VK_NULL_HANDLE),
		fence(VK_NULL_HANDLE) {}

	virtual ~AOBench() {
		if (device) {
			vkDeviceWaitIdle(device);
			if (fence) {
				vkDestroyFence(device, fence, nullptr);
			}
			if (commandPool) {
				vkDestroyCommandPool(device, commandPool, nullptr);
			}
			if (pipeline) {
				vkDestroyPipeline(device, pipeline, nullptr);
			}
			if (pipelineLayout) {
				vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
			}
			if (shaderModule) {
				vkDestroyShaderModule(device, shaderModule, nullptr);
			}
			if (descriptorSetLayout) {
				vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);
			}
			if (descriptorPool) {
				vkDestroyDescriptorPool(device, descriptorPool, nullptr);
			}
			if (memory) {
				vkFreeMemory(device, memory, nullptr);
			}
			if (buffer) {
				vkDestroyBuffer(device, buffer, nullptr);
			}
			vkDestroyDevice(device, nullptr);
		}

		if (instance) {
			vkDestroyInstance(instance, nullptr);
		}
	}

	void run() {
		createInstance();
		acquirePhysicalDevice();
		findQueueFamily();
		createDevice();
		createBuffer();
		createDescriptorSet();
		createShaderModule();
		createPipeline();
		createCommandBuffer();
		submitCommandBuffer();
		saveImage();
	}

private:
	void createInstance() {
		VkApplicationInfo appInfo = {VK_STRUCTURE_TYPE_APPLICATION_INFO};
		appInfo.apiVersion = VK_API_VERSION_1_0;
		appInfo.pApplicationName = "vkaobench";
		appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);

		VkInstanceCreateInfo createInfo = {VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
		createInfo.pApplicationInfo = &appInfo;
#ifdef _DEBUG
		createInfo.enabledLayerCount = 1;
		constexpr auto VALIDATION_LAYER_NAME = "VK_LAYER_LUNARG_standard_validation";
		createInfo.ppEnabledLayerNames = &VALIDATION_LAYER_NAME;
#endif

		VK_CHECK(vkCreateInstance(&createInfo, nullptr, &instance));
	}

	void acquirePhysicalDevice() {
		uint32_t numPhysicalDevices = 0;
		VK_CHECK(vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, nullptr));
		if (numPhysicalDevices == 0) {
			throw std::exception("physical device not found");
		}

		const auto physicalDevices = std::make_unique<VkPhysicalDevice[]>(numPhysicalDevices);
		VK_CHECK(vkEnumeratePhysicalDevices(instance, &numPhysicalDevices, physicalDevices.get()));

		physicalDevice = physicalDevices[0];
	}

	void findQueueFamily() {
		uint32_t numQueueFamilyProperties = 0;
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilyProperties, nullptr);

		const auto queueFamilyProperties = std::make_unique<VkQueueFamilyProperties[]>(numQueueFamilyProperties);
		vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &numQueueFamilyProperties, queueFamilyProperties.get());

		for (uint32_t i = 0; i < numQueueFamilyProperties; ++i) {
			if (queueFamilyProperties[i].queueCount > 0 && queueFamilyProperties[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
				queueFamilyIndex = i;
				return;
			}
		}

		throw std::exception("queue family with VK_QUEUE_COMPUTE_BIT not found");
	}

	void createDevice() {
		VkDeviceQueueCreateInfo queueCreateInfo = {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
		queueCreateInfo.queueFamilyIndex = queueFamilyIndex;
		queueCreateInfo.queueCount = 1;
		const float priority = 1.f;
		queueCreateInfo.pQueuePriorities = &priority;

		VkDeviceCreateInfo deviceCreateInfo = {VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
		deviceCreateInfo.queueCreateInfoCount = 1;
		deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

		VK_CHECK(vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device));

		vkGetDeviceQueue(device, queueFamilyIndex, 0, &queue);
	}

	uint32_t findMemoryTypeIndex(uint32_t memoryTypeBits) {
		VkPhysicalDeviceMemoryProperties memoryProperties;
		vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memoryProperties);

		for (uint32_t i = 0; i < memoryProperties.memoryTypeCount; ++i) {
			if ((1 << i) & memoryTypeBits) {
				const auto& flags = memoryProperties.memoryTypes[i].propertyFlags;
				if (flags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT &&
					flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) {

					return i;
				}
			}
		}

		throw std::exception("suitable memory type not found");
	}

	void createBuffer() {
		VkBufferCreateInfo createInfo = {VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
		createInfo.queueFamilyIndexCount = 1;
		createInfo.pQueueFamilyIndices = &queueFamilyIndex;
		createInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
		createInfo.size = BUFFER_SIZE;
		createInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

		VK_CHECK(vkCreateBuffer(device, &createInfo, nullptr, &buffer));

		VkMemoryRequirements memoryRequirements;
		vkGetBufferMemoryRequirements(device, buffer, &memoryRequirements);

		VkMemoryAllocateInfo memoryAllocateInfo = {VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
		memoryAllocateInfo.allocationSize = memoryRequirements.size;
		memoryAllocateInfo.memoryTypeIndex = findMemoryTypeIndex(memoryRequirements.memoryTypeBits);

		VK_CHECK(vkAllocateMemory(device, &memoryAllocateInfo, nullptr, &memory));
		VK_CHECK(vkBindBufferMemory(device, buffer, memory, 0));
	}

	void createDescriptorSet() {
        VkDescriptorSetLayoutBinding descSetLayoutBinding = {};
        descSetLayoutBinding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        descSetLayoutBinding.binding = 0;
        descSetLayoutBinding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        descSetLayoutBinding.descriptorCount = 1;

		VkDescriptorSetLayoutCreateInfo descSetLayoutCreateInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
		descSetLayoutCreateInfo.bindingCount = 1;
		descSetLayoutCreateInfo.pBindings = &descSetLayoutBinding;

		VK_CHECK(vkCreateDescriptorSetLayout(device, &descSetLayoutCreateInfo, nullptr, &descriptorSetLayout));

		VkDescriptorPoolSize descPoolSize;
		descPoolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		descPoolSize.descriptorCount = 1;

		VkDescriptorPoolCreateInfo descPoolCreateInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
		descPoolCreateInfo.poolSizeCount = 1;
		descPoolCreateInfo.pPoolSizes = &descPoolSize;
		descPoolCreateInfo.maxSets = 1;

		VK_CHECK(vkCreateDescriptorPool(device, &descPoolCreateInfo, nullptr, &descriptorPool));

		VkDescriptorSetAllocateInfo descSetAllocateInfo = {VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
		descSetAllocateInfo.descriptorPool = descriptorPool;
		descSetAllocateInfo.descriptorSetCount = 1;
		descSetAllocateInfo.pSetLayouts = &descriptorSetLayout;

		VK_CHECK(vkAllocateDescriptorSets(device, &descSetAllocateInfo, &descriptorSet));

		VkDescriptorBufferInfo descBufferInfo = {};
		descBufferInfo.buffer = buffer;
		descBufferInfo.range = BUFFER_SIZE;

		VkWriteDescriptorSet writeDescSet = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
		writeDescSet.descriptorCount = 1;
		writeDescSet.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		writeDescSet.dstSet = descriptorSet;
		writeDescSet.dstBinding = 0;
		writeDescSet.pBufferInfo = &descBufferInfo;

		vkUpdateDescriptorSets(device, 1, &writeDescSet, 0, nullptr);
	}

	void createShaderModule() {
		std::ifstream ifs("comp.spv", std::ios::binary);
		if (!ifs) {
			throw std::exception("could not open shader binary");
		}

		const std::streampos begin = ifs.tellg();
		ifs.seekg(0, std::ios::end);
		const std::streampos end = ifs.tellg();
		const auto size = static_cast<size_t>(end - begin);

		const auto buffer = std::make_unique<char[]>(size);
		ifs.seekg(0);
		ifs.read(buffer.get(), size);
		ifs.close();

		VkShaderModuleCreateInfo createInfo = {VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
		createInfo.pCode = reinterpret_cast<uint32_t*>(buffer.get());
		createInfo.codeSize = size;

		VK_CHECK(vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule));
	}

	void createPipeline() {
		constexpr struct SpecializationData {
			uint32_t workgroupSize = WORKGROUP_SIZE;
			uint32_t width = WIDTH;
			uint32_t height = HEIGHT;
		} specializationData;

		constexpr std::array<VkSpecializationMapEntry, 3> specializationMapEntries = {{
			{0, offsetof(SpecializationData, workgroupSize), sizeof(specializationData.workgroupSize)},
			{1, offsetof(SpecializationData, width), sizeof(specializationData.width)},
			{2, offsetof(SpecializationData, height), sizeof(specializationData.height)}
		}};

		VkSpecializationInfo specializationInfo = {};
		specializationInfo.mapEntryCount = specializationMapEntries.size();
		specializationInfo.pMapEntries = specializationMapEntries.data();
		specializationInfo.dataSize = sizeof(specializationData);
		specializationInfo.pData = &specializationData;

		VkPipelineShaderStageCreateInfo shaderStageCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
		shaderStageCreateInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
		shaderStageCreateInfo.module = shaderModule;
		shaderStageCreateInfo.pName = "main";
		shaderStageCreateInfo.pSpecializationInfo = &specializationInfo;

		VkPipelineLayoutCreateInfo layoutCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
		layoutCreateInfo.setLayoutCount = 1;
		layoutCreateInfo.pSetLayouts = &descriptorSetLayout;

		VK_CHECK(vkCreatePipelineLayout(device, &layoutCreateInfo, nullptr, &pipelineLayout));

		VkComputePipelineCreateInfo pipelineCreateInfo = {VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
		pipelineCreateInfo.stage = shaderStageCreateInfo;
		pipelineCreateInfo.layout = pipelineLayout;

		VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineCreateInfo, nullptr, &pipeline));
	}

	void createCommandBuffer() {
		VkCommandPoolCreateInfo cmdPoolCreateInfo = {VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
		cmdPoolCreateInfo.queueFamilyIndex = queueFamilyIndex;
		cmdPoolCreateInfo.flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT;

		VK_CHECK(vkCreateCommandPool(device, &cmdPoolCreateInfo, nullptr, &commandPool));

		VkCommandBufferAllocateInfo cmdBufferAllocateInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
		cmdBufferAllocateInfo.commandPool = commandPool;
		cmdBufferAllocateInfo.commandBufferCount = 1;
		cmdBufferAllocateInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;

		VK_CHECK(vkAllocateCommandBuffers(device, &cmdBufferAllocateInfo, &commandBuffer));

		VkCommandBufferBeginInfo cmdBufferBeginInfo = {VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
		cmdBufferBeginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

		VK_CHECK(vkBeginCommandBuffer(commandBuffer, &cmdBufferBeginInfo));
		vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
		vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
		vkCmdDispatch(commandBuffer,
			static_cast<uint32_t>(std::ceil(static_cast<float>(WIDTH) / WORKGROUP_SIZE)),
			static_cast<uint32_t>(std::ceil(static_cast<float>(HEIGHT) / WORKGROUP_SIZE)), 1);
		VK_CHECK(vkEndCommandBuffer(commandBuffer));
	}

	void submitCommandBuffer() {
		VkSubmitInfo submitInfo = {VK_STRUCTURE_TYPE_SUBMIT_INFO};
		submitInfo.commandBufferCount = 1;
		submitInfo.pCommandBuffers = &commandBuffer;

		VkFenceCreateInfo fenceCreateInfo = {VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
		VK_CHECK(vkCreateFence(device, &fenceCreateInfo, nullptr, &fence));

		VK_CHECK(vkQueueSubmit(queue, 1, &submitInfo, fence));
	}

	void saveImage() {
		const auto buffer = std::make_unique<char[]>(3 * WIDTH * HEIGHT);

		VK_CHECK(vkWaitForFences(device, 1, &fence, VK_TRUE, std::numeric_limits<uint64_t>::max()));

		vec4* pixels;
		VK_CHECK(vkMapMemory(device, memory, 0, BUFFER_SIZE, 0, reinterpret_cast<void**>(&pixels)));
		for (int i = 0; i < WIDTH * HEIGHT; ++i) {
			buffer[3 * i + 0] = static_cast<char>(255 * pixels[i].r);
			buffer[3 * i + 1] = static_cast<char>(255 * pixels[i].g);
			buffer[3 * i + 2] = static_cast<char>(255 * pixels[i].b);
		}
		vkUnmapMemory(device, memory);

		std::ofstream ofs("ao.ppm", std::ios::binary);
		ofs << "P6" << std::endl
			<< WIDTH << " " << HEIGHT << std::endl
			<< "255" << std::endl;
		ofs.write(buffer.get(), 3 * WIDTH * HEIGHT);
	}

	VkInstance instance;
	VkPhysicalDevice physicalDevice;
	uint32_t queueFamilyIndex;
	VkDevice device;
	VkQueue queue;
	VkBuffer buffer;
	VkDeviceMemory memory;
	VkDescriptorSetLayout descriptorSetLayout;
	VkDescriptorPool descriptorPool;
	VkDescriptorSet descriptorSet;
	VkShaderModule shaderModule;
	VkPipelineLayout pipelineLayout;
	VkPipeline pipeline;
	VkCommandPool commandPool;
	VkCommandBuffer commandBuffer;
	VkFence fence;
};

int main() {
	try {
		AOBench aobench;
		aobench.run();
	} catch (const std::exception& ex) {
		std::cerr << ex.what() << std::endl;
		return EXIT_FAILURE;
	}

    return EXIT_SUCCESS;
}