// TensorRT plugin: INT32 -> INT8 requant
// C++ 负责：插件生命周期、形状推理、类型校验、序列化、以及调用 CUDA kernel

#include <NvInfer.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace nvinfer1;

// CUDA 实现（在 custom_op.cu 中编译），这里仅声明供 C++ 调用
extern "C" void requant_int32_to_int8_cuda(
	const int32_t* input,
	int8_t* output,
	int64_t n,
	float scale1,
	float scale2,
	cudaStream_t stream
);

namespace {
constexpr const char* kPLUGIN_NAME = "RequantInt32ToInt8";
constexpr const char* kPLUGIN_VERSION = "1";

inline int64_t volume(const Dims& d) {
	int64_t v = 1;
	for (int i = 0; i < d.nbDims; ++i) v *= d.d[i];
	return v;
}
} // namespace

class RequantInt32ToInt8Plugin : public IPluginV2DynamicExt {
public:
	RequantInt32ToInt8Plugin(float scale1, float scale2)
		: mScale1(scale1), mScale2(scale2) {}

	RequantInt32ToInt8Plugin(const void* data, size_t length) {
		const char* d = reinterpret_cast<const char*>(data);
		std::memcpy(&mScale1, d, sizeof(float));
		d += sizeof(float);
		std::memcpy(&mScale2, d, sizeof(float));
	}

	const char* getPluginType() const noexcept override { return kPLUGIN_NAME; }
	const char* getPluginVersion() const noexcept override { return kPLUGIN_VERSION; }
	int getNbOutputs() const noexcept override { return 1; }

	DimsExprs getOutputDimensions(
		int outputIndex,
		const DimsExprs* inputs,
		int nbInputs,
		IExprBuilder& exprBuilder
	) noexcept override {
		return inputs[0];
	}

	int initialize() noexcept override { return 0; }
	void terminate() noexcept override {}

	size_t getWorkspaceSize(
		const PluginTensorDesc* inputs,
		int nbInputs,
		const PluginTensorDesc* outputs,
		int nbOutputs
	) const noexcept override {
		return 0;
	}

	// enqueue 在推理时被调用：把输入/输出指针和 scale 传给 CUDA kernel
	int enqueue(
		const PluginTensorDesc* inputDesc,
		const PluginTensorDesc* outputDesc,
		const void* const* inputs,
		void* const* outputs,
		void* workspace,
		cudaStream_t stream
	) noexcept override {
		const auto& in = inputDesc[0];
		int64_t n = volume(in.dims);
		requant_int32_to_int8_cuda(
			reinterpret_cast<const int32_t*>(inputs[0]),
			reinterpret_cast<int8_t*>(outputs[0]),
			n,
			mScale1,
			mScale2,
			stream
		);
		return 0;
	}

	size_t getSerializationSize() const noexcept override {
		return sizeof(float) * 2;
	}

	void serialize(void* buffer) const noexcept override {
		char* d = reinterpret_cast<char*>(buffer);
		std::memcpy(d, &mScale1, sizeof(float));
		d += sizeof(float);
		std::memcpy(d, &mScale2, sizeof(float));
	}

	// 声明输入/输出类型：INT32 -> INT8（线性布局）
	bool supportsFormatCombination(
		int pos,
		const PluginTensorDesc* inOut,
		int nbInputs,
		int nbOutputs
	) noexcept override {
		const PluginTensorDesc& desc = inOut[pos];
		if (pos == 0) {
			return desc.type == DataType::kINT32 && desc.format == TensorFormat::kLINEAR;
		}
		if (pos == 1) {
			return desc.type == DataType::kINT8 && desc.format == TensorFormat::kLINEAR;
		}
		return false;
	}

	// 输出类型固定为 INT8
	DataType getOutputDataType(
		int index,
		const DataType* inputTypes,
		int nbInputs
	) const noexcept override {
		return DataType::kINT8;
	}

	void configurePlugin(
		const DynamicPluginTensorDesc* inputs,
		int nbInputs,
		const DynamicPluginTensorDesc* outputs,
		int nbOutputs
	) noexcept override {}

	IPluginV2DynamicExt* clone() const noexcept override {
		auto* plugin = new RequantInt32ToInt8Plugin(mScale1, mScale2);
		plugin->setPluginNamespace(mNamespace.c_str());
		return plugin;
	}

	void destroy() noexcept override { delete this; }

	void setPluginNamespace(const char* pluginNamespace) noexcept override {
		mNamespace = pluginNamespace ? pluginNamespace : "";
	}

	const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
	float mScale1{1.0f};
	float mScale2{1.0f};
	std::string mNamespace;
};

class RequantInt32ToInt8PluginCreator : public IPluginCreator {
public:
	RequantInt32ToInt8PluginCreator() {
		mFields.emplace_back(PluginField{"scale1", nullptr, PluginFieldType::kFLOAT32, 1});
		mFields.emplace_back(PluginField{"scale2", nullptr, PluginFieldType::kFLOAT32, 1});
		mFC.nbFields = static_cast<int>(mFields.size());
		mFC.fields = mFields.data();
	}

	const char* getPluginName() const noexcept override { return kPLUGIN_NAME; }
	const char* getPluginVersion() const noexcept override { return kPLUGIN_VERSION; }
	const PluginFieldCollection* getFieldNames() noexcept override { return &mFC; }

	IPluginV2* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
		float scale1 = 1.0f;
		float scale2 = 1.0f;
		for (int i = 0; i < fc->nbFields; ++i) {
			const auto& f = fc->fields[i];
			if (!std::strcmp(f.name, "scale1")) {
				scale1 = *static_cast<const float*>(f.data);
			} else if (!std::strcmp(f.name, "scale2")) {
				scale2 = *static_cast<const float*>(f.data);
			}
		}
		auto* plugin = new RequantInt32ToInt8Plugin(scale1, scale2);
		plugin->setPluginNamespace(mNamespace.c_str());
		return plugin;
	}

	IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override {
		auto* plugin = new RequantInt32ToInt8Plugin(serialData, serialLength);
		plugin->setPluginNamespace(mNamespace.c_str());
		return plugin;
	}

	void setPluginNamespace(const char* pluginNamespace) noexcept override {
		mNamespace = pluginNamespace ? pluginNamespace : "";
	}

	const char* getPluginNamespace() const noexcept override { return mNamespace.c_str(); }

private:
	std::string mNamespace;
	PluginFieldCollection mFC{};
	std::vector<PluginField> mFields;
};

REGISTER_TENSORRT_PLUGIN(RequantInt32ToInt8PluginCreator);
