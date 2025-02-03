#include <onnxruntime/core/providers/cpu/cpu_provider_factory.h>
#include <onnxruntime/core/providers/provider_api.h>
#include <onnxruntime/core/inference_session.h>
#include <onnxruntime/core/providers/tensor.h>
#include <iostream>
#include <vector>

int main() {
    // ONNX Runtime initialize करना
    onnxruntime::Env env(onnxruntime::LoggingLevel::WARNING, "ONNXModel");

    // Model लोड करना
    std::string model_path = "path_to_your_model.onnx"; // अपने ONNX मॉडल का पथ यहाँ डालें
    onnxruntime::SessionOptions session_options;
    onnxruntime::InferenceSession session(env);
    session.Load(model_path, session_options);

    // Input data बनाना
    std::vector<float> input_data = { /* अपनी इनपुट वैल्यूज डालें */ };
    std::vector<int64_t> input_dims = { /* Input डेटा के लिए डाइमेंशन्स डालें */ };

    // Input tensor बनाना
    onnxruntime::Tensor input_tensor(input_data, input_dims);

    // Model inference करना
    std::vector<onnxruntime::Tensor> output_tensors;
    session.Run({}, {input_tensor}, {}, output_tensors);

    // Output प्राप्त करना
    float* output_data = output_tensors[0].data<float>();

    // Output प्रदर्शित करना (जैसे, परिणाम दिखाना)
    std::cout << "Model Output: " << output_data[0] << std::endl;

    return 0;
}
