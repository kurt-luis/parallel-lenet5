#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../../include/utils/file_io.h"

using namespace std;


__global__ void fc1_kernel(const float* input, const float* weights, const float* biases, float* output, 
                           int InFeatures, int OutFeatures) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx < OutFeatures) {
        float sum = biases[out_idx];

        for (int in_idx = 0; in_idx < InFeatures; ++in_idx) {
            int w_idx = out_idx * InFeatures + in_idx;
            sum += input[in_idx] * weights[w_idx];
        }

        output[out_idx] = fmaxf(0.0f, sum);
    }
}

int main() {
    try {
        auto h_input = load_binary("../results/predictions/golden_pool2.bin");
        auto h_weights = load_binary("../weights/fc1_weight.bin");
        auto h_biases = load_binary("../weights/fc1_bias.bin");
        auto golden_fc1 = load_binary("../results/predictions/golden_fc1_relu.bin");

        int InFeatures = 16 * 5 * 5;
        int OutFeatures = 120;
        
        size_t bytes_input = h_input.size() * sizeof(float);
        size_t bytes_weights = h_weights.size() * sizeof(float);
        size_t bytes_biases = h_biases.size() * sizeof(float);
        size_t bytes_output = OutFeatures * sizeof(float);

        float *d_input, *d_weights, *d_biases, *d_output;
        cudaMalloc(&d_input, bytes_input);
        cudaMalloc(&d_weights, bytes_weights);
        cudaMalloc(&d_biases, bytes_biases);
        cudaMalloc(&d_output, bytes_output);

        cudaMemcpy(d_input, h_input.data(), bytes_input, cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, h_weights.data(), bytes_weights, cudaMemcpyHostToDevice);
        cudaMemcpy(d_biases, h_biases.data(), bytes_biases, cudaMemcpyHostToDevice);

        int threadsPerBlock = 128;
        int numBlocks = (OutFeatures + threadsPerBlock - 1) / threadsPerBlock;

        fc1_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weights, d_biases, d_output, 
                                                   InFeatures, OutFeatures);

        cudaDeviceSynchronize();

        vector<float> h_output(OutFeatures);
        cudaMemcpy(h_output.data(), d_output, bytes_output, cudaMemcpyDeviceToHost);

        cout << "Verifying CUDA FC1...\n";
        verify_tensors(h_output, golden_fc1, 1e-4);

        cudaFree(d_input);
        cudaFree(d_weights);
        cudaFree(d_biases);
        cudaFree(d_output);

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}