#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../../include/utils/file_io.h"

using namespace std;

__global__ void fc3_kernel(const float* input, const float* weights, const float* biases, float* output, 
                           int InFeatures, int OutFeatures) {
    
    int out_idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (out_idx < OutFeatures) {
        float sum = biases[out_idx];

        for (int in_idx = 0; in_idx < InFeatures; ++in_idx) {
            int w_idx = out_idx * InFeatures + in_idx;
            sum += input[in_idx] * weights[w_idx];
        }

        output[out_idx] = sum;
    }
}

int main() {
    try {
        auto h_input = load_binary("../results/predictions/golden_fc2_relu.bin");
        auto h_weights = load_binary("../weights/fc3_weight.bin");
        auto h_biases = load_binary("../weights/fc3_bias.bin");
        auto golden_fc3 = load_binary("../results/predictions/golden_fc3_out.bin");

        int InFeatures = 84;
        int OutFeatures = 10;
        
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

        int threadsPerBlock = 32;
        int numBlocks = (OutFeatures + threadsPerBlock - 1) / threadsPerBlock;

        fc3_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weights, d_biases, d_output, 
                                                   InFeatures, OutFeatures);

        cudaDeviceSynchronize();

        vector<float> h_output(OutFeatures);
        cudaMemcpy(h_output.data(), d_output, bytes_output, cudaMemcpyDeviceToHost);

        cout << "Verifying CUDA FC3...\n";
        verify_tensors(h_output, golden_fc3, 1e-4);

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