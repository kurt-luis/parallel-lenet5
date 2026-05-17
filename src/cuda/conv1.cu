#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "../../include/utils/file_io.h"

using namespace std;

__global__ void conv1_kernel(const float* input, const float* weights, const float* biases, float* output, 
                             int InC, int InH, int InW, int OutC, int OutH, int OutW, int K) {
    
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z * blockDim.z + threadIdx.z;

    if (ow < OutW && oh < OutH && oc < OutC) {
        float sum = biases[oc];

        for (int ic = 0; ic < InC; ++ic) {
            for (int kh = 0; kh < K; ++kh) {
                for (int kw = 0; kw < K; ++kw) {
                    int in_idx = ic * (InH * InW) + (oh + kh) * InW + (ow + kw);
                    int w_idx = oc * (InC * K * K) + ic * (K * K) + kh * K + kw;
                    sum += input[in_idx] * weights[w_idx];
                }
            }
        }

        int out_idx = oc * (OutH * OutW) + oh * OutW + ow;
        output[out_idx] = sum;
    }
}

int main() {
    try {
        auto h_input = load_binary("../data/processed/test_image_0.bin");
        auto h_weights = load_binary("../weights/conv1_weight.bin");
        auto h_biases = load_binary("../weights/conv1_bias.bin");
        auto golden_conv1 = load_binary("../results/predictions/golden_conv1.bin");

        int InC = 3, InH = 32, InW = 32, OutC = 6, K = 5;
        int OutH = InH - K + 1;
        int OutW = InW - K + 1;
        
        size_t bytes_input = h_input.size() * sizeof(float);
        size_t bytes_weights = h_weights.size() * sizeof(float);
        size_t bytes_biases = h_biases.size() * sizeof(float);
        size_t bytes_output = OutC * OutH * OutW * sizeof(float);

        float *d_input, *d_weights, *d_biases, *d_output;
        cudaMalloc(&d_input, bytes_input);
        cudaMalloc(&d_weights, bytes_weights);
        cudaMalloc(&d_biases, bytes_biases);
        cudaMalloc(&d_output, bytes_output);

        cudaMemcpy(d_input, h_input.data(), bytes_input, cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, h_weights.data(), bytes_weights, cudaMemcpyHostToDevice);
        cudaMemcpy(d_biases, h_biases.data(), bytes_biases, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16, 1);
        dim3 numBlocks((OutW + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (OutH + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       OutC);

        conv1_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_weights, d_biases, d_output, 
                                                     InC, InH, InW, OutC, OutH, OutW, K);

        cudaDeviceSynchronize();

        vector<float> h_output(OutC * OutH * OutW);
        cudaMemcpy(h_output.data(), d_output, bytes_output, cudaMemcpyDeviceToHost);

        cout << "Verifying CUDA Conv1...\n";
        verify_tensors(h_output, golden_conv1, 1e-4);

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