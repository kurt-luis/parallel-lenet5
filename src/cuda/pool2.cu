#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include "../../include/utils/file_io.h"

using namespace std;

__global__ void pool2_kernel(const float* input, float* output, 
                             int C, int InH, int InW, int OutH, int OutW, int PoolK, int PoolS) {
    
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c  = blockIdx.z * blockDim.z + threadIdx.z;

    if (ow < OutW && oh < OutH && c < C) {
        float max_val = 0.0f; 

        for (int kh = 0; kh < PoolK; ++kh) {
            for (int kw = 0; kw < PoolK; ++kw) {
                int ih = oh * PoolS + kh;
                int iw = ow * PoolS + kw;
                
                int in_idx = c * (InH * InW) + ih * InW + iw;
                float val = input[in_idx];
                
                max_val = fmaxf(max_val, val); 
            }
        }

        int out_idx = c * (OutH * OutW) + oh * OutW + ow;
        output[out_idx] = max_val;
    }
}

int main() {
    try {
        auto h_input = load_binary("../results/predictions/golden_conv2.bin");
        auto golden_pool2 = load_binary("../results/predictions/golden_pool2.bin");

        int C = 16, InH = 10, InW = 10;
        int PoolK = 2, PoolS = 2;
        int OutH = InH / PoolS; // 5
        int OutW = InW / PoolS; // 5
        
        size_t bytes_input = h_input.size() * sizeof(float);
        size_t bytes_output = C * OutH * OutW * sizeof(float);

        float *d_input, *d_output;
        cudaMalloc(&d_input, bytes_input);
        cudaMalloc(&d_output, bytes_output);

        cudaMemcpy(d_input, h_input.data(), bytes_input, cudaMemcpyHostToDevice);

        dim3 threadsPerBlock(16, 16, 1);
        dim3 numBlocks((OutW + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (OutH + threadsPerBlock.y - 1) / threadsPerBlock.y,
                       C);

        pool2_kernel<<<numBlocks, threadsPerBlock>>>(d_input, d_output, 
                                                     C, InH, InW, OutH, OutW, PoolK, PoolS);

        cudaDeviceSynchronize();

        vector<float> h_output(C * OutH * OutW);
        cudaMemcpy(h_output.data(), d_output, bytes_output, cudaMemcpyDeviceToHost);

        cout << "Verifying CUDA Pool2...\n";
        verify_tensors(h_output, golden_pool2, 1e-4);

        cudaFree(d_input);
        cudaFree(d_output);

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}