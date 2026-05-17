#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include <algorithm>
#include "utils/file_io.h"

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


__global__ void pool1_kernel(const float* input, float* output, 
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


__global__ void conv2_kernel(const float* input, const float* weights, const float* biases, float* output, 
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


__global__ void fc2_kernel(const float* input, const float* weights, const float* biases, float* output, 
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
        auto h_input = load_binary("../data/processed/test_image_0.bin");

        auto w_conv1 = load_binary("../weights/conv1_weight.bin");
        auto b_conv1 = load_binary("../weights/conv1_bias.bin");
        auto w_conv2 = load_binary("../weights/conv2_weight.bin");
        auto b_conv2 = load_binary("../weights/conv2_bias.bin");
        auto w_fc1   = load_binary("../weights/fc1_weight.bin");
        auto b_fc1   = load_binary("../weights/fc1_bias.bin");
        auto w_fc2   = load_binary("../weights/fc2_weight.bin");
        auto b_fc2   = load_binary("../weights/fc2_bias.bin");
        auto w_fc3   = load_binary("../weights/fc3_weight.bin");
        auto b_fc3   = load_binary("../weights/fc3_bias.bin");
        
        float *d_wc1, *d_bc1, *d_wc2, *d_bc2, *d_wfc1, *d_bfc1, *d_wfc2, *d_bfc2, *d_wfc3, *d_bfc3;
        
        float *d_in, *d_conv1, *d_pool1, *d_conv2, *d_pool2, *d_fc1, *d_fc2, *d_out;

        cudaMalloc(&d_in,    3 * 32 * 32 * sizeof(float));
        cudaMalloc(&d_conv1, 6 * 28 * 28 * sizeof(float));
        cudaMalloc(&d_pool1, 6 * 14 * 14 * sizeof(float));
        cudaMalloc(&d_conv2, 16 * 10 * 10 * sizeof(float));
        cudaMalloc(&d_pool2, 16 * 5 * 5 * sizeof(float));
        cudaMalloc(&d_fc1,   120 * sizeof(float));
        cudaMalloc(&d_fc2,   84 * sizeof(float));
        cudaMalloc(&d_out,   10 * sizeof(float));

        cudaMalloc(&d_wc1, w_conv1.size() * sizeof(float)); cudaMemcpy(d_wc1, w_conv1.data(), w_conv1.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_bc1, b_conv1.size() * sizeof(float)); cudaMemcpy(d_bc1, b_conv1.data(), b_conv1.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_wc2, w_conv2.size() * sizeof(float)); cudaMemcpy(d_wc2, w_conv2.data(), w_conv2.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_bc2, b_conv2.size() * sizeof(float)); cudaMemcpy(d_bc2, b_conv2.data(), b_conv2.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_wfc1, w_fc1.size() * sizeof(float));  cudaMemcpy(d_wfc1, w_fc1.data(), w_fc1.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_bfc1, b_fc1.size() * sizeof(float));  cudaMemcpy(d_bfc1, b_fc1.data(), b_fc1.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_wfc2, w_fc2.size() * sizeof(float));  cudaMemcpy(d_wfc2, w_fc2.data(), w_fc2.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_bfc2, b_fc2.size() * sizeof(float));  cudaMemcpy(d_bfc2, b_fc2.data(), b_fc2.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_wfc3, w_fc3.size() * sizeof(float));  cudaMemcpy(d_wfc3, w_fc3.data(), w_fc3.size() * sizeof(float), cudaMemcpyHostToDevice);
        cudaMalloc(&d_bfc3, b_fc3.size() * sizeof(float));  cudaMemcpy(d_bfc3, b_fc3.data(), b_fc3.size() * sizeof(float), cudaMemcpyHostToDevice);

        cudaMemcpy(d_in, h_input.data(), h_input.size() * sizeof(float), cudaMemcpyHostToDevice);

        cout << "Executing LeNet-5 Forward Pass...\n";

        // Conv1
        dim3 b_c1((28 + 15) / 16, (28 + 15) / 16, 6);
        conv1_kernel<<<b_c1, dim3(16,16,1)>>>(d_in, d_wc1, d_bc1, d_conv1, 3, 32, 32, 6, 28, 28, 5);

        // Pool1
        dim3 b_p1((14 + 15) / 16, (14 + 15) / 16, 6);
        pool1_kernel<<<b_p1, dim3(16,16,1)>>>(d_conv1, d_pool1, 6, 28, 28, 14, 14, 2, 2);

        // Conv2
        dim3 b_c2((10 + 15) / 16, (10 + 15) / 16, 16);
        conv2_kernel<<<b_c2, dim3(16,16,1)>>>(d_pool1, d_wc2, d_bc2, d_conv2, 6, 14, 14, 16, 10, 10, 5);

        // Pool2
        dim3 b_p2((5 + 15) / 16, (5 + 15) / 16, 16);
        pool2_kernel<<<b_p2, dim3(16,16,1)>>>(d_conv2, d_pool2, 16, 10, 10, 5, 5, 2, 2);

        // FC1
        fc1_kernel<<<(120 + 127) / 128, 128>>>(d_pool2, d_wfc1, d_bfc1, d_fc1, 400, 120);

        // FC2
        fc2_kernel<<<(84 + 127) / 128, 128>>>(d_fc1, d_wfc2, d_bfc2, d_fc2, 120, 84);

        // FC3
        fc3_kernel<<<(10 + 31) / 32, 32>>>(d_fc2, d_wfc3, d_bfc3, d_out, 84, 10);

        cudaDeviceSynchronize();

        // Bring final 10 numbers back to CPU
        vector<float> final_logits(10);
        cudaMemcpy(final_logits.data(), d_out, 10 * sizeof(float), cudaMemcpyDeviceToHost);

        int best_class = 0;
        float max_score = final_logits[0];
        for(int i = 1; i < 10; ++i) {
            if(final_logits[i] > max_score) {
                max_score = final_logits[i];
                best_class = i;
            }
        }

        const char* classes[] = {"plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"};
        
        cout << "\n====================================\n";
        cout << "   INFERENCE COMPLETE\n";
        cout << "====================================\n";
        for(int i=0; i<10; ++i) {
            cout << "Class " << i << " (" << classes[i] << "): \t" << final_logits[i] << "\n";
        }
        cout << "------------------------------------\n";
        cout << ">> PREDICTION: " << classes[best_class] << " <<\n";
        cout << "====================================\n\n";
        
        cudaFree(d_in);
        cudaFree(d_conv1);
        cudaFree(d_pool1);
        cudaFree(d_conv2);
        cudaFree(d_pool2);
        cudaFree(d_fc1);
        cudaFree(d_fc2);
        cudaFree(d_out);

        cudaFree(d_wc1);
        cudaFree(d_wc2);
        cudaFree(d_wfc1);
        cudaFree(d_wfc2);
        cudaFree(d_wfc3);

        cudaFree(d_bc1);
        cudaFree(d_bc2);
        cudaFree(d_bfc1);
        cudaFree(d_bfc2);
        cudaFree(d_bfc3);

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}