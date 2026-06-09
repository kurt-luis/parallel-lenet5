#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <cuda_runtime.h>

using namespace std;

__constant__ float c_wc1[450];    // conv1: 6 * 3 * 5 * 5
__constant__ float c_bc1[6];
__constant__ float c_wc2[2400];   // conv2: 16 * 6 * 5 * 5
__constant__ float c_bc2[16];

__constant__ float c_wfc2[10080]; // fc2: 84 * 120
__constant__ float c_bfc2[84];
__constant__ float c_wfc3[840];   // fc3: 10 * 84
__constant__ float c_bfc3[10];


void readBin(const string& filename, vector<float>& data) {
    ifstream file(filename, ios::binary);
    if (!file) {
        cerr << "Error: Failed to open " << filename << endl;
        exit(1);
    }
    file.seekg(0, ios::end);
    size_t size = file.tellg();
    file.seekg(0, ios::beg);
    data.resize(size / sizeof(float));
    file.read(reinterpret_cast<char*>(data.data()), size);
    file.close();
}

// conv1 implementation: shared memory tiling
#define TILE_W 16
#define K_SIZE 5
#define IN_TILE_W (TILE_W + K_SIZE - 1) 

__global__ void conv1_shared(const float* input, float* output, 
                             int InH, int InW, int OutH, int OutW) {
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int out_c = blockIdx.z; 
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    int out_x = bx * TILE_W + tx;
    int out_y = by * TILE_W + ty;
    
    __shared__ float s_in[3][IN_TILE_W][IN_TILE_W];
    
    int tid = ty * TILE_W + tx; 
    
    for(int c = 0; c < 3; c++) {
        for(int i = tid; i < IN_TILE_W * IN_TILE_W; i += (TILE_W * TILE_W)) {
            int s_y = i / IN_TILE_W;
            int s_x = i % IN_TILE_W;
            
            int in_y = by * TILE_W + s_y;
            int in_x = bx * TILE_W + s_x;
            
            if(in_y < InH && in_x < InW) {
                s_in[c][s_y][s_x] = input[c * (InH * InW) + in_y * InW + in_x];
            } else {
                s_in[c][s_y][s_x] = 0.0f;
            }
        }
    }
    
    __syncthreads(); 
    
    if (out_x < OutW && out_y < OutH) {
        float sum = c_bc1[out_c]; 
        
        for(int c = 0; c < 3; c++) {
            for(int ky = 0; ky < K_SIZE; ky++) {
                for(int kx = 0; kx < K_SIZE; kx++) {
                    int w_idx = out_c * (3 * K_SIZE * K_SIZE) + c * (K_SIZE * K_SIZE) + ky * K_SIZE + kx;
                    sum += s_in[c][ty + ky][tx + kx] * c_wc1[w_idx];
                }
            }
        }
        output[out_c * (OutH * OutW) + out_y * OutW + out_x] = sum;
    }
}

// pool
__global__ void pool_kernel(const float* input, float* output, 
                            int C, int InH, int InW, int OutH, int OutW, int K, int S) {
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c  = blockIdx.z * blockDim.z + threadIdx.z;

    if (ow < OutW && oh < OutH && c < C) {
        float max_val = 0.0f;
        for (int kh = 0; kh < K; ++kh) {
            for (int kw = 0; kw < K; ++kw) {
                int in_idx = c * (InH * InW) + (oh * S + kh) * InW + (ow * S + kw);
                max_val = fmaxf(max_val, input[in_idx]);
            }
        }
        output[c * (OutH * OutW) + oh * OutW + ow] = max_val;
    }
}

// conv2: shared memory
__global__ void conv2_shared(const float* input, float* output, 
                             int InC, int InH, int InW, int OutC, int OutH, int OutW, int K) {
    int out_c = blockIdx.z;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    __shared__ float s_in[6][14][14];
    
    int tid = ty * blockDim.x + tx; 
    int total_pixels = InC * InH * InW;
    
    for (int i = tid; i < total_pixels; i += (blockDim.x * blockDim.y)) {
        int c = i / (InH * InW);
        int rem = i % (InH * InW);
        int h = rem / InW;
        int w = rem % InW;
        
        s_in[c][h][w] = input[c * (InH * InW) + h * InW + w];
    }
    
    __syncthreads();
    
    if (tx < OutW && ty < OutH) { 
        float sum = c_bc2[out_c];
        
        for (int c = 0; c < InC; c++) {
            for (int ky = 0; ky < K; ky++) {
                for (int kx = 0; kx < K; kx++) {
                    int w_idx = out_c * (InC * K * K) + c * (K * K) + ky * K + kx;
                    sum += s_in[c][ty + ky][tx + kx] * c_wc2[w_idx];
                }
            }
        }
        output[out_c * (OutH * OutW) + ty * OutW + tx] = sum;
    }
}

// fully connected layers
__global__ void fc1_shared(const float* input, const float* weights, const float* biases, float* output, int InFeatures) {
    int out_idx = blockIdx.x; 
    int tid = threadIdx.x;
    extern __shared__ float sdata[]; 

    float sum = 0.0f;
    for (int in_idx = tid; in_idx < InFeatures; in_idx += blockDim.x) {
        sum += input[in_idx] * weights[out_idx * InFeatures + in_idx];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[out_idx] = fmaxf(0.0f, sdata[0] + biases[out_idx]); 
}

__global__ void fc2_shared(const float* input, float* output, int InFeatures) {
    int out_idx = blockIdx.x; 
    int tid = threadIdx.x;
    extern __shared__ float sdata[]; 

    float sum = 0.0f;
    for (int in_idx = tid; in_idx < InFeatures; in_idx += blockDim.x) {
        sum += input[in_idx] * c_wfc2[out_idx * InFeatures + in_idx]; 
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[out_idx] = fmaxf(0.0f, sdata[0] + c_bfc2[out_idx]); 
}

__global__ void fc3_shared(const float* input, float* output, int InFeatures) {
    int out_idx = blockIdx.x; 
    int tid = threadIdx.x;
    extern __shared__ float sdata[]; 

    float sum = 0.0f;
    for (int in_idx = tid; in_idx < InFeatures; in_idx += blockDim.x) {
        sum += input[in_idx] * c_wfc3[out_idx * InFeatures + in_idx]; 
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[out_idx] = sdata[0] + c_bfc3[out_idx]; 
}


int main() {
    cout << "Loading weights and data...\n";
    vector<float> h_in, w_c1, b_c1, w_c2, b_c2, w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3;
    
    readBin("../data/processed/test_image_0.bin", h_in);
    readBin("../weights/conv1_weight.bin", w_c1); readBin("../weights/conv1_bias.bin", b_c1);
    readBin("../weights/conv2_weight.bin", w_c2); readBin("../weights/conv2_bias.bin", b_c2);
    readBin("../weights/fc1_weight.bin", w_fc1);  readBin("../weights/fc1_bias.bin", b_fc1);
    readBin("../weights/fc2_weight.bin", w_fc2);  readBin("../weights/fc2_bias.bin", b_fc2);
    readBin("../weights/fc3_weight.bin", w_fc3);  readBin("../weights/fc3_bias.bin", b_fc3);

    cudaMemcpyToSymbol(c_wc1, w_c1.data(), w_c1.size() * sizeof(float));
    cudaMemcpyToSymbol(c_bc1, b_c1.data(), b_c1.size() * sizeof(float));
    cudaMemcpyToSymbol(c_wc2, w_c2.data(), w_c2.size() * sizeof(float));
    cudaMemcpyToSymbol(c_bc2, b_c2.data(), b_c2.size() * sizeof(float));
    cudaMemcpyToSymbol(c_wfc2, w_fc2.data(), w_fc2.size() * sizeof(float));
    cudaMemcpyToSymbol(c_bfc2, b_fc2.data(), b_fc2.size() * sizeof(float));
    cudaMemcpyToSymbol(c_wfc3, w_fc3.data(), w_fc3.size() * sizeof(float));
    cudaMemcpyToSymbol(c_bfc3, b_fc3.data(), b_fc3.size() * sizeof(float));

    float *d_in, *d_c1, *d_p1, *d_c2, *d_p2, *d_fc1, *d_fc2, *d_out;
    float *d_wfc1, *d_bfc1;

    cudaMalloc(&d_in, 3 * 32 * 32 * sizeof(float));
    cudaMalloc(&d_c1, 6 * 28 * 28 * sizeof(float));
    cudaMalloc(&d_p1, 6 * 14 * 14 * sizeof(float));
    cudaMalloc(&d_c2, 16 * 10 * 10 * sizeof(float));
    cudaMalloc(&d_p2, 16 * 5 * 5 * sizeof(float));
    cudaMalloc(&d_fc1, 120 * sizeof(float));
    cudaMalloc(&d_fc2, 84 * sizeof(float));
    cudaMalloc(&d_out, 10 * sizeof(float));
    
    cudaMalloc(&d_wfc1, w_fc1.size() * sizeof(float));
    cudaMalloc(&d_bfc1, b_fc1.size() * sizeof(float));

    cudaMemcpy(d_in, h_in.data(), h_in.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_wfc1, w_fc1.data(), w_fc1.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bfc1, b_fc1.data(), b_fc1.size() * sizeof(float), cudaMemcpyHostToDevice);

    cout << "Executing OPTIMIZED Forward Pass for Profiling...\n";

    // conv1
    dim3 grid_c1((28 + 15) / 16, (28 + 15) / 16, 6);
    conv1_shared<<<grid_c1, dim3(16, 16, 1)>>>(d_in, d_c1, 32, 32, 28, 28);

    // pool1
    dim3 grid_p1((14 + 15) / 16, (14 + 15) / 16, 6);
    pool_kernel<<<grid_p1, dim3(16,16,1)>>>(d_c1, d_p1, 6, 28, 28, 14, 14, 2, 2);

    // conv2
    dim3 grid_c2(1, 1, 16); 
    conv2_shared<<<grid_c2, dim3(16,16,1)>>>(d_p1, d_c2, 6, 14, 14, 16, 10, 10, 5);

    // pool2
    dim3 grid_p2((5 + 15) / 16, (5 + 15) / 16, 16);
    pool_kernel<<<grid_p2, dim3(16,16,1)>>>(d_c2, d_p2, 16, 10, 10, 5, 5, 2, 2);

    // fc1
    int th_fc1 = 128;
    fc1_shared<<<120, th_fc1, th_fc1 * sizeof(float)>>>(d_p2, d_wfc1, d_bfc1, d_fc1, 400);

    // fc2
    int th_fc2 = 64;
    fc2_shared<<<84, th_fc2, th_fc2 * sizeof(float)>>>(d_fc1, d_fc2, 120);

    // fc3
    int th_fc3 = 32;
    fc3_shared<<<10, th_fc3, th_fc3 * sizeof(float)>>>(d_fc2, d_out, 84);
    
    cudaDeviceSynchronize();

    vector<float> h_out(10);
    cudaMemcpy(h_out.data(), d_out, 10 * sizeof(float), cudaMemcpyDeviceToHost);
    
    float max_val = h_out[0];
    int pred = 0;
    for (int i = 1; i < 10; ++i) {
        if (h_out[i] > max_val) { max_val = h_out[i]; pred = i; }
    }
    
    cout << "\n====================================\n";
    cout << ">> PREDICTION CLASS: " << pred << " <<\n";
    cout << "====================================\n";

    cudaFree(d_in); cudaFree(d_c1); cudaFree(d_p1); cudaFree(d_c2); 
    cudaFree(d_p2); cudaFree(d_fc1); cudaFree(d_fc2); cudaFree(d_out);
    cudaFree(d_wfc1); cudaFree(d_bfc1);

    return 0;
}