#include <iostream>
#include <vector>
#include <limits>
#include <algorithm>
#include "../../include/utils/file_io.h"

using namespace std;

inline int idx4d(int n, int c, int h, int w, int C, int H, int W) {
    return n * (C * H * W) + c * (H * W) + h * W + w;
}

int main() {
    try {
        
        // FIRST LAYER: Conv1
        auto input = load_binary("../data/processed/test_image_0.bin");
        auto weights = load_binary("../weights/conv1_weight.bin");
        auto biases = load_binary("../weights/conv1_bias.bin");
        auto golden_conv1 = load_binary("../results/predictions/golden_conv1.bin");

        int N = 1, InC = 3, InH = 32, InW = 32, OutC = 6, K = 5;
        int OutH = InH - K + 1;
        int OutW = InW - K + 1;

        vector<float> my_conv1(N * OutC * OutH * OutW, 0.0f);

        for (int n = 0; n < N; ++n) {
            for (int oc = 0; oc < OutC; ++oc) {
                for (int oh = 0; oh < OutH; ++oh) {
                    for (int ow = 0; ow < OutW; ++ow) {
                        float sum = biases[oc];
                        for (int ic = 0; ic < InC; ++ic) {
                            for (int kh = 0; kh < K; ++kh) {
                                for (int kw = 0; kw < K; ++kw) {
                                    int in_idx = idx4d(n, ic, oh + kh, ow + kw, InC, InH, InW);
                                    int w_idx = idx4d(oc, ic, kh, kw, InC, K, K);
                                    sum += input[in_idx] * weights[w_idx];
                                }
                            }
                        }
                        int out_idx = idx4d(n, oc, oh, ow, OutC, OutH, OutW);
                        my_conv1[out_idx] = sum;
                    }
                }
            }
        }

        cout << "Verifying Conv1...\n";
        verify_tensors(my_conv1, golden_conv1, 1e-4);

        // SECOND LAYER: ReLU + Max Pool 1
        auto golden_pool1 = load_binary("../results/predictions/golden_pool1.bin");

        int PoolK = 2;      
        int PoolS = 2;      
        int PoolOutH = OutH / PoolS; 
        int PoolOutW = OutW / PoolS; 

        vector<float> my_pool1(N * OutC * PoolOutH * PoolOutW, 0.0f);

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < OutC; ++c) {
                for (int ph = 0; ph < PoolOutH; ++ph) {
                    for (int pw = 0; pw < PoolOutW; ++pw) {
                        
                        float max_val = -numeric_limits<float>::infinity();

                        for (int kh = 0; kh < PoolK; ++kh) {
                            for (int kw = 0; kw < PoolK; ++kw) {
                                
                                int ih = ph * PoolS + kh;
                                int iw = pw * PoolS + kw;
                                
                                int in_idx = idx4d(n, c, ih, iw, OutC, OutH, OutW);
                                float val = my_conv1[in_idx];
                                
                                val = max(0.0f, val);
                                max_val = max(max_val, val);
                            }
                        }

                        int out_idx = idx4d(n, c, ph, pw, OutC, PoolOutH, PoolOutW);
                        my_pool1[out_idx] = max_val;
                    }
                }
            }
        }

        cout << "Verifying Pool1...\n";
        verify_tensors(my_pool1, golden_pool1, 1e-4);

        // THIRD LAYER: Conv2
        auto weights2 = load_binary("../weights/conv2_weight.bin");
        auto biases2 = load_binary("../weights/conv2_bias.bin");
        auto golden_conv2 = load_binary("../results/predictions/golden_conv2.bin");

        // Parameters 
        int InC2 = 6, InH2 = 14, InW2 = 14;
        int OutC2 = 16, K2 = 5;
        int OutH2 = InH2 - K2 + 1; // 14 - 5 + 1 = 10
        int OutW2 = InW2 - K2 + 1; // 14 - 5 + 1 = 10

        vector<float> my_conv2(N * OutC2 * OutH2 * OutW2, 0.0f);

        for (int n = 0; n < N; ++n) {
            for (int oc = 0; oc < OutC2; ++oc) {
                for (int oh = 0; oh < OutH2; ++oh) {
                    for (int ow = 0; ow < OutW2; ++ow) {
                        
                        float sum = biases2[oc];
                        
                        for (int ic = 0; ic < InC2; ++ic) {
                            for (int kh = 0; kh < K2; ++kh) {
                                for (int kw = 0; kw < K2; ++kw) {
                                    
                                    int in_idx = idx4d(n, ic, oh + kh, ow + kw, InC2, InH2, InW2);
                                    int w_idx = idx4d(oc, ic, kh, kw, InC2, K2, K2);
                                    
                                    sum += my_pool1[in_idx] * weights2[w_idx]; 
                                }
                            }
                        }
                        
                        int out_idx = idx4d(n, oc, oh, ow, OutC2, OutH2, OutW2);
                        my_conv2[out_idx] = sum;
                    }
                }
            }
        }

        cout << "Verifying Conv2...\n";
        verify_tensors(my_conv2, golden_conv2, 1e-4);

        // FOURTH LATER: ReLU + Max Pool 2
        auto golden_pool2 = load_binary("../results/predictions/golden_pool2.bin");

        int PoolK2 = 2;      
        int PoolS2 = 2;      
        int PoolOutH2 = OutH2 / PoolS2; // 10 / 2 = 5
        int PoolOutW2 = OutW2 / PoolS2; // 10 / 2 = 5

        vector<float> my_pool2(N * OutC2 * PoolOutH2 * PoolOutW2, 0.0f);

        for (int n = 0; n < N; ++n) {
            for (int c = 0; c < OutC2; ++c) {
                for (int ph = 0; ph < PoolOutH2; ++ph) {
                    for (int pw = 0; pw < PoolOutW2; ++pw) {
                        float max_val = 0.0f;

                        for (int kh = 0; kh < PoolK2; ++kh) {
                            for (int kw = 0; kw < PoolK2; ++kw) {
                                int ih = ph * PoolS2 + kh;
                                int iw = pw * PoolS2 + kw;
                                
                                int in_idx = idx4d(n, c, ih, iw, OutC2, OutH2, OutW2);
                                float val = my_conv2[in_idx]; 
                                
                                max_val = max(max_val, val);
                            }
                        }
                        
                        int out_idx = idx4d(n, c, ph, pw, OutC2, PoolOutH2, PoolOutW2);
                        my_pool2[out_idx] = max_val;
                    }
                }
            }
        }

        cout << "Verifying Pool2...\n";
        verify_tensors(my_pool2, golden_pool2, 1e-4);

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}