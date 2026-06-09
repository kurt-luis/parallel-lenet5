#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <limits>
#include <algorithm>

using namespace std;
using namespace std::chrono;

inline int idx4d(int n, int c, int h, int w, int C, int H, int W) {
    return n * (C * H * W) + c * (H * W) + h * W + w;
}

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

int main() {
    vector<float> h_in, w_c1, b_c1, w_c2, b_c2, w_fc1, b_fc1, w_fc2, b_fc2, w_fc3, b_fc3;
    
    readBin("../data/processed/test_image_0.bin", h_in);
    readBin("../weights/conv1_weight.bin", w_c1); readBin("../weights/conv1_bias.bin", b_c1);
    readBin("../weights/conv2_weight.bin", w_c2); readBin("../weights/conv2_bias.bin", b_c2);
    readBin("../weights/fc1_weight.bin", w_fc1);  readBin("../weights/fc1_bias.bin", b_fc1);
    readBin("../weights/fc2_weight.bin", w_fc2);  readBin("../weights/fc2_bias.bin", b_fc2);
    readBin("../weights/fc3_weight.bin", w_fc3);  readBin("../weights/fc3_bias.bin", b_fc3);

    int N = 1;

    cout << "Executing CPU Forward Pass...\n\n";

    high_resolution_clock::time_point t1, t2;
    double time_c1, time_p1, time_c2, time_p2, time_fc1, time_fc2, time_fc3;

    // conv1
    int InC = 3, InH = 32, InW = 32, OutC = 6, K = 5;
    int OutH = InH - K + 1;
    int OutW = InW - K + 1;
    vector<float> my_conv1(N * OutC * OutH * OutW, 0.0f);

    t1 = high_resolution_clock::now();
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OutC; ++oc) {
            for (int oh = 0; oh < OutH; ++oh) {
                for (int ow = 0; ow < OutW; ++ow) {
                    float sum = b_c1[oc];
                    for (int ic = 0; ic < InC; ++ic) {
                        for (int kh = 0; kh < K; ++kh) {
                            for (int kw = 0; kw < K; ++kw) {
                                int in_idx = idx4d(n, ic, oh + kh, ow + kw, InC, InH, InW);
                                int w_idx = idx4d(oc, ic, kh, kw, InC, K, K);
                                sum += h_in[in_idx] * w_c1[w_idx];
                            }
                        }
                    }
                    int out_idx = idx4d(n, oc, oh, ow, OutC, OutH, OutW);
                    my_conv1[out_idx] = sum;
                }
            }
        }
    }
    t2 = high_resolution_clock::now();
    time_c1 = duration_cast<microseconds>(t2 - t1).count();

    // pool1
    int PoolK = 2, PoolS = 2;      
    int PoolOutH = OutH / PoolS; 
    int PoolOutW = OutW / PoolS; 
    vector<float> my_pool1(N * OutC * PoolOutH * PoolOutW, 0.0f);

    t1 = high_resolution_clock::now();
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
                            val = max(0.0f, val); // ReLU integrated here
                            max_val = max(max_val, val);
                        }
                    }
                    int out_idx = idx4d(n, c, ph, pw, OutC, PoolOutH, PoolOutW);
                    my_pool1[out_idx] = max_val;
                }
            }
        }
    }
    t2 = high_resolution_clock::now();
    time_p1 = duration_cast<microseconds>(t2 - t1).count();

    // conv2
    int InC2 = 6, InH2 = 14, InW2 = 14;
    int OutC2 = 16, K2 = 5;
    int OutH2 = InH2 - K2 + 1; 
    int OutW2 = InW2 - K2 + 1; 
    vector<float> my_conv2(N * OutC2 * OutH2 * OutW2, 0.0f);

    t1 = high_resolution_clock::now();
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < OutC2; ++oc) {
            for (int oh = 0; oh < OutH2; ++oh) {
                for (int ow = 0; ow < OutW2; ++ow) {
                    float sum = b_c2[oc];
                    for (int ic = 0; ic < InC2; ++ic) {
                        for (int kh = 0; kh < K2; ++kh) {
                            for (int kw = 0; kw < K2; ++kw) {
                                int in_idx = idx4d(n, ic, oh + kh, ow + kw, InC2, InH2, InW2);
                                int w_idx = idx4d(oc, ic, kh, kw, InC2, K2, K2);
                                sum += my_pool1[in_idx] * w_c2[w_idx]; 
                            }
                        }
                    }
                    int out_idx = idx4d(n, oc, oh, ow, OutC2, OutH2, OutW2);
                    my_conv2[out_idx] = sum;
                }
            }
        }
    }
    t2 = high_resolution_clock::now();
    time_c2 = duration_cast<microseconds>(t2 - t1).count();

    // pool 2
    int PoolK2 = 2, PoolS2 = 2;      
    int PoolOutH2 = OutH2 / PoolS2; 
    int PoolOutW2 = OutW2 / PoolS2; 
    vector<float> my_pool2(N * OutC2 * PoolOutH2 * PoolOutW2, 0.0f);

    t1 = high_resolution_clock::now();
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < OutC2; ++c) {
            for (int ph = 0; ph < PoolOutH2; ++ph) {
                for (int pw = 0; pw < PoolOutW2; ++pw) {
                    float max_val = 0.0f; // ReLU integrated via 0.0f initialization
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
    t2 = high_resolution_clock::now();
    time_p2 = duration_cast<microseconds>(t2 - t1).count();

    // fc1
    int InFeatures1 = 16 * 5 * 5, OutFeatures1 = 120;
    vector<float> my_fc1(N * OutFeatures1, 0.0f);

    t1 = high_resolution_clock::now();
    for (int n = 0; n < N; ++n) {
        for (int out_idx = 0; out_idx < OutFeatures1; ++out_idx) {
            float sum = b_fc1[out_idx];
            for (int in_idx = 0; in_idx < InFeatures1; ++in_idx) {
                int w_idx = out_idx * InFeatures1 + in_idx;
                sum += my_pool2[n * InFeatures1 + in_idx] * w_fc1[w_idx];
            }
            my_fc1[n * OutFeatures1 + out_idx] = max(0.0f, sum);
        }
    }
    t2 = high_resolution_clock::now();
    time_fc1 = duration_cast<microseconds>(t2 - t1).count();

    // fc2
    int InFeatures2 = 120, OutFeatures2 = 84;
    vector<float> my_fc2(N * OutFeatures2, 0.0f);

    t1 = high_resolution_clock::now();
    for (int n = 0; n < N; ++n) {
        for (int out_idx = 0; out_idx < OutFeatures2; ++out_idx) {
            float sum = b_fc2[out_idx];
            for (int in_idx = 0; in_idx < InFeatures2; ++in_idx) {
                int w_idx = out_idx * InFeatures2 + in_idx;
                sum += my_fc1[n * InFeatures2 + in_idx] * w_fc2[w_idx];
            }
            my_fc2[n * OutFeatures2 + out_idx] = max(0.0f, sum);
        }
    }
    t2 = high_resolution_clock::now();
    time_fc2 = duration_cast<microseconds>(t2 - t1).count();

    // fc3
    int InFeatures3 = 84, OutFeatures3 = 10;
    vector<float> my_fc3(N * OutFeatures3, 0.0f);

    t1 = high_resolution_clock::now();
    for (int n = 0; n < N; ++n) {
        for (int out_idx = 0; out_idx < OutFeatures3; ++out_idx) {
            float sum = b_fc3[out_idx];
            for (int in_idx = 0; in_idx < InFeatures3; ++in_idx) {
                int w_idx = out_idx * InFeatures3 + in_idx;
                sum += my_fc2[n * InFeatures3 + in_idx] * w_fc3[w_idx];
            }
            my_fc3[n * OutFeatures3 + out_idx] = sum;
        }
    }
    t2 = high_resolution_clock::now();
    time_fc3 = duration_cast<microseconds>(t2 - t1).count();

    // profile
    cout << "--- CPU EXECUTION TIMES (us) ---\n";
    cout << "Conv1: " << time_c1 << " us\n";
    cout << "Pool1: " << time_p1 << " us\n";
    cout << "Conv2: " << time_c2 << " us\n";
    cout << "Pool2: " << time_p2 << " us\n";
    cout << "FC1:   " << time_fc1 << " us\n";
    cout << "FC2:   " << time_fc2 << " us\n";
    cout << "FC3:   " << time_fc3 << " us\n";
    cout << "Total: " << (time_c1 + time_p1 + time_c2 + time_p2 + time_fc1 + time_fc2 + time_fc3) << " us\n";
    cout << "--------------------------------\n";

    float max_val = my_fc3[0];
    int pred = 0;
    for (int i = 1; i < 10; ++i) {
        if (my_fc3[i] > max_val) { max_val = my_fc3[i]; pred = i; }
    }
    
    cout << ">> PREDICTION CLASS: " << pred << " <<\n";

    // some metrics
    double conv1_flops = 710304.0;
    double conv2_flops = 481600.0;
    double fc1_flops   = 96120.0;

    double conv1_bytes = 32928.0;
    double conv2_bytes = 20768.0;
    double fc1_bytes   = 194560.0;

    // FLOPs / time_in_seconds / 1e9
    double conv1_gflops = (conv1_flops / (time_c1 * 1e-6)) / 1e9;
    double conv2_gflops = (conv2_flops / (time_c2 * 1e-6)) / 1e9;
    double fc1_gflops   = (fc1_flops / (time_fc1 * 1e-6)) / 1e9;

    // Bytes / time_in_seconds / 1e9
    double conv1_gbs = (conv1_bytes / (time_c1 * 1e-6)) / 1e9;
    double conv2_gbs = (conv2_bytes / (time_c2 * 1e-6)) / 1e9;
    double fc1_gbs   = (fc1_bytes / (time_fc1 * 1e-6)) / 1e9;

    cout << "\n--- HARDWARE UTILIZATION METRICS ---\n";
    cout << "Layer  | Time (us) | GFLOP/s | Bandwidth (GB/s)\n";
    cout << "-----------------------------------------------\n";
    printf("Conv1  | %9.2f | %7.2f | %14.2f\n", time_c1, conv1_gflops, conv1_gbs);
    printf("Conv2  | %9.2f | %7.2f | %14.2f\n", time_c2, conv2_gflops, conv2_gbs);
    printf("FC1    | %9.2f | %7.2f | %14.2f\n", time_fc1, fc1_gflops, fc1_gbs);
    cout << "-----------------------------------------------\n";

    return 0;
}