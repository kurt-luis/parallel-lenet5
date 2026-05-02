#include <iostream>
#include <vector>
#include "../../include/utils/file_io.h" // Include your new header

using namespace std;

inline int idx4d(int n, int c, int h, int w, int C, int H, int W) {
    return n * (C * H * W) + c * (H * W) + h * W + w;
}

int main() {
    try {
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

        verify_tensors(my_conv1, golden_conv1, 1e-4);

    } catch (const exception& e) {
        cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}