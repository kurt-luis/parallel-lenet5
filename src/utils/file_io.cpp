#include "../../include/utils/file_io.h"
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cmath>

using namespace std;

vector<float> load_binary(const string& filepath) {
    ifstream file(filepath, ios::binary | ios::ate);
    if (!file.is_open()) throw runtime_error("Failed to open: " + filepath);
    streamsize size = file.tellg();
    file.seekg(0, ios::beg);
    vector<float> buffer(size / sizeof(float));
    file.read(reinterpret_cast<char*>(buffer.data()), size);
    return buffer;
}

bool verify_tensors(const vector<float>& my_out, const vector<float>& gold_out, float eps) {
    if (my_out.size() != gold_out.size()) {
        cerr << "Size mismatch! Mine: " << my_out.size() << ", Gold: " << gold_out.size() << "\n";
        return false;
    }
    int errors = 0;
    for (size_t i = 0; i < my_out.size(); ++i) {
        if (abs(my_out[i] - gold_out[i]) > eps) {
            if (errors < 5) 
                cerr << "Mismatch at " << i << " | Expected: " << gold_out[i] << " Got: " << my_out[i] << "\n";
            errors++;
        }
    }
    if (errors > 0) {
        cerr << "FAILED with " << errors << " errors.\n";
        return false;
    }
    cout << "Verification PASSED!\n";
    return true;
}