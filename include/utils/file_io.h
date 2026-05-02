#ifndef FILE_IO_H
#define FILE_IO_H

#include <vector>
#include <string>

// Declarations only
std::vector<float> load_binary(const std::string& filepath);
bool verify_tensors(const std::vector<float>& my_out, const std::vector<float>& gold_out, float eps = 1e-4);

#endif // FILE_IO_H