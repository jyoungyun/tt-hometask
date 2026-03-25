#include <vector>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <iostream>
#include <iomanip>

constexpr uint32_t kRows = 32;
constexpr uint32_t kCols = 1024;

std::vector<float> make_input() {
    std::vector<float> input(kRows * kCols);
    for (size_t row = 0; row < kRows; ++row) {
       for (size_t col = 0; col < kCols; ++col) {
          input[row * kCols + col] = static_cast<float>(col % 8);
       }
    }
    return input;
}

std::vector<float> softmax_reference(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    for (size_t row = 0; row < kRows; ++row) {
        int base = row * kCols;

        float max_val = input[base];
        for (size_t col = 1; col < kCols; ++col) {
            max_val = std::max(max_val, input[base+col]);
        }

        float sum = 0.f;
        for (size_t col = 0; col < kCols; ++col) {
            float e = std::exp(input[base+col] - max_val);
            output[base + col] = e;
            sum += e;
        }

        for (size_t col = 0; col < kCols; ++col) {
            output[base + col] /= sum;
        }
    }

    return output;
}

int main() {

    auto input = make_input();
    auto golden = softmax_reference(input);

    std::cout << "input" << std::endl;
    for (size_t i = 0; i < 10; ++i) {
        std::cout << input[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "golden" << std::endl;
    for (size_t i = 0; i < 10; ++i) {
        std::cout << std::fixed << std::setprecision(6) << golden[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}

