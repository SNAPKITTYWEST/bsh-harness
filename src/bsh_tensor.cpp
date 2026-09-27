#include "bsh_tensor.hpp"
#include <cstring>
#include <cmath>

namespace bsh {

Tensor::Tensor(size_t rows, size_t cols) 
    : rows_(rows), cols_(cols), data_(new float[rows * cols]) {
    std::memset(data_.get(), 0, rows * cols * sizeof(float));
}

Tensor::~Tensor() = default;

void Tensor::fill(float value) {
    for (size_t i = 0; i < rows_ * cols_; ++i) {
        data_[i] = value;
    }
}

float& Tensor::at(size_t row, size_t col) {
    return data_[row * cols_ + col];
}

const float& Tensor::at(size_t row, size_t col) const {
    return data_[row * cols_ + col];
}

float* Tensor::data() {
    return data_.get();
}

const float* Tensor::data() const {
    return data_.get();
}

size_t Tensor::rows() const { return rows_; }
size_t Tensor::cols() const { return cols_; }
size_t Tensor::size() const { return rows_ * cols_; }

Tensor Tensor::transpose() const {
    Tensor result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i) {
        for (size_t j = 0; j < cols_; ++j) {
            result.at(j, i) = this->at(i, j);
        }
    }
    return result;
}

void Tensor::softmax(Tensor& output) const {
    if (output.rows() != rows_ || output.cols() != cols_) {
        throw std::runtime_error("Output tensor dimension mismatch");
    }

    for (size_t i = 0; i < rows_; ++i) {
        float max_val = data_[i * cols_];
        for (size_t j = 0; j < cols_; ++j) {
            if (data_[i * cols_ + j] > max_val) {
                max_val = data_[i * cols_ + j];
            }
        }

        float sum = 0.0f;
        for (size_t j = 0; j < cols_; ++j) {
            output.data_[i * cols_ + j] = std::exp(data_[i * cols_ + j] - max_val);
            sum += output.data_[i * cols_ + j];
        }

        for (size_t j = 0; j < cols_; ++j) {
            output.data_[i * cols_ + j] /= sum;
        }
    }
}

} // namespace bsh
