#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>

namespace bsh {

class Tensor {
public:
    explicit Tensor(size_t rows, size_t cols);
    ~Tensor();

    Tensor(const Tensor&) = default;
    Tensor& operator=(const Tensor&) = default;
    Tensor(Tensor&&) = default;
    Tensor& operator=(Tensor&&) = default;

    void fill(float value);
    float& at(size_t row, size_t col);
    const float& at(size_t row, size_t col) const;
    
    float* data();
    const float* data() const;
    
    size_t rows() const;
    size_t cols() const;
    size_t size() const;
    
    Tensor transpose() const;
    void softmax(Tensor& output) const;

private:
    size_t rows_;
    size_t cols_;
    std::unique_ptr<float[]> data_;
};

} // namespace bsh
