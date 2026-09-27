#pragma once

#include "bsh_tensor.hpp"
#include <vector>
#include <string>

namespace bsh {

class Engine {
public:
    Engine();
    ~Engine();
    
    void initialize();
    void process(const Tensor& input, Tensor& output);
    void shutdown();
    
    std::string get_version() const;
    std::string get_status() const;
    
private:
    bool initialized_;
    std::vector<Tensor> state_;
    
    void apply_attention(const Tensor& query, const Tensor& key, 
                        const Tensor& value, Tensor& output);
};

} // namespace bsh
