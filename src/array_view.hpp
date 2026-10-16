#pragma once
#include <cstddef>
#include <stdexcept>

namespace kf {

// Non-owning contiguous array view. Drop-in for const std::vector<T>& input
// parameters so Python bindings can pass numpy buffers without copying.
template <typename T>
class ConstArray {
public:
    ConstArray() : p_(nullptr), n_(0) {}
    ConstArray(const T *p, std::size_t n) : p_(p), n_(n) {
        if (n && !p) throw std::invalid_argument("ConstArray: null data with n>0");
    }

    const T *data() const { return p_; }
    std::size_t size() const { return n_; }
    bool empty() const { return n_ == 0; }
    const T &operator[](std::size_t i) const { return p_[i]; }
    const T &at(std::size_t i) const {
        if (i >= n_) throw std::out_of_range("ConstArray::at");
        return p_[i];
    }
    const T *begin() const { return p_; }
    const T *end() const { return p_ + n_; }

private:
    const T *p_;
    std::size_t n_;
};

}  // namespace kf
