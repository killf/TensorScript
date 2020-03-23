#ifndef TENSORSCRIPT_MATH_HPP
#define TENSORSCRIPT_MATH_HPP

#include "tensor.hpp"

namespace TensorScript {
    template<typename T1, typename T2, typename T3>
    void add(const Tensor <T1> &a, const Tensor <T2> &b, Tensor <T3> &c);
}

#endif //TENSORSCRIPT_MATH_HPP
