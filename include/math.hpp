#ifndef TENSORSCRIPT_MATH_HPP
#define TENSORSCRIPT_MATH_HPP

#include "tensor.hpp"

namespace TensorScript {
  template<typename T1, typename T2, typename T3>
  void add(const Tensor<T1> &a, const Tensor<T2> &b, Tensor<T3> &dst) {
    auto length = a.shape().size();
    auto pa = a.data(), pb = b.data(), pc = dst.data();

    for (int i = 0; i < length; i++, pa++, pb++, pc++) {
      *pc = *pa + *pb;
    }
  }


  template<typename T1, typename T2, typename T3>
  void sub(const Tensor<T1> &a, const Tensor<T2> &b, Tensor<T3> &dst) {
    auto length = a.shape().size();
    auto pa = a.data(), pb = b.data(), pc = dst.data();

    for (int i = 0; i < length; i++, pa++, pb++, pc++) {
      *pc = *pa - *pb;
    }
  }

  template<typename T1, typename T2, typename T3>
  void mul(const Tensor<T1> &a, const Tensor<T2> &b, Tensor<T3> &dst) {
    auto length = a.shape().size();
    auto pa = a.data(), pb = b.data(), pc = dst.data();

    for (int i = 0; i < length; i++, pa++, pb++, pc++) {
      *pc = *pa * *pb;
    }
  }

  template<typename T1, typename T2, typename T3>
  void div(const Tensor<T1> &a, const Tensor<T2> &b, Tensor<T3> &dst) {
    auto length = a.shape().size();
    auto pa = a.data(), pb = b.data(), pc = dst.data();

    for (int i = 0; i < length; i++, pa++, pb++, pc++) {
      *pc = *pa / *pb;
    }
  }
}

#endif //TENSORSCRIPT_MATH_HPP
