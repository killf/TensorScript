#ifndef TENSORSCRIPT_MATH_HPP
#define TENSORSCRIPT_MATH_HPP

#include "tensor.hpp"

namespace TensorScript {
		template<typename T>
		void add(const Tensor <T> &a, const Tensor <T> &b, Tensor <T> &dst);

		template<typename T>
		void sub(const Tensor <T> &a, const Tensor <T> &b, Tensor <T> &dst);
}

#endif //TENSORSCRIPT_MATH_HPP
