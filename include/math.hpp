#ifndef TENSORSCRIPT_MATH_HPP
#define TENSORSCRIPT_MATH_HPP

#include "tensor.hpp"

namespace TensorScript {
		void add(const Tensor &a, const Tensor &b, Tensor &dst);

		void sub(const Tensor &a, const Tensor &b, Tensor &dst);
}

#endif //TENSORSCRIPT_MATH_HPP
