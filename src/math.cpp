#include "math.hpp"

namespace TensorScript {
		void add(const Tensor &a, const Tensor &b, Tensor &dst) {
			auto length = a.shape().size();
			auto pa = a.data(), pb = b.data(), pc = dst.data();

			for (int i = 0; i < length; i++, pa++, pb++, pc++) {
				*pc = *pa + *pb;
			}
		}

		void sub(const Tensor &a, const Tensor &b, Tensor &dst) {
			auto length = a.shape().size();
			auto pa = a.data(), pb = b.data(), pc = dst.data();

			for (int i = 0; i < length; i++, pa++, pb++, pc++) {
				*pc = *pa - *pb;
			}
		}

}