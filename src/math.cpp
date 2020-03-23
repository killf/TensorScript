//#include "math.hpp"
//
//namespace TensorScript {
//		template<typename T>
//		void add(const Tensor<T> &a, const Tensor<T> &b, Tensor<T> &dst) {
//			for (int i = 0; i < a.shape().size(); i++) {
//				dst.data()[i] = a.data()[i] + b.data()[i];
//			}
//		}
//		template void add<float>(const Tensor<float> &a, const Tensor<float> &b, Tensor<float> &dst);
//
//}