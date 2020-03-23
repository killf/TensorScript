#include "math.hpp"

namespace TensorScript {
    template<typename T1, typename T2, typename T3>
    void add(const Tensor<T1> &a, const Tensor<T2> &b, Tensor<T3> &c) {
        T1 *p1 = a.data();
        T2 *p2 = b.data();
        T3 *p3 = c.data();

        for (int i = 0; i < a.shape().size(); i++, p1++, p2++, p3++) {
            *p3 = (*p1) + (*p2);
        }
    }

    template
    void add<float, float, float>(const Tensor<float> &a, const Tensor<float> &b, Tensor<float> &c);
}