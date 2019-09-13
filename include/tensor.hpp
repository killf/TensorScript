#ifndef TENSORSCRIPT_TENSOR_HPP
#define TENSORSCRIPT_TENSOR_HPP

#include <memory>

namespace TensorScript {
    using namespace std;

    /**
     * 数据类型
     * */
    enum DataType {
        i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64
    };

    /**
     * 张量的形状
     * */
    class TensorShape {

    };

    /**
     * 张量
     * */
    class Tensor {
    public:
        Tensor(TensorShape shape, DataType dataType);

    public:
        void reshape(TensorShape shape);

        Tensor asType(DataType data_type);

        Tensor slice();

    public:
        inline void *getData() const { return data.get(); }

        inline TensorShape getShape() const { return shape; }

        inline DataType getDataType() const { return data_type; }

    protected:
        shared_ptr<void> data;      // 允许多个张量共享同一块内存，例如：切片、广播
        TensorShape shape;
        DataType data_type;
    };
}

#endif //TENSORSCRIPT_TENSOR_HPP
