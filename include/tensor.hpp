#ifndef TENSORSCRIPT_TENSOR_HPP
#define TENSORSCRIPT_TENSOR_HPP

#include <memory>
#include <vector>
#include <sstream>
#include <assert.h>
#include <stdarg.h>
#include <cstring>

namespace TensorScript {
    using namespace std;

    /**
     * 自定义错误
     * */
    class CustomError : public exception {
    public:
        explicit CustomError(string msg);

        explicit CustomError(const char *__restrict format, ...);

        const char *what() const noexcept override;

    private:
        string _msg;
    };

    /**
     * 张量的形状
     * */
    class TensorShape {
    public:
        /**
         * 默认构造函数
         * */
        TensorShape() = default;

        /**
         * 构造TensorShape
         * */
        TensorShape(const initializer_list<int> &list);

        /**
         * 访问指定的维度
         * */
        inline int operator[](int index) const;

        /**
         * 根据下表计算索引
         * */
        int index(const initializer_list<int> &index) const;

        /**
         * 格式化
         * */
        inline string toString() const;

        /**
         * 获取元素的个数，标量的size为1，但ndim为0
         * */
        inline int size() const { return _size; };

        /**
         * 获取维数
         * */
        inline int ndim() const { return _dims.size(); };
    private:
        void check_index(const vector<int> &index) const;

    private:
        vector<int> _dims;
        int _size = 1;
    };

    ostream &operator<<(ostream &stream, const TensorShape &obj);

    /**
     * 数据类型
     * */
    enum DataType {
        I8, I16, I32, I64,
        U8, U16, U32, U64,
        f16, f32, f64
    };

    template<typename T, DataType dtype>
    bool check_type();

    /**
     * 张量
     * */
    template<typename T>
    class Tensor {
    public:
        /**
         * 构造张量，并分配空间
         * */
        explicit Tensor(const TensorShape &shape);

        /**
         * 构造张量，并使用指定的值进行初始化
         * */
        Tensor(const TensorShape &shape, T value);

    public:
        /**
         * 调整张量的shape
         * */
        void reshape(const TensorShape &shape);

        /**
         * 使用指定的值填充张量
         * */
        void fill(T value);

    public:
        /**
         * 访问张量中的元素
         * */
        T operator[](const initializer_list<int> &index) const;

        /**
         * 访问张量中的元素
         * */
        T &operator[](const initializer_list<int> &index);

        /**
         * 格式化
         * */
        inline string toString() const;

    public:
        /**
         * 张量的shape
         * */
        inline TensorShape shape() const { return _shape; };

        /**
         * 张量的地址
         * */
        inline T *data() const { return _data.get(); };

    private:
        TensorShape _shape;
        shared_ptr<T> _data;
    };

    template<typename T>
    ostream &operator<<(ostream &stream, const Tensor<T> &obj);
}

#endif //TENSORSCRIPT_TENSOR_HPP
