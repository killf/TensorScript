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
        explicit CustomError(string msg) : _msg(std::move(msg)) {}

        explicit CustomError(const char *__restrict format, ...) {
            static char buf[512];
            int max_len = sizeof(buf);

            va_list ap;
            va_start(ap, format);
            int len = vsnprintf(buf, max_len - 1, format, ap);
            va_end(ap);

            if (len < max_len) {
                _msg = string(buf);
            } else {
                auto str = new char[len + 2];

                va_start(ap, format);
                vsprintf(str, format, ap);
                va_end(ap);

                _msg = string(str);
                delete[] str;
            }
        }

        const char *what() const noexcept override {
            return _msg.c_str();
        }

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
        TensorShape(const initializer_list<int> &list) : _dims(list) {
            for (auto i:_dims)_size *= i;
        }

        /**
         * 获取元素的个数，标量的size为1，但ndim为0
         * */
        inline int size() const { return _size; }

        /**
         * 获取维数
         * */
        inline int ndim() const { return _dims.size(); }

        /**
         * 访问指定的维度
         * */
        inline int operator[](int index) const { return _dims[index]; }

        /**
         * 根据下表计算索引
         * */
        int index(const initializer_list<int> &index) const {
            vector<int> index_v(index);
            check_index(index_v);

            int ind = 0;
            for (int i = 0; i < _dims.size(); i++) {
                int step = index_v[i];
                for (int j = i + 1; j < _dims.size(); j++) step *= _dims[j];
                ind += step;
            }

            return ind;
        }

        /**
         * 格式化
         * */
        inline string toString() const {
            stringstream ss;
            ss << this;
            return ss.str();
        }

    private:
        void check_index(const vector<int> &index) const {
            if (index.size() > ndim()) {
                throw CustomError("IndexError：too many indices for array");
            } else if (index.size() < ndim()) {
                throw CustomError("IndexError：暂不支持切片");
            }

            for (int i = 0; i < _dims.size(); i++) {
                if (index[i] > _dims[i]) {
                    throw CustomError("IndexError：index %d is out of bounds for axis %d with size %d", index[i], i,
                                      _dims[i]);
                }
            }
        }

    private:
        vector<int> _dims;
        int _size = 1;
    };

    ostream &operator<<(ostream &stream, const TensorShape &obj) {
        stream << "(";
        if (obj.ndim() > 0) stream << obj[0];
        for (int i = 1; i < obj.ndim(); i++)stream << "," << obj[i];
        return stream << ")";
    }

    /**
     * 数据缓存区
     * */
    class Buffer {
    public:
        Buffer() = default;

        Buffer(size_t size) : _size(size) {
            _data = aligned_alloc(32, size);
        }

    public:
        inline void *data() const { return _data; }

        inline size_t size() const { return _size; }

    private:
        void *_data;
        size_t _size;
    };

    /**
     * 数据类型
     * */
    enum DataType {
        I8, I16, I32, I64,
        U8, U16, U32, U64,
        f16, f32, f64
    };

    template<typename T, DataType dtype>
    bool check_type() {
        if (is_same<T, int>::value && dtype == I8) {
            return true;
        }
    }

    /**
     * 张量
     * */
    class Tensor {
    public:
        /**
         * 构造张量，并分配空间
         * */
        explicit Tensor(const TensorShape &shape) : _shape(shape) {
            _buf.reset(new Buffer(shape.size()));
        }

        /**
         * 构造张量，并使用指定的值进行初始化
         * */
        Tensor(const TensorShape &shape, T value) : Tensor(shape) {
            fill(value);
        }

    public:
        /**
         * 调整张量的shape
         * */
        void reshape(const TensorShape &shape) {
            if (shape.size() != _shape.size()) {
                stringstream ss;
                ss << "ValueError: cannot reshape tensor of size " << _shape.size() << " into shape " << shape;
                throw CustomError(ss.str());
            }

            _shape = shape;
        }

        /**
         * 使用指定的值填充张量
         * */
        template<typename T>
        void fill(T value) {

            auto ptr = data<T>();
            for (auto i = 0; i < _shape.size(); i++, ptr++) {
                *ptr = value;
            }
        }

        /**
         * 使用1填充张量，等价于`fill(1)`
         * */
        inline void ones() {
            fill(static_cast<T>(1));
        }

        /**
         * 使用0填充张量，等价于`fill(0)`
         * */
        inline void zeros() {
            fill(static_cast<T>(0));
        }

    public:
        /**
         * 访问张量中的元素
         * */
        template<typename T>
        T operator[](const initializer_list<int> &index) const {
            auto ind = _shape.index(index);
            return data<T>()[ind];
        }

        /**
         * 访问张量中的元素
         * */
        template<typename T>
        T &operator[](const initializer_list<int> &index) {
            auto ind = _shape.index(index);
            return (data<T>()[ind]);
        }

        /**
         * 格式化
         * */
        inline string toString() const {
            stringstream ss;
            ss << this;
            return ss.str();
        }

    public:
        /**
         * 张量的名称，可选值
         * */
        inline string name() const { return _name; }

        /**
         * 张量的shape
         * */
        inline TensorShape shape() const { return _shape; }

        /**
         * 张量的地址
         * */
        template<typename T>
        inline T *data() const { return dynamic_pointer_cast<T>(_buf->data()); }

    private:
        string _name;
        TensorShape _shape;
        DataType _dtype;
        shared_ptr<Buffer> _buf;
    };

    ostream &operator<<(ostream &stream, const Tensor &obj) {
        if (obj.name().empty())
            return stream << "Tensor" << obj.shape().toString();
        else
            return stream << "Tensor:" << obj.name() << obj.shape().toString();
    }
}

#endif //TENSORSCRIPT_TENSOR_HPP
