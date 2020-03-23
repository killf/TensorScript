#include <iostream>
#include "tensor.hpp"

namespace TensorScript {

    CustomError::CustomError(string msg) : _msg(std::move(msg)) {}

    CustomError::CustomError(const char *__restrict format, ...) {
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

    const char *CustomError::what() const noexcept {
        return _msg.c_str();
    }

    TensorShape::TensorShape(const initializer_list<int> &list) : _dims(list) {
        for (auto i:_dims)_size *= i;
    }

    inline int TensorShape::operator[](int index) const { return _dims[index]; }

    int TensorShape::index(const initializer_list<int> &index) const {
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

    inline string TensorShape::toString() const {
        stringstream ss;
        ss << this;
        return ss.str();
    }

    void TensorShape::check_index(const vector<int> &index) const {
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

    ostream &operator<<(ostream &stream, const TensorShape &obj) {
        stream << "(";
        if (obj.ndim() > 0) stream << obj[0];
        for (int i = 1; i < obj.ndim(); i++)stream << "," << obj[i];
        return stream << ")";
    }

    template<typename T, DataType dtype>
    bool check_type() {
        if (is_same<T, int>::value && dtype == I8) {
            return true;
        }
    }

    template<typename T>
    Tensor<T>::Tensor(const TensorShape &shape) : _shape(shape) {
        _data.reset(new T[shape.size()]());
    }

    template<typename T>
    Tensor<T>::Tensor(const TensorShape &shape, T value) : Tensor(shape) {
        fill(value);
    }

    template<typename T>
    void Tensor<T>::reshape(const TensorShape &shape) {
        if (shape.size() != _shape.size()) {
            stringstream ss;
            ss << "ValueError: cannot reshape tensor of size " << _shape.size() << " into shape " << shape;
            throw CustomError(ss.str());
        }

        _shape = shape;
    }

    template<typename T>
    void Tensor<T>::fill(T value) {
        T *ptr = _data.get();
        for (auto i = 0; i < _shape.size(); i++, ptr++) {
            *ptr = value;
        }
    }

    template<typename T>
    T Tensor<T>::operator[](const initializer_list<int> &index) const {
        auto ind = _shape.index(index);
        return _data.get()[ind];
    }

    template<typename T>
    T &Tensor<T>::operator[](const initializer_list<int> &index) {
        auto ind = _shape.index(index);
        return _data.get()[ind];
    }

    template<typename T>
    inline string Tensor<T>::toString() const {
        stringstream ss;
        ss << this;
        return ss.str();
    }

    template<typename T>
    ostream &operator<<(ostream &stream, const Tensor<T> &obj) {
        if (obj.name().empty())
            return stream << "Tensor" << obj.shape().toString();
        else
            return stream << "Tensor:" << obj.name() << obj.shape().toString();
    }

    template
    class Tensor<float>;
}
