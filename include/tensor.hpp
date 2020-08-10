#ifndef TENSORSCRIPT_TENSOR_HPP
#define TENSORSCRIPT_TENSOR_HPP

#include <memory>
#include <vector>
#include <sstream>
#include <assert.h>
#include <stdarg.h>
#include <cstring>
#include <type_traits>

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
    TensorShape(const initializer_list<size_t> &list) : _dims(list) {
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
    inline int operator[](size_t index) const { return _dims[index]; }

    /**
     * 根据下表计算索引
     * */
    int index(const initializer_list<size_t> &index) const {
      vector<size_t> index_v(index);
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

      ss << "(";
      if (ndim() > 0) ss << _dims[0];
      for (int i = 1; i < ndim(); i++)ss << "," << _dims[i];
      ss << ")";

      return ss.str();
    }

  private:
    void check_index(const vector<size_t> &index) const {
      if (index.size() > ndim()) {
        throw CustomError("IndexError：too many indices for array");
      } else if (index.size() < ndim()) {
        throw CustomError("IndexError：暂不支持切片");
      }

      for (int i = 0; i < _dims.size(); i++) {
        if (index[i] > _dims[i]) {
          throw CustomError("IndexError：index %d is out of bounds for axis %d with size %d", index[i], i, _dims[i]);
        }
      }
    }

  private:
    vector<size_t> _dims;
    int _size = 1;
  };

  inline ostream &operator<<(ostream &stream, const TensorShape &obj) { return stream << obj; }

  /**
   * 张量
   * */
  template<typename T=float>
  class Tensor {
  public:
    /**
     * 构造张量，并分配空间
     * */
    explicit Tensor(const TensorShape &shape) : _shape(shape) {
      _data.reset(new T[shape.size()]);
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
    void fill(T value) {
      auto ptr = data();
      for (auto i = 0; i < _shape.size(); i++, ptr++) {
        *ptr = value;
      }
    }

  public:
    /**
     * 创建一个张量，未初始化
     * */
    static Tensor<T> *empty(const TensorShape &shape) {
      return new Tensor(shape);
    }

    /**
     * 创建一个shape与target一样的张量，未初始化
     * */
    static Tensor<T> *empty_like(const Tensor &target) {
      return new Tensor(target.shape());
    }

    /**
     * 创建一个张量，初始化为0
     * */
    static Tensor<T> *zeros(const TensorShape &shape) {
      return new Tensor(shape, 0);
    }

    /**
     * 创建一个shape与target一样张量，初始化为0
     * */
    static Tensor<T> *zeros_like(const Tensor &target) {
      return new Tensor(target.shape(), 0);
    }

    /**
     * 创建一个张量，初始化为1
     * */
    static Tensor<T> *ones(const TensorShape &shape) {
      return new Tensor(shape, 1);
    }

    /**
     * 创建一个shape与target一样张量，初始化为1
     * */
    static Tensor<T> *ones_like(const Tensor &target) {
      return new Tensor(target.shape(), 1);
    }

  public:
    /**
     * 访问张量中的元素
     * */
    T operator[](const initializer_list<size_t> &index) const {
      auto ind = _shape.index(index);
      return data()[ind];
    }

    /**
     * 访问张量中的元素
     * */
    T &operator[](const initializer_list<size_t> &index) {
      auto ind = _shape.index(index);
      return (data()[ind]);
    }

    /**
     * 格式化
     * */
    inline string toString() const {
      stringstream ss;
      ss << *this;
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
    inline size_t ndim() const { return _shape.size(); }

    /**
     * 张量的shape
     * */
    inline TensorShape shape() const { return _shape; }

    /**
     * 张量的size
     * */
    inline size_t size() const {
      size_t size = 1;
      for (size_t i = 0; i < ndim(); i++) size *= _shape[i];
      return size;
    }

    /**
     * 张量的地址
     * */
    inline T *data() const { return _data.get(); }

    /**
     * 张量的类型
     * */
    inline const char *dtype() const { return typeid(T).name(); }

  private:
    string _name;
    TensorShape _shape;
    shared_ptr<T> _data;
  };

  template<typename T>
  ostream &operator<<(ostream &stream, const Tensor<T> &obj) {
    if (obj.name().empty())
      return stream << "Tensor[" << obj.dtype() << "]:" << obj.shape().toString();
    else
      return stream << "Tensor[" << obj.dtype() << "]:" << obj.name() << obj.shape().toString();
  }
}

#endif //TENSORSCRIPT_TENSOR_HPP
