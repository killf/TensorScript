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
		 * 值错误
		 * */
		class ValueError : public exception {
		public:
				explicit ValueError(const string &&msg) : _msg(msg) {}

				const char *what() const noexcept override {
					return _msg.c_str();
				}

		private:
				string _msg;
		};

		/**
		 * 自定义错误
		 * */
		class CustomError : public exception {
		public:
				explicit CustomError(const char *__restrict format, ...) {
					va_list ap;
					va_start(ap, format);

					int max_len = strlen(format) + 4096;
					auto buf = new char[max_len];
					vsnprintf(buf, max_len, format, ap);

					va_end(ap);

					_msg = string(buf);
					delete (buf);
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
				TensorShape() : _dims({}) {}

				TensorShape(const initializer_list<int> &list) : _dims(list) {
					_size = 1;
					for (auto i:_dims)_size *= i;
				}

				inline int size() const { return _size; }

				inline int ndim() const { return _dims.size(); }

				inline int operator()(int index) const { return _dims[index]; }

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

				string toString() const {
					stringstream ss;

					ss << "(";
					if (_dims.size() > 0) ss << _dims[0];
					for (int i = 1; i < _dims.size(); i++)ss << "," << _dims[i];
					ss << ")";

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
							throw CustomError("IndexError：index %d is out of bounds for axis %d with size %d", index[i], i, _dims[i]);
						}
					}
				}

		private:
				vector<int> _dims;
				int _size;
		};

		ostream &operator<<(ostream &stream, const TensorShape &obj) {
			return stream << obj.toString();
		}

		/**
		 * 张量
		 * */
		template<typename T>
		class Tensor {
		public:
				explicit Tensor(const TensorShape &shape) : _shape(shape) {
					_data.reset(new T[shape.size()]);
				}

				Tensor(string name, const TensorShape &shape) : _name(std::move(name)), _shape(shape) {
					_data.reset(new T[shape.size()]);
				}

		public:
				void reshape(const TensorShape &shape) {
					if (shape.size() != _shape.size()) {
						stringstream ss;
						ss << "ValueError: cannot reshape tensor of size " << _shape.size() << " into shape " << shape;
						throw ValueError(ss.str());
					}

					_shape = shape;
				}

				T operator()(const initializer_list<int> &index) const {
					auto ind = _shape.index(index);
					return data()[ind];
				}

				T &operator()(const initializer_list<int> &index) {
					auto ind = _shape.index(index);
					return (data()[ind]);
				}

				void fill(T value) {
					auto ptr = data();
					for (auto i = 0; i < _shape.size(); i++, ptr++) {
						*ptr = value;
					}
				}

		public:
				inline string name() const { return _name; }

				inline TensorShape shape() const { return _shape; }

				inline T *data() const { return _data.get(); }

		private:
				string _name;
				TensorShape _shape;
				shared_ptr<T> _data;
		};

		template<typename T>
		ostream &operator<<(ostream &stream, const Tensor<T> &obj) {
			if (obj.name().empty())
				return stream << "Tensor" << obj.shape().toString();
			else
				return stream << "Tensor:" << obj.name() << obj.shape().toString();
		}
}

#endif //TENSORSCRIPT_TENSOR_HPP
