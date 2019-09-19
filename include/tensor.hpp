#ifndef TENSORSCRIPT_TENSOR_HPP
#define TENSORSCRIPT_TENSOR_HPP

#include <memory>
#include <vector>
#include <sstream>
#include <assert.h>

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
		 * 数据类型
		 * */
		enum DataType {
				i8, i16, i32, i64, u8, u16, u32, u64, f16, f32, f64
		};

		/**
		 * 获取元素大小
		 * */
		size_t getElementSize(DataType dtype) {
			switch (dtype) {
				case i8:
				case u8:
					return 1;
				case i16:
				case u16:
				case f16:
					return 2;
				case i32:
				case u32:
				case f32:
					return 4;
				case i64:
				case u64:
				case f64:
					return 8;
				default:
					assert(false);
			}
		}

		/**
		 * 张量的形状
		 * */
		class TensorShape {
		public:
				TensorShape() : _dims({}) {}

				TensorShape(const initializer_list<int> &list) : _dims(list) {}

				inline int size() const {
					int total = 1;
					for (auto i:_dims)total *= i;
					return total;
				}

				inline int ndim() const { return _dims.size(); }

				inline int operator()(int index) const { return _dims[index]; }

				int index(const TensorShape &index) const {
					int ind = 0;
					for (int i = 0; i < _dims.size(); i++) {
						int step = index(i);
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
				vector<int> _dims;
		};

		ostream &operator<<(ostream &stream, const TensorShape &obj) {
			return stream << obj.toString();
		}

		/**
		 * 张量
		 * */
		class Tensor {
		public:
				Tensor(const TensorShape &shape, DataType dtype, string name)
								: _shape(shape), _dtype(dtype), _name(move(name)) {
					_data = make_shared<void>(malloc(shape.size() * getElementSize(dtype)), [](void *p) { free(p); });
				}

				Tensor(const TensorShape &shape, DataType dtype)
								: _shape(shape), _dtype(dtype) {
					_data = make_shared<void>(malloc(shape.size() * getElementSize(dtype)), [](void *p) { free(p); });
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

				//Tensor asType(DataType data_type);

				template<typename T>
				T at(const TensorShape &index) {
					auto ind = _shape.index(index);
					auto data = static_cast<T *> (_data.get());
					return data[ind];
				}

				template<typename T>
				T &at(const TensorShape &index) {
					auto ind = _shape.index(index);
					auto data = static_cast<T *> (_data.get());
					return data[ind];
				}

		public:
				template<typename T>
				inline T *data() const { return (T *) _data.get(); }

				inline const string &name() const { return _name; }

				inline const TensorShape shape() const { return _shape; }

				inline DataType dtype() const { return _dtype; }

		private:
				string _name;
				TensorShape _shape;
				shared_ptr<void> _data;      // 允许多个张量共享同一块内存，例如：切片、广播
				DataType _dtype;
		};

		ostream &operator<<(ostream &stream, const Tensor &obj) {
			if (obj.name().empty())
				return stream << "Tensor" << obj.shape().toString();
			else
				return stream << "Tensor:" << obj.name() << obj.shape().toString();
		}
}

#endif //TENSORSCRIPT_TENSOR_HPP
