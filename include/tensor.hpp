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

				T operator()(const TensorShape &index) const {
					auto ind = _shape.index(index);
					return data()[ind];
				}

				T &operator()(const TensorShape &index) {
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
