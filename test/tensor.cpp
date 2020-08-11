#include <iostream>
#include <stdlib.h>
#include "tensor.hpp"

using namespace TensorScript;

template<typename T>
void print_tensor(Tensor<T> *tensor) {
  for (size_t i = 0; i < tensor->size(); i++) {
    if (is_floating_point<T>::value)
      printf("%f ", tensor->data()[i]);
    else
      printf("%d ", tensor->data()[i]);
  }
  printf("\n");
}

int main(int argc, char *argv[]) {
  Tensor<int> a({3, 4}, 0);
  cout << a << endl;

  print_tensor(&a);
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 4; j++) {
      a[{i, j}] = i * 4 + j;
    }
  }
  print_tensor(&a);

  auto b = Tensor<float>::rand({3, 4});
  print_tensor(b);
  delete b;

  return 0;
}