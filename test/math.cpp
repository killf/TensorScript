#include <iostream>
#include <stdlib.h>
#include <cstdlib>
#include <cmath>

template<typename T>
void func(T a) {
    if (std::is_same<T, int>::value) {
        printf("123123\n");
    }
}


int main(int argc, char *argv[]) {
    func(1);

    return 0;
}