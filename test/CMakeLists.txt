include_directories(include)

add_executable(test_tensor tensor.cpp)
target_link_libraries(test_tensor TensorScript)

add_executable(test_math math.cpp)
target_link_libraries(test_math TensorScript)

enable_testing()
add_test(NAME test_tensor COMMAND test_tensor)
add_test(NAME test_math COMMAND test_math)
