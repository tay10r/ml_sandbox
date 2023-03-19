cmake_minimum_required(VERSION 3.14.7)

project(nn LANGUAGES CXX)

option(NN_ENABLE_TEST "Whether or not to build the test program." ON)

add_library(nn
  nn.h
  nn.cpp)

if(CMAKE_COMPILER_IS_GNUCXX)
  target_compile_options(nn PRIVATE -Wall -Wextra -Werror -Wfatal-errors)
endif(CMAKE_COMPILER_IS_GNUCXX)

target_include_directories(nn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

if(NN_ENABLE_TEST)
  add_executable(nn_test nn_test.cpp)
  target_link_libraries(nn_test PRIVATE nn)
  add_test(NAME NNTest COMMAND $<TARGET_FILE:nn_test>)
endif(NN_ENABLE_TEST)
