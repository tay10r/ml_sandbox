cmake_minimum_required(VERSION 3.14.7)

project(sgd LANGUAGES CXX)

add_library(sgd
  sgd.h
  sgd.cpp)

target_include_directories(sgd PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
