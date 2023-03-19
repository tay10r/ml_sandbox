cmake_minimum_required(VERSION 3.14.7)

project(training_set_generator)

include(FetchContent)

FetchContent_Declare(bvh URL "https://github.com/madmann91/bvh/archive/refs/heads/master.zip")
FetchContent_MakeAvailable(bvh)

FetchContent_Declare(glm URL "https://github.com/g-truc/glm/archive/refs/heads/master.zip")
FetchContent_MakeAvailable(glm)

add_executable(main
  main.cpp
  image.h
  image.cpp
  renderer.h
  renderer.cpp
  scene.h
  scene.cpp
  color_generator.h
  color_generator.cpp
  third_party/stb_image_write.h
  third_party/stb_image_write.c)

target_compile_definitions(main PRIVATE "MODEL_PATH=\"${CMAKE_CURRENT_SOURCE_DIR}/models\"")

target_compile_features(main PRIVATE cxx_std_20)

target_link_libraries(main PRIVATE bvh glm)
