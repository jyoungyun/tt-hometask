cmake_minimum_required(VERSION 3.20)
project(tt_hometask LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(tt_softmax_cpu
    src/main.cpp
)

target_compile_options(tt_softmax_cpu PRIVATE
    -Wall
    -Wextra
    -O2
)
