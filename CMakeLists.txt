cmake_minimum_required(VERSION 3.20)
project(tt_hometask LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(TT_METAL_HOME "${CMAKE_SOURCE_DIR}")

# Common include roots
set(TT_METAL_INCLUDE_DIRS
    ${TT_METAL_HOME}
    ${TT_METAL_HOME}/tt_metal
    /usr/include/metalium-thirdparty
)

add_executable(tt_softmax
    src/main.cpp
)

target_compile_features(tt_softmax PRIVATE cxx_std_20)
target_compile_options(tt_softmax PRIVATE
    -std=gnu++20
    -Wall
    -Wextra
    -O2
)

target_include_directories(tt_softmax PRIVATE
    ${TT_METAL_INCLUDE_DIRS}
)

target_link_libraries(tt_softmax PRIVATE
    tt_metal
    device
    tt_stl
    fmt
    /usr/local/lib/libmpi_ulfm_stub.so
)
