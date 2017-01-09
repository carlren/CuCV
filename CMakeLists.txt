cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

# modules
add_subdirectory(CuCV)

# Dummy
add_executable(playground Playground.cpp)
target_link_libraries(playground
    CuCv)
