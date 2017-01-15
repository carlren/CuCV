cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

Project(CuCVApp)

# modules
add_subdirectory(CuCV)


# Boost
find_package(Boost 1.53.0 REQUIRED COMPONENTS chrono filesystem program_options system thread)
add_library(boost INTERFACE)
target_include_directories(boost INTERFACE ${Boost_INCLUDE_DIRS})
target_link_libraries(boost INTERFACE ${Boost_LIBRARIES})

# OpenCV
find_package(OpenCV 3.0.0 REQUIRED)
add_library(opencv INTERFACE)
target_include_directories(opencv INTERFACE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(opencv INTERFACE ${OpenCV_LIBS})


# Playground, Boost and OpenCV are used as dependency here
add_executable(playground Playground.cpp)
target_link_libraries(playground
    CuCv
    opencv
    boost)
