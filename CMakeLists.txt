cmake_minimum_required(VERSION 3.0)
project(parallelIterative)

set(CMAKE_CXX_STANDARD 17)

include_directories(SYSTEM lib/Eigen)
link_directories(lib/Eigen)


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()



set(SOURCE_FILES src/main.cpp src/denseBlocksJacobi.h src/utils.h src/jacobi.h src/denseOverlappingJacobi.h)
add_executable(parallelIterative ${SOURCE_FILES})