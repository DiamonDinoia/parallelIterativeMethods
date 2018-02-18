cmake_minimum_required(VERSION 3.0)
project(parallelIterative)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX /arch:SSE4 /arch:MMX")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -static")
#message(${CMAKE_CXX_FLAGS})
set(CMAKE_CXX_FLAGS_DEBUG "-g")

include_directories(SYSTEM lib/Eigen)
link_directories(lib/Eigen)

add_definitions(-DSPARSE)

find_package(OpenMP)
if (OPENMP_FOUND)
    message("OPEMMP FOUND")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

#find_program(CCACHE_FOUND ccache)
#if(CCACHE_FOUND)
#    message("CMAKE FOUND")
#    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ccache)
#    SET_PROPERTY(GLOBAL PROPERTY RULE_LAUNCH_LINK ccache)
#endif(CCACHE_FOUND)

file(GLOB SOURCE_FILES "src/*.cpp" "src/*.h*")
#set(SOURCE_FILES src/main.cpp src/denseBlocksJacobi.h src/utils.h src/denseParallelJacobi.h src/denseOverlappingJacobi.h
#        src/denseAsyncJacobi src/denseAsyncBlocksJacobi.h src/denseAsyncOverlappingJacobi.h src/denseSerialJacobi.h)

add_executable(parallelIterative ${SOURCE_FILES})

target_link_libraries(parallelIterative)