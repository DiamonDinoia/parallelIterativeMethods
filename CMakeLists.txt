cmake_minimum_required(VERSION 3.0)
project(parallelIterative)

set(CMAKE_CXX_STANDARD 17)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /arch:AVX /arch:SSE4 /arch:MMX")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -static")
set(CMAKE_CXX_FLAGS_DEBUG "-g")

include_directories(SYSTEM lib/Eigen)
link_directories(lib/Eigen)

add_definitions(-DSPARSE)
#add_definitions(-DDENSE)

find_package(OpenMP)
if (OPENMP_FOUND)
    message("OPEMMP FOUND")
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()


file(GLOB SOURCE_FILES "src/*.cpp" "src/*.h*")


add_executable(parallelIterative ${SOURCE_FILES})

target_link_libraries(parallelIterative)