# CMake minimun version
cmake_minimum_required(VERSION 2.8)

# executable name
set(TARGET_NAME nppSample)

# solution name
project(${TARGET_NAME})

if(MSVC)
    add_definitions("/wd4265 /wd4266 /wd4350 /wd4365 /wd4435 /wd4514 /wd4625 /wd4626 /wd4640 /wd4668 /wd4710 /wd4819 /wd4820 /wd4946")
endif(MSVC)

# find CUDA
find_package(CUDA REQUIRED)

# executable
cuda_add_executable(${TARGET_NAME} main.cpp)

# Find OpenMP
find_package(OpenMP REQUIRED)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

target_link_libraries(${TARGET_NAME} ${CUDA_LIBRARIES} ${CUDA_npp_LIBRARY})