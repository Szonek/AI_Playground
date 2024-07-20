cmake_minimum_required(VERSION 3.25)
project(AI_Playground VERSION 1.0)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED On)
set(CMAKE_CXX_EXTENSIONS Off)


option(BUILD_CUDA "Build with CUDA backen" OFF)

add_library(directml SHARED IMPORTED)
set_target_properties(directml PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/directml_content-src/bin/x64-win/DirectML.dll"
    IMPORTED_IMPLIB   "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/directml_content-src/bin/x64-win/DirectML.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/directml_content-src/include")
	
add_library(d3d12 SHARED IMPORTED)
set_target_properties(d3d12 PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/d3d12_content-src/build/native/bin/x64/D3D12Core.dll"
    IMPORTED_IMPLIB "d3d12.lib"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/d3d12_content-src/build/native/include")
	
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/thirdparty/directmlx-src)

if(BUILD_CUDA)
	add_definitions(-DBUILD_CUDA)
	# Find CUDA
	cmake_policy(SET CMP0074 NEW) # Use CUDAToolkit_ROOT as a cmake prefix.
	find_package(CUDAToolkit REQUIRED)
	set(CUDA_LIB "${CUDA_cuda_driver_LIBRARY}")
	include_directories("${CUDAToolkit_INCLUDE_DIRS}")
	get_filename_component(CUDA_HOME "${CUDAToolkit_BIN_DIR}" DIRECTORY)
	find_library(CUDADEVRT_LIB cudadevrt PATH "${CUDAToolkit_LIBRARY_DIR}")
	message(STATUS "Using CUDA_HOME: ${CUDA_HOME}")
	message(STATUS "Using CUDA_LIB:  ${CUDA_LIB}")

	# Find libNVVM and nvvm.h.
	# (Linux: nvvm/lib64, windows: nvvm/lib/x64)
	# Find the nvvm directory in the toolkit.
	find_file(LIBNVVM_HOME nvvm PATHS "$ENV{LIBNVVM_HOME}" "${CUDA_HOME}")
	message(STATUS "Using LIBNVVM_HOME: ${LIBNVVM_HOME}")

	find_library(NVVM_LIB nvvm PATHS "${LIBNVVM_HOME}/lib64" "${LIBNVVM_HOME}/lib/x64")
	find_file(NVVM_H nvvm.h PATH "${LIBNVVM_HOME}/include")
	get_filename_component(NVVM_INCLUDE_DIR ${NVVM_H} DIRECTORY)
	include_directories(${NVVM_INCLUDE_DIR})
	message(STATUS "Using libnvvm header:      ${NVVM_H}")
	message(STATUS "Using libnvvm header path: ${NVVM_INCLUDE_DIR}")
	message(STATUS "Using libnvvm library:     ${NVVM_LIB}")
endif()
add_subdirectory(AI_Playground)