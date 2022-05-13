cmake_minimum_required(VERSION 3.18)
project(CommonBackendUtils VERSION 1.0.0 DESCRIPTION "Common utilities used by all backend libraries" LANGUAGES CXX)

option(BUILD_TESTING "Build unit tests?" OFF)
option(BUILD_SHARED_LIBS "Build shared library?" OFF)
set(CMAKE_DEBUG_POSTFIX "d" CACHE STRING "Choose debug postfix" FORCE)
set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose build type" FORCE)

#set(libsrc 	src/descriptors/ContextDescriptor.cpp
#			src/descriptors/ConvolutionDescriptor.cpp
#			src/descriptors/DropoutDescriptor.cpp
#			src/descriptors/MemoryDescriptor.cpp
#			src/descriptors/OptimizerDescriptor.cpp
#			src/descriptors/PoolingDescriptor.cpp
#			src/descriptors/TensorDescriptor.cpp
#			src/backend_utils.cpp)

#if(BUILD_SHARED_LIBS)
#  set(LIBRARY_NAME "CommonBackendUtils")
#  add_library(${LIBRARY_NAME} SHARED ${libsrc})
#  set_target_properties(${LIBRARY_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
#else()
#  set(LIBRARY_NAME "CommonBackendUtils_static")
#  add_library(${LIBRARY_NAME} STATIC ${libsrc})
#endif()

#set_target_properties(${LIBRARY_NAME} PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
#target_include_directories(${LIBRARY_NAME} PUBLIC "${PROJECT_SOURCE_DIR}/include")

install(FILES "include/Avocado/backend_defs.h" DESTINATION include/Avocado)
