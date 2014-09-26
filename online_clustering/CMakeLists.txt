cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(onlineclust)

set(ONLINECLUST_NAMESPACE onlineclust)

# Set up our directory structure for output libraries and binaries
if(NOT CMAKE_RUNTIME_OUTPUT_DIRECTORY)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()
if(NOT CMAKE_LIBRARY_OUTPUT_DIRECTORY)
  if(UNIX)
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
  else()
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
  endif()
endif()
if(NOT CMAKE_ARCHIVE_OUTPUT_DIRECTORY)
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
endif()
mark_as_advanced(CMAKE_RUNTIME_OUTPUT_DIRECTORY
  CMAKE_LIBRARY_OUTPUT_DIRECTORY
  CMAKE_ARCHIVE_OUTPUT_DIRECTORY
  )
set(CMAKE_OUTPUT_DIR "${CMAKE_BINARY_DIR}/include")

# set compiler flags for c++
set(CMAKE_CXX_FLAGS "-std=c++11")

# Bring in helper functions to make dealing with cache internal variables easier
#include(CacheInternalHelpers)

# QT version
set(DESIRED_QT_VERSION 4)

# options
option(USE_VTK "Use the VTK library for 3D visualization"  ON)
option(USE_PCL "Use the PCL library for Point Clouds Processing" ON)

# Find VTK if necessary
#if(USE_VTK)	
#endif()

# Turn on all compiler warnings
#include(EnableAllWarnings)

# Enable Doxygen build with 'make doxygen'
#message(STATUS "Using Doxygen...")
#include(UseDoxygen)

# add sub-directories
add_subdirectory(3D_point_cloud)

#add_executable(${PROJECT_NAME} main_visRangeImage.cpp ${HEADER_FILES})
#install (TARGETS "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/*" DESTINATION ${CMAKE_SOURCE_DIR}/test)


