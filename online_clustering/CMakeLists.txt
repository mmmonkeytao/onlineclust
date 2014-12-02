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

# Make sure stuff goes in to the correct places in the build dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Point CMake to the directory containing various build scripts
set(CMakeScripts_MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${CMakeScripts_MODULE_DIR})

# set compiler flags for c++
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall") 
#-Werror -Wfatal-errors")

# Bring in helper functions to make dealing with cache internal variables easier
include(CacheInternalHelpers)

# QT version
set(DESIRED_QT_VERSION 4)

#######  Package Use ########
OPTION(USE_OPENCV "Using OpenCV Library." ON) 
if(${USE_OPENCV})
find_package(OpenCV REQUIRED)
endif()

OPTION(USE_EIGEN "Using Eigen3 Library." ON)
if(USE_EIGEN)
find_package(Eigen3 REQUIRED)
endif()


# We want to create dynamic libraries
set(BUILD_SHARED_LIBS true)

# options
option(USE_VTK "Use the VTK library for 3D visualization"  ON)
option(USE_PCL "Use the PCL library for Point Clouds Processing" ON)

# Find VTK if necessary
#if(USE_VTK)	
#endif()

# Turn on all compiler warnings
include(EnableAllWarnings)

# Enable Doxygen build with 'make doxygen'
message(STATUS "Using Doxygen...")
include(UseDoxygen)

set(ONLINECLUST_LIBRARIES
  CACHE INTERNAL
  "All individual libraries that go into the global one"
)

# add sub-directories
#add_subdirectory(3d_point_cloud)
add_subdirectory(data_proc)
add_subdirectory(hmp)

#add_subdirectory(config)
add_subdirectory(test)
#add_subdirectory(app)

# Empty the export targets list
set(ONLINECLUST_TARGETS
  CACHE INTERNAL
  "Targets which should be exported"
)

mark_as_advanced(BUILD_Tests)

#########################
# CONFIG SCRIPT CREATION
#########################

# Provide a cmake config script for the build directory
if (ONLINECLUST_LIBRARIES)
  list(REMOVE_DUPLICATES ONLINECLUST_LIBRARIES)
  export(TARGETS ${ONLINECLUST_LIBRARIES} FILE UseONLINECLUST.cmake)
endif ()

#configure_file( "${PROJECT_SOURCE_DIR}/config/onlineclustConfig.cmake.in" "${PROJECT_BINARY_DIR}/onlineclustConfig.cmake" @ONLY )

# Stick build location in the central CMake user package registry, so that it may be easily found
if ("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" GREATER 2.8)
  export(PACKAGE ONLINECLUST)
endif()


###############
# INSTALLATION
###############

# Note: The ONLINE3D library target handles its own installation

# Create configuration files that external projects may use to include ONLINE3D
# This will pick up stuff from sub-directories that export targets with the
# 'ONLINE3DExport' tag
#if (ONLINECLUST_TARGETS)
#  install(EXPORT ONLINECLUSTExport
#          FILE UseONLINECLUST.cmake
#          DESTINATION lib/${ONLINECLUST_NAMESPACE}
#  )
#endif ()
#install (TARGETS /bin/ RUNTIME DESTINATION  ${CMAKE_SOURCE_DIR}/test/exe/)
# We're going to reuse the ONLINE3DConfig.cmake.in file from before, so we'll
# need to override the includes path to point at the right place in the install
# location
# TODO : This needs fixing to include paths to dependencies too.
#set(ONLINECLUST_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/include/${ONLINECLUST_NAMESPACE})
#set(ONLINECLUST_LIB_DIR ${CMAKE_INSTALL_PREFIX}/lib)
#configure_file( "${PROJECT_SOURCE_DIR}/config/onlineclustConfig.cmake.in" "${PROJECT_BINARY_DIR}/onlineclustConfig.cmake.install" @ONLY )

#install(FILES "${PROJECT_BINARY_DIR}/onlineclustConfig.cmake.install"
#        DESTINATION lib/${ONLINECLUST_NAMESPACE}
#        RENAME onlineclustConfig.cmake
#)

#add_executable(${PROJECT_NAME} main_visRangeImage.cpp ${HEADER_FILES})
install (PROGRAMS "${CMAKE_BINARY_DIR}/bin/" COMPONENT RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/test/exe/")


