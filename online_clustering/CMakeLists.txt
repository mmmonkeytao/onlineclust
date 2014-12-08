cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
#if(COMMAND cmake_policy)
cmake_policy(SET CMP0042 NEW)
#endif(COMMAND cmake_policy)

project(onlineclust)

set(ONLINECLUST_NAMESPACE onlineclust)

# Make sure stuff goes in to the correct places in the build dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Point CMake to the directory containing various build scripts
set(CMakeScripts_MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
list(APPEND CMAKE_MODULE_PATH "${CMakeScripts_MODULE_DIR}")

# set compiler flags for c++
#set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CXX_FLAGS "-std=c++11 -Wall -g -O3 ")

#-Werror -Wfatal-errors")

# Bring in helper functions to make dealing with cache internal variables easier
include(CacheInternalHelpers)

#######  Package Use ########
# QT version
set(DESIRED_QT_VERSION 4)

########

####### OpenCV ##########
#OPTION(USE_OPENCV "Using OpenCV Library." ON)
#set(OPENCV_DIR "/usr/local/Cellar/opencv/2.4.9/share/OpenCV/")
#if(${USE_OPENCV})
list(APPEND CMAKE_MODULE_PATH "/usr/local/Cellar/opencv/2.4.9/share/OpenCV")
find_package(OpenCV 2.4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

####### Eigen
#OPTION(USE_EIGEN "Using Eigen3 Library." ON)
set(EIGEN_INCLUDE_DIR "/usr/local/Cellar/eigen/3.2.2/include/")
include_directories(${EIGEN_INCLUDE_DIR})


# use gsl lib
#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/gplib/cmake")
#find_package(GSL REQUIRED)
set(GSL_INCLUD_DIR "/usr/local/Cellar/gsl/1.16")
include_directories(${GSL_INCLUDE_DIR})
#link_directories(${GSL_LIBRARY_DIRS})

# options
#option(USE_VTK "Use the VTK library for 3D visualization"  ON)
#option(USE_PCL "Use the PCL library for Point Clouds Processing" ON)

# include pcl directories
#list(APPEND CMAKE_MODULE_PATH "/usr/local/Cellar/pcl/1.7.2/share/pcl-1.7/")
list(APPEND CMAKE_MODULE_PATH "/usr/local/Cellar/pcl/1.7.2/share/pcl-1.7/")
find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

# Find VTK if necessary
list(APPEND CMAKE_MODULE_PATH "/usr/local/Cellar/vtk/6.1.0/lib/cmake/vtk-6.1/")
find_package(VTK 6.1 REQUIRED)
include(${VTK_USE_FILE})
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
add_subdirectory(pcloud)
add_subdirectory(data_proc)
add_subdirectory(hmp)
add_subdirectory(test)
add_subdirectory(clustering)

#add_subdirectory(config)
#add_subdirectory(app)

# We want to create dynamic libraries
set(BUILD_SHARED_LIBS true)

# Empty the export targets list
set(ONLINECLUST_TARGETS
  CACHE INTERNAL
  "Targets which should be exported"
)

#mark_as_advanced(BUILD_Tests)

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
#install (PROGRAMS "${CMAKE_BINARY_DIR}/bin/" COMPONENT RUNTIME DESTINATION "${C#MAKE_SOURCE_DIR}/test/exe/")

install(PROGRAMS "${CMAKE_BINARY_DIR}/bin/dataproc_test" COMPONENT RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/app/dataproc/")

install(PROGRAMS "${CMAKE_BINARY_DIR}/bin/hmp_test" COMPONENT RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/app/hmp/")

install(PROGRAMS "${CMAKE_BINARY_DIR}/bin/omp_test" COMPONENT RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/app/hmp/")

install(PROGRAMS "${CMAKE_BINARY_DIR}/bin/pcloud_vis_test" COMPONENT RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/app/pcloud/")

install(PROGRAMS "${CMAKE_BINARY_DIR}/bin/pcloud_2rgbd_test" COMPONENT RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/app/pcloud/")

install(PROGRAMS "${CMAKE_BINARY_DIR}/bin/clust_test" COMPONENT RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/app/clustering/")

install(PROGRAMS "${CMAKE_BINARY_DIR}/bin/sys" COMPONENT RUNTIME DESTINATION "${CMAKE_SOURCE_DIR}/app/sys/")

