# Install script for directory: /Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/GPlib" TYPE FILE FILES
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Constants.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_InputParams.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Vector.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Matrix.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_DataSet.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_DataReader.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_UniversalDataReader.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Exception.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Base.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_CovarianceFunction.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_SquaredExponentialWght.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_SigmoidFunction.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Evaluation.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Histogram.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_ObjectiveFunction.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Optimizer.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_OptimizerCG.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Base.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Classification.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_BinaryClassification.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_BinaryClassificationEP.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_BinaryClassificationIVM.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_PolyadicClassification.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_PolyadicClassificationEP.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_PolyadicClassificationIVM.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Regression.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_SparseRegression.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/SVM_Binary.hh"
    "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/KernelPCA.hh"
    )
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/build/GP_train")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_train" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_train")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_train")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/build/GP_predict")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_predict" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_predict")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_predict")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/build/GP_active_learning")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_active_learning" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_active_learning")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_active_learning")
    endif()
  endif()
endif()

if(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/build/libgplib.a")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgplib.a" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgplib.a")
    execute_process(COMMAND "/usr/bin/ranlib" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libgplib.a")
  endif()
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

file(WRITE "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/build/${CMAKE_INSTALL_MANIFEST}" "")
foreach(file ${CMAKE_INSTALL_MANIFEST_FILES})
  file(APPEND "/Users/taoyeandy/Documents/workspace/thesis/thesis_code/online_clustering/gplib/build/${CMAKE_INSTALL_MANIFEST}" "${file}\n")
endforeach()
