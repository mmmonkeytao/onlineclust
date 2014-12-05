# Install script for directory: /home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib

# Set the install prefix
IF(NOT DEFINED CMAKE_INSTALL_PREFIX)
  SET(CMAKE_INSTALL_PREFIX "/usr/local")
ENDIF(NOT DEFINED CMAKE_INSTALL_PREFIX)
STRING(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
IF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  IF(BUILD_TYPE)
    STRING(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  ELSE(BUILD_TYPE)
    SET(CMAKE_INSTALL_CONFIG_NAME "")
  ENDIF(BUILD_TYPE)
  MESSAGE(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
ENDIF(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)

# Set the component getting installed.
IF(NOT CMAKE_INSTALL_COMPONENT)
  IF(COMPONENT)
    MESSAGE(STATUS "Install component: \"${COMPONENT}\"")
    SET(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  ELSE(COMPONENT)
    SET(CMAKE_INSTALL_COMPONENT)
  ENDIF(COMPONENT)
ENDIF(NOT CMAKE_INSTALL_COMPONENT)

# Install shared libraries without execute permission?
IF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  SET(CMAKE_INSTALL_SO_NO_EXE "1")
ENDIF(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/GPlib" TYPE FILE FILES
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Constants.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_InputParams.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Vector.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Matrix.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_DataSet.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_DataReader.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_UniversalDataReader.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Exception.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Base.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_CovarianceFunction.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_SquaredExponentialWght.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_SigmoidFunction.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Evaluation.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Histogram.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_ObjectiveFunction.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Optimizer.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_OptimizerCG.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Base.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Classification.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_BinaryClassification.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_BinaryClassificationEP.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_BinaryClassificationIVM.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_PolyadicClassification.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_PolyadicClassificationEP.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_PolyadicClassificationIVM.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_Regression.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/GP_SparseRegression.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/SVM_Binary.hh"
    "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/include/GPlib/KernelPCA.hh"
    )
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_train" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_train")
    FILE(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_train"
         RPATH "")
  ENDIF()
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/build/GP_train")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_train" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_train")
    FILE(RPATH_REMOVE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_train")
    IF(CMAKE_INSTALL_DO_STRIP)
      EXECUTE_PROCESS(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_train")
    ENDIF(CMAKE_INSTALL_DO_STRIP)
  ENDIF()
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_predict" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_predict")
    FILE(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_predict"
         RPATH "")
  ENDIF()
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/build/GP_predict")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_predict" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_predict")
    FILE(RPATH_REMOVE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_predict")
    IF(CMAKE_INSTALL_DO_STRIP)
      EXECUTE_PROCESS(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_predict")
    ENDIF(CMAKE_INSTALL_DO_STRIP)
  ENDIF()
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_active_learning" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_active_learning")
    FILE(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_active_learning"
         RPATH "")
  ENDIF()
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE EXECUTABLE FILES "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/build/GP_active_learning")
  IF(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_active_learning" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_active_learning")
    FILE(RPATH_REMOVE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_active_learning")
    IF(CMAKE_INSTALL_DO_STRIP)
      EXECUTE_PROCESS(COMMAND "/usr/bin/strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/bin/GP_active_learning")
    ENDIF(CMAKE_INSTALL_DO_STRIP)
  ENDIF()
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")
  FILE(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE STATIC_LIBRARY FILES "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/build/libgplib.a")
ENDIF(NOT CMAKE_INSTALL_COMPONENT OR "${CMAKE_INSTALL_COMPONENT}" STREQUAL "Unspecified")

IF(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
ELSE(CMAKE_INSTALL_COMPONENT)
  SET(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
ENDIF(CMAKE_INSTALL_COMPONENT)

FILE(WRITE "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/build/${CMAKE_INSTALL_MANIFEST}" "")
FOREACH(file ${CMAKE_INSTALL_MANIFEST_FILES})
  FILE(APPEND "/home/jimmy/workspace/thesis/thesis_code/online_clustering/gplib/build/${CMAKE_INSTALL_MANIFEST}" "${file}\n")
ENDFOREACH(file)
