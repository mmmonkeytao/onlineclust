CMAKE_MINIMUM_REQUIRED(VERSION 2.8)

# --------------------------------------------------------------
# Indicate CMake 2.7 and above that we don't want to mix relative
#  and absolute paths in linker lib lists.
# Run "cmake --help-policy CMP0003" for more information.
# --------------------------------------------------------------
if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

project(GPLIB)
 
# Point CMake to the directory containing various build scripts
set(CMakeScripts_MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_MODULE_PATH ${CMakeScripts_MODULE_DIR})

include(CacheInternalHelpers)

find_package(GSL)
find_library(GSL_LIBRARY gsl)
find_library(GSLCBLAS_LIBRARY gslcblas)
include_directories(${GSL_INCLUDE_DIR})

find_package(BOOST REQUIRED)
set(BOOST_INCLUDE_DIR "/usr/local/Cellar/boost/1.56.0/include/")
include_directories(${BOOST_INCLUDE_DIR})

find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in 
		      ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(doc ALL ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
		               WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
			       COMMENT "Generating API documentation with Doxygen" VERBATIM)
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/doc)
endif(DOXYGEN_FOUND)

set(SOURCES
	src/GP_Exception.cc src/GP_InputParams.cc 
	src/GP_Vector.cc src/GP_Matrix.cc src/GP_Tensor.cc src/GP_Optimizer.cc src/GP_OptimizerCG.cc
	src/GP_SigmoidFunction.cc src/GP_Evaluation.cc src/GP_Histogram.cc)

set(HEADER_SUBDIR include/GPlib)

set(PUBLIC_HEADERS
	${HEADER_SUBDIR}/GP_Constants.hh
	${HEADER_SUBDIR}/GP_InputParams.hh
	${HEADER_SUBDIR}/GP_Vector.hh
	${HEADER_SUBDIR}/GP_Matrix.hh
	${HEADER_SUBDIR}/GP_DataSet.hh
	${HEADER_SUBDIR}/GP_DataReader.hh
	${HEADER_SUBDIR}/GP_UniversalDataReader.hh
	${HEADER_SUBDIR}/GP_Exception.hh
	${HEADER_SUBDIR}/GP_Base.hh	
	${HEADER_SUBDIR}/GP_CovarianceFunction.hh	
	${HEADER_SUBDIR}/GP_SquaredExponentialWght.hh	
	${HEADER_SUBDIR}/GP_SigmoidFunction.hh	
	${HEADER_SUBDIR}/GP_Evaluation.hh	
	${HEADER_SUBDIR}/GP_Histogram.hh	
	${HEADER_SUBDIR}/GP_ObjectiveFunction.hh	
	${HEADER_SUBDIR}/GP_Optimizer.hh	
	${HEADER_SUBDIR}/GP_OptimizerCG.hh	
	${HEADER_SUBDIR}/GP_Base.hh	
	${HEADER_SUBDIR}/GP_Classification.hh	
	${HEADER_SUBDIR}/GP_BinaryClassification.hh	
	${HEADER_SUBDIR}/GP_BinaryClassificationEP.hh	
	${HEADER_SUBDIR}/GP_BinaryClassificationIVM.hh	
	${HEADER_SUBDIR}/GP_PolyadicClassification.hh	
	${HEADER_SUBDIR}/GP_PolyadicClassificationEP.hh	
	${HEADER_SUBDIR}/GP_PolyadicClassificationIVM.hh	
	${HEADER_SUBDIR}/GP_Regression.hh	
	${HEADER_SUBDIR}/GP_SparseRegression.hh	
	${HEADER_SUBDIR}/SVM_Binary.hh	
	${HEADER_SUBDIR}/KernelPCA.hh	
)

cache_internal_init_unique(GPLIB_INCLUDE 
	${GSL_INCLUDE} ${CMAKE_CURRENT_SOURCE_DIR}/include)

include_directories(${GPLIB_INCLUDE})

#add_library(gplib SHARED ${SOURCES} ${PUBLIC_HEADERS} ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
add_library(gplib ${SOURCES} ${PUBLIC_HEADERS} ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
target_link_libraries(gplib ${GSL_LIBRARY})


add_executable(GP_make_data test/gp_make_data.cc ${CONF_FILE})
target_link_libraries(GP_make_data gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_active_learning test/gp_active_learning.cc ${CONF_FILE})
target_link_libraries(GP_active_learning gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_train test/gp_train.cc)
target_link_libraries(GP_train gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_train_multi test/gp_train_multi.cc)
target_link_libraries(GP_train_multi gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_predict test/gp_predict.cc)
target_link_libraries(GP_predict gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_predict_multi test/gp_predict_multi.cc)
target_link_libraries(GP_predict_multi gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_regression_train test/gp_regression_train.cc)
target_link_libraries(GP_regression_train gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_eptest_wght test/gp_eptest_wght.cc)
target_link_libraries(GP_eptest_wght gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(IVM_wght_test test/ivm_wght_test.cc)
target_link_libraries(IVM_wght_test gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_reg_wght_test test/regr_wght_test.cc)
target_link_libraries(GP_reg_wght_test gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_eptest test/gp_eptest.cc)
target_link_libraries(GP_eptest gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})

#add_executable(GP_epmulti test/gp_epmulti.cc)
#target_link_libraries(GP_epmulti gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_ivmtest test/gp_ivmtest.cc)
target_link_libraries(GP_ivmtest gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_ivm_multi test/gp_ivm_multi.cc)
target_link_libraries(GP_ivm_multi gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_dense_multi test/gp_dense_multi.cc)
target_link_libraries(GP_dense_multi gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_ivminctest test/gp_ivminctest.cc)
target_link_libraries(GP_ivminctest gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_mitest test/gp_mitest.cc)
target_link_libraries(GP_mitest gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
#add_executable(GP_matrixtest test/gp_matrixtest.cc)
#target_link_libraries(GP_matrixtest gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(GP_simple test/gp_simpleclassf.cc)
target_link_libraries(GP_simple gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(kPCA test/kPCA.cc)
target_link_libraries(kPCA gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 
add_executable(BayesOpt test/bayes_opt.cc)
target_link_libraries(BayesOpt gplib ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})
 

install(FILES ${PUBLIC_HEADERS} DESTINATION ${HEADER_SUBDIR})
install(TARGETS GP_train GP_predict GP_active_learning RUNTIME DESTINATION bin)
install(TARGETS gplib ARCHIVE DESTINATION lib)
#install(TARGETS gplib LIBRARY DESTINATION lib)


export(TARGETS gplib FILE UseGPlib.cmake)

configure_file( "cmake/GPlibConfig.cmake.in" "${PROJECT_BINARY_DIR}/GPlibConfig.cmake" @ONLY )

# Stick build location in the central CMake user package registry, so that it may be easily found
if ("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" GREATER 2.7)
  export(PACKAGE GPlib)
endif()

