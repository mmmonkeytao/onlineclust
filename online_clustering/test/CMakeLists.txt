### Test data processing library ###
include_directories(${ONLINECLUST_DATAPROC_INCLUDE})
add_executable(${ONLINECLUST_DATAPROC_LIBRARY}_test data_proc/dataproc_test.cpp)
target_link_libraries(${ONLINECLUST_DATAPROC_LIBRARY}_test ${ONLINECLUST_DATAPROC_LIBRARY})


### Test omp library in hmp ###
include_directories(${ONLINECLUST_HMP_INCLUDE})
add_executable(omp_test hmp/omp_test.cpp)
target_link_libraries(omp_test ${ONLINECLUST_HMP_LIBRARY})

