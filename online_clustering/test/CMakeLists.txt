### Test data processing library ###
include_directories(${ONLINECLUST_DATAPROC_INCLUDE})
add_executable(dataproc_test data_proc/dataproc_test.cpp)
target_link_libraries(dataproc_test ${ONLINECLUST_DATAPROC_LIBRARY})

### Test omp library in hmp ###
include_directories(${ONLINECLUST_HMP_INCLUDE})
add_executable(omp_test hmp/omp_test.cpp)
target_link_libraries(omp_test ${ONLINECLUST_HMP_LIBRARY})

add_executable(hmp_test hmp/hmp_test.cpp)
target_link_libraries(hmp_test ${ONLINECLUST_HMP_LIBRARY}  ${ONLINECLUST_DATAPROC_LIBRARY})

### test pcloud visualize point clouds
include_directories(${ONLINECLUST_PCLOUD_INCLUDE})
add_executable(pcloud_vis_test pcloud/pcloud_vis_test.cpp)
target_link_libraries(pcloud_vis_test ${ONLINECLUST_PCLOUD_LIBRARY})

add_executable(pcloud_2rgbd_test pcloud/pcloud_2rgbd_test.cpp)
target_link_libraries(pcloud_2rgbd_test ${ONLINECLUST_PCLOUD_LIBRARY})

### test clustering ########
#message("${ONLINECLUST_CLUST_INCLUDE}")
include_directories("${ONLINECLUST_CLUST_INCLUDE}")
add_executable(clust_test clustering/clust_test.cpp)
target_link_libraries(clust_test ${ONLINECLUST_CLUST_LIBRARY})

#add_executable(cluster_tracks clustering/cluster_tracks.cpp)
#target_link_libraries(cluster_tracks ${ONLINECLUST_CLUST_LIBRARY})

### test overall system #######
#include_directories(${ONLINECLUST_HMP_INCLUDE})
add_executable(sys sys/main.cpp)
target_link_libraries(sys ${ONLINECLUST_HMP_LIBRARY})


