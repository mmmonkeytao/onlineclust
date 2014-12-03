# Point CMake to the directory containing various build scripts
#set(CMakeScripts_MODULE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
#set(CMAKE_MODULE_PATH ${CMakeScripts_MODULE_DIR})

#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN_INCLUDE_DIR})

#find_package(GPlib REQUIRED)
#include_directories(${GPlib_DIR}/../include)
#link_directories(${GPlib_DIR})

#set(HEADER_SUBDIR include/OSC)

#list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../gplib/build/")
find_package(GPlib REQUIRED)
#include_directories(${GPlib_DIR}/../include)
#link_directories(${GPlibDIR})

set(SOURCES src/Vertex.cpp 
            src/Mesh.cpp 
	    src/Cluster.cpp 
	    src/ClusterList.cpp 
	    src/OnlineStarClustering.cpp)

set(HEADER_SUBDIR "include")

set(PUBLIC_HEADERS ${HEADER_SUBDIR}/Vertex.h
		   ${HEADER_SUBDIR}/Mesh.h
		   ${HEADER_SUBDIR}/Cluster.h
		   ${HEADER_SUBDIR}/ClusterList.h
		   ${HEADER_SUBDIR}/OnlineStarClustering.h
		   ${HEADER_SUBDIR}/KernelOSC.h)

set(ONLINECLUST_CLUST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ONLINECLUST_CLUST_LIBRARY online_clust_clust)

include_directories(${ONLINECLUST_CLUST_INCLUDE})

cache_internal_init_unique(ONLINECLUST_CLUST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
cache_internal_init_unique(ONLINECLUST_CLUST_LIBRARY online_clust_clust)

add_library(${ONLINECLUST_CLUST_LIBRARY} ${SOURCES} ${PUBLIC_HEADERS})
target_link_libraries(${ONLINECLUST_CLUST_LIBRARY} ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})

cache_internal_append_unique(ONLINECLUST_LIBRARIES ${ONLINECLUST_CLUST_LIBRARY})


#include_directories(${OSC_INCLUDE})

#add_library(osc ${SOURCES} ${PUBLIC_HEADERS} ${CUDA_HEADERS} ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})

#add_executable(osctest test/osctest.cc)
#target_link_libraries(osctest osc ${STDLIB}) 

#add_executable(cluster_tracks test/cluster_tracks.cc)
#target_link_libraries(cluster_tracks osc ${STDLIB})

#install(FILES ${PUBLIC_HEADERS} DESTINATION ${HEADER_SUBDIR})
