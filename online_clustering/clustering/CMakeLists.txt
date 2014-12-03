find_package(GPlib REQUIRED)
#include_directories(${GPlib_DIR}/../include)

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
set(ONLINECLUST_CLUST_LIBRARY online_clust_clustering)

include_directories(${ONLINECLUST_CLUST_INCLUDE})

cache_internal_init_unique(ONLINECLUST_CLUST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
cache_internal_init_unique(ONLINECLUST_CLUST_LIBRARY online_clust_clustering)

add_library(${ONLINECLUST_CLUST_LIBRARY} ${SOURCES} ${PUBLIC_HEADERS})
target_link_libraries(${ONLINECLUST_CLUST_LIBRARY} ${GSL_LIBRARY} ${GSLCBLAS_LIBRARY})

cache_internal_append_unique(ONLINECLUST_LIBRARIES ${ONLINECLUST_CLUST_LIBRARY})

