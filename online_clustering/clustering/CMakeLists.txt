set(SOURCES src/Vertex.cpp  
            src/Cluster.cpp  
	    src/OnlineStarClustering.cpp  
	    src/ClusterList.cpp  
	    src/Mesh.cpp)

set(HEADER_SUBDIR "include")

set(PUBLIC_HEADERS  include/Vertex.h
                    include/KernelOSC.h  
		    include/Mesh.h  
		    include/Cluster.h  
		    include/OnlineStarClustering.h  
		    include/ClusterList.h)


set(ONLINECLUST_CLUST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ONLINECLUST_CLUST_LIBRARY online_clust_clustering)


include_directories(${ONLINECLUST_CLUST_INCLUDE})

cache_internal_init_unique(ONLINECLUST_CLUST_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
cache_internal_init_unique(ONLINECLUST_CLUST_LIBRARY online_clust_clustering)

add_library(${ONLINECLUST_CLUST_LIBRARY} ${SOURCES} ${PUBLIC_HEADERS})
target_link_libraries(${ONLINECLUST_CLUST_LIBRARY} ${GSL_LIBRARIES})

cache_internal_append_unique(ONLINECLUST_LIBRARIES ${ONLINECLUST_CLUST_LIBRARY})