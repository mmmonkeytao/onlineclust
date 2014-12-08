set(HEADER_SUBDIR "include")

set(PUBLIC_HEADERS ${HEADER_SUBDIR}/pcloud.h)

set(SOURCES src/pcloud.cpp 
            src/pcloud_RangeImage.cpp
            src/pcloud_segment.cpp) 
#           src/segmentation.cpp 
#           src/cloud_normal.cpp 
#	   src/cluster_extract.cpp)

set(ONLINECLUST_PCLOUD_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ONLINECLUST_PCLOUD_LIBRARY online_clust_pcloud)

include_directories(${ONLINECLUST_PCLOUD_INCLUDE})

cache_internal_init_unique(ONLINECLUST_PCLOUD_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
cache_internal_init_unique(ONLINECLUST_PCLOUD_LIBRARY online_clust_pcloud)

add_library(${ONLINECLUST_PCLOUD_LIBRARY} ${SOURCES} ${PUBLIC_HEADERS})

target_link_libraries(${ONLINECLUST_PCLOUD_LIBRARY} ${PCL_LIBRARIES} ${VTK_LIBRARIES})

cache_internal_append_unique(ONLINECLUST_LIBRARIES ${ONLINECLUST_PCLOUD_LIBRARY})


#include_directories("src/" "include/")
#add_library(LIB ${SOURCE_FILES} ${HEADER_FILES})

#add_executable(pc2ri "src/main.cpp" ${HEADER_FILES})
#add_executable(vis_pc2ri "src/main_visRangeImage.cpp" ${HEADER_FILES})

#target_link_libraries (pc2ri ${PCL_LIBRARIES} LIB)
#target_link_libraries (vis_pc2ri ${PCL_LIBRARIES} LIB ${VTK_LIBRARIES}) 

