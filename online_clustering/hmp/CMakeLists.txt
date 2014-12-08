set(SOURCES src/hmp.cpp
            src/omp.cpp
	    src/helper.cpp) 

set(HEADER_SUBDIR "include")

set(PUBLIC_HEADERS include/hmp.h)

set(ONLINECLUST_HMP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ONLINECLUST_HMP_LIBRARY online_clust_hmp)

#include_directories(EIGEN3_INCLUDE_DIR)
include_directories(${ONLINECLUST_HMP_INCLUDE})

cache_internal_init_unique(ONLINECLUST_HMP_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
cache_internal_init_unique(ONLINECLUST_HMP_LIBRARY online_clust_hmp)

add_library(${ONLINECLUST_HMP_LIBRARY} ${SOURCES} ${PUBLIC_HEADERS})
#target_link_libraries(${ONLINECLUST_HMP_LIBRARY} )

cache_internal_append_unique(ONLINECLUST_LIBRARIES ${ONLINECLUST_HMP_LIBRARY})

#install(FILES ${PUBLIC_HEADERS} DESTINATION ${HEADER_SUBDIR})
#install(TARGETS ${ONLINECLUST_HMP_LIBRARY} DESTINATION lib)
