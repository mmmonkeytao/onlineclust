set(SOURCES src/data_proc.cpp)

set(HEADER_SUBDIR "include")

set(PUBLIC_HEADERS ${HEADER_SUBDIR}/data_proc.h)

# append include files
#list(APPEND GLOBAL_INC_PATH ${PUBLIC_HEADERS})

set(ONLINECLUST_DATAPROC_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(ONLINECLUST_DATAPROC_LIBRARY online_clust_dataproc)

include_directories(${ONLINECLUST_DATAPROC_INCLUDE})

cache_internal_init_unique(ONLINECLUST_DATAPROC_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/include)
cache_internal_init_unique(ONLINECLUST_DATAPROC_LIBRARY online_clust_dataproc)

add_library(${ONLINECLUST_DATAPROC_LIBRARY} ${SOURCES} ${PUBLIC_HEADERS})
target_link_libraries(${ONLINECLUST_DATAPROC_LIBRARY} ${OpenCV_LIBS})

cache_internal_append_unique(ONLINECLUST_LIBRARIES ${ONLINECLUST_DATAPROC_LIBRARY})

#install(FILES ${PUBLIC_HEADERS} DESTINATION ${HEADER_SUBDIR})
#install(TARGETS ${ONLINECLUST_DATAPROC_LIBRARY} DESTINATION lib)
