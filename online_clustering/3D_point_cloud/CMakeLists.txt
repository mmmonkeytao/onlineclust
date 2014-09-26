find_package(PCL 1.7 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(VTK REQUIRED)
include(${VTK_USE_FILE})

set(HEADER_SUBDIR include/${ONLINECLUST_NAMESPACE})


set(HEADER_FILES include/pcl_func.hpp)
set(SOURCE_FILES src/RangeImage.cpp 
                 src/helper.cpp 
		 src/segmentation.cpp 
		 src/cloud_normal.cpp 
		 src/cluster_extract.cpp)

include_directories("src/" "include/")
add_library(LIB ${SOURCE_FILES} ${HEADER_FILES})

add_executable(pc2ri "src/main.cpp" ${HEADER_FILES})
add_executable(vis_pc2ri "src/main_visRangeImage.cpp" ${HEADER_FILES})

target_link_libraries (pc2ri ${PCL_LIBRARIES} LIB)
target_link_libraries (vis_pc2ri ${PCL_LIBRARIES} LIB ${VTK_LIBRARIES}) 

