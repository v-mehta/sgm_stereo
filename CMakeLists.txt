cmake_minimum_required(VERSION 2.8.3)
project(sgm_stereo)

## Add support for C++11, supported in ROS Kinetic and newer
add_definitions(-std=c++11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -msse4.2 -msse3 -msse2")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

find_package(OpenCV REQUIRED)
find_package(PNG REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  cv_bridge
  image_geometry
  image_transport
  message_filters
  roscpp
  sensor_msgs
  stereo_msgs
  stereo_image_proc
)

catkin_package(
  INCLUDE_DIRS include
  LIBRARIES sgm_stereo
  CATKIN_DEPENDS cv_bridge image_geometry image_transport message_filters roscpp sensor_msgs stereo_image_proc
  DEPENDS system_lib OpenMP
)

###########
## Build ##
###########

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PNG_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(sgm_stereo_node src/sgm_stereo_node.cpp src/SGMStereo.cpp)
target_compile_options(sgm_stereo_node PRIVATE ${OpenMP_FLAGS})
add_dependencies(sgm_stereo_node ${catkin_EXPORTED_TARGETS})
target_link_libraries(sgm_stereo_node ${catkin_LIBRARIES} ${PNG_LIBRARY} ${OpenCV_LIBS} ${OpenMP_LIBS})

add_library(sgm_stereo src/SGMStereo.cpp)
target_compile_options(sgm_stereo PRIVATE ${OpenMP_FLAGS})
add_dependencies(sgm_stereo ${catkin_EXPORTED_TARGETS})
target_link_libraries(sgm_stereo ${catkin_LIBRARIES} ${PNG_LIBRARY} ${OpenCV_LIBS} ${OpenMP_LIBS})
