cmake_minimum_required(VERSION 2.8.3)
project(object_find)

set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11") 
set(CMAKE_CXX_FLAHS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_BUILD_TYPE Release) 
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake_modules ${CMAKE_MODULE_PATH})

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  cv_bridge
  sensor_msgs
  image_transport
  message_filters
  pcl_ros
  tf
  )

## Generate messages in the 'msg' folder


## Generate added messages and services with any dependencies listed here


catkin_package(
  INCLUDE_DIRS src
  CATKIN_DEPENDS roscpp rospy std_msgs  cv_bridge  sensor_msgs  image_transport  message_filters  pcl_ros  tf)

###########
## Build ##
###########
find_package(OpenCV REQUIRED)
find_package( CUDA 7 REQUIRED )
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
MESSAGE( STATUS "CUDA_INCLUDE_DIRS:  " ${CUDA_INCLUDE_DIRS} )
include_directories(${OpenCV_INCLUDE_DIRS} )
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLEWF_INCLUDE_DIRS})
include_directories(${SIFT_INCLUDE_DIRS})
include_directories(${QT_INCLUDE_DIRS})
link_directories(${SIFT_LIB})
MESSAGE( STATUS "SIFT_LIB: " ${SIFT_LIB} )
link_directories(/home/michele/libraries/SiftGPU/build/src/SiftGPU)

add_definitions(-DGLEW_MX -DGLEW_STATIC)
## Specify additional locations of header files
## Your package locations should be listed before other locations
# include_directories(include)
include_directories(
  ${catkin_INCLUDE_DIRS}
)

## Declare a cpp library

## Declare a cpp executable
add_executable(object_find src/camera_tracking_sift.cpp)
add_executable (object_find_sift_cuda  src/cuda_sift.cpp src/sift.cpp)

## Add cmake target dependencies of the executable/library
## as an example, message headers may need to be generated before nodes

## Specify libraries to link a library or executable target against
target_link_libraries(object_find  ${catkin_LIBRARIES}  ${OpenCV_LIBRARIES})
target_link_libraries(object_find_sift_cuda ${catkin_LIBRARIES} ${CUDA_LIBRARIES} ${OpenCV_LIBS}  ${GLEW_LIB} ${GLFW_LIB} siftgpu)

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# install(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables and/or libraries for installation
# install(TARGETS percro_6dofhaptic percro_6dofhaptic_node
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_percro_6dofhaptic.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
