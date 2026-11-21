cmake_minimum_required(VERSION 3.8)
project(perception)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(crp_msgs REQUIRED)
find_package(std_msgs REQUIRED)

# ONNX Runtime (C++ release csomag kicsomagolva, pl. /opt/onnxruntime)
#   colcon build --cmake-args -DONNXRUNTIME_ROOT=/opt/onnxruntime
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime telepítési könyvtár")
find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
  HINTS ${ONNXRUNTIME_ROOT}/include
  PATH_SUFFIXES onnxruntime onnxruntime/core/session)
find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib)
if(NOT ONNXRUNTIME_INCLUDE_DIR OR NOT ONNXRUNTIME_LIBRARY)
  message(FATAL_ERROR "ONNX Runtime nem található, add meg: -DONNXRUNTIME_ROOT=<útvonal>")
endif()
get_filename_component(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LIBRARY} DIRECTORY)

include_directories(include)

add_library(pointcloud_preprocessor src/pointcloud_preprocessor.cpp)
target_compile_features(pointcloud_preprocessor PUBLIC cxx_std_14)
target_include_directories(pointcloud_preprocessor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_executable(perception_node src/perceptionNode.cpp)
target_link_libraries(perception_node pointcloud_preprocessor)

ament_target_dependencies(perception_node
  rclcpp
  sensor_msgs
  crp_msgs
  std_msgs
)
target_include_directories(perception_node PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
target_link_libraries(perception_node ${ONNXRUNTIME_LIBRARY})
set_target_properties(perception_node PROPERTIES INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}")

install(TARGETS
  perception_node
  DESTINATION lib/${PROJECT_NAME}
)

install(TARGETS
  pointcloud_preprocessor
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
  include/
  DESTINATION include
)

install(DIRECTORY
  config
  launch
  model
  DESTINATION share/${PROJECT_NAME}
)

include(CTest)
if(BUILD_TESTING)
  add_executable(test_pointcloud_preprocessor test/test_pointcloud_preprocessor.cpp)
  target_link_libraries(test_pointcloud_preprocessor pointcloud_preprocessor)
  add_test(NAME pointcloud_preprocessor COMMAND test_pointcloud_preprocessor)
endif()

ament_package()
