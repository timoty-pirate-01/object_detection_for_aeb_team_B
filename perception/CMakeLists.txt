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
find_package(geometry_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

option(BUILD_ONNX_NODE "Build the ONNX-backed perception node when ONNX Runtime is installed" ON)
set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime telepítési könyvtár")

include_directories(include)

add_library(pointcloud_preprocessor src/pointcloud_preprocessor.cpp)
target_compile_features(pointcloud_preprocessor PUBLIC cxx_std_14)
target_include_directories(pointcloud_preprocessor PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

add_library(live_multisweep src/live_multisweep.cpp)
target_compile_features(live_multisweep PUBLIC cxx_std_14)
target_include_directories(live_multisweep PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)
target_link_libraries(live_multisweep pointcloud_preprocessor)
ament_target_dependencies(live_multisweep
  geometry_msgs
  rclcpp
  sensor_msgs
  tf2
  tf2_geometry_msgs
  tf2_ros
)

if(BUILD_ONNX_NODE)
  find_path(ONNXRUNTIME_INCLUDE_DIR onnxruntime_cxx_api.h
    HINTS ${ONNXRUNTIME_ROOT}/include
    PATH_SUFFIXES onnxruntime onnxruntime/core/session)
  find_library(ONNXRUNTIME_LIBRARY onnxruntime HINTS ${ONNXRUNTIME_ROOT}/lib)

  if(ONNXRUNTIME_INCLUDE_DIR AND ONNXRUNTIME_LIBRARY)
    get_filename_component(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LIBRARY} DIRECTORY)
    add_executable(perception_node src/perceptionNode.cpp)
    target_link_libraries(perception_node pointcloud_preprocessor live_multisweep)
    ament_target_dependencies(perception_node
      rclcpp
      sensor_msgs
      crp_msgs
      std_msgs
      tf2
      tf2_ros
      tf2_geometry_msgs
    )
    target_include_directories(perception_node PRIVATE ${ONNXRUNTIME_INCLUDE_DIR})
    target_link_libraries(perception_node ${ONNXRUNTIME_LIBRARY})
    set_target_properties(perception_node PROPERTIES INSTALL_RPATH "${ONNXRUNTIME_LIB_DIR}")
    install(TARGETS perception_node DESTINATION lib/${PROJECT_NAME})
  else()
    message(WARNING
      "ONNX Runtime was not found. Building preprocessing libraries and tests only.")
  endif()
endif()

install(TARGETS
  pointcloud_preprocessor
  live_multisweep
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY
  include/
  DESTINATION include
)

install(DIRECTORY
  model
  DESTINATION share/${PROJECT_NAME}
)

include(CTest)
if(BUILD_TESTING)
  add_executable(test_pointcloud_preprocessor test/test_pointcloud_preprocessor.cpp)
  target_link_libraries(test_pointcloud_preprocessor pointcloud_preprocessor)
  add_test(NAME pointcloud_preprocessor COMMAND test_pointcloud_preprocessor)

  add_executable(test_live_multisweep test/test_live_multisweep.cpp)
  target_link_libraries(test_live_multisweep live_multisweep)
  ament_target_dependencies(test_live_multisweep
    geometry_msgs
    rclcpp
    sensor_msgs
    tf2
    tf2_geometry_msgs
    tf2_ros
  )
  add_test(NAME live_multisweep COMMAND test_live_multisweep)

  add_executable(nuscenes_voxel_reference test/nuscenes_voxel_reference.cpp)
  target_link_libraries(nuscenes_voxel_reference pointcloud_preprocessor)
endif()

ament_package()
