cmake_minimum_required(VERSION 3.8)
project(yolov8_bringup)

# set(CMAKE_MODULE_PATH "/home/lee52/ros2_ws/py310/lib/python3.10/site-packages/cmake/data/share/cmake-3.26/Modules" ${CMAKE_MODULE_PATH})

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)

install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}/
)

ament_package()
