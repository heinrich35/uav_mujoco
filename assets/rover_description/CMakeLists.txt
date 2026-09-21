cmake_minimum_required(VERSION 3.10.2)
project(rover_description)
find_package(ament_cmake REQUIRED)

ament_package()

install(DIRECTORY meshes urdf config DESTINATION share/${PROJECT_NAME})


