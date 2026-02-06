cmake_minimum_required(VERSION 3.9.0)
project(htkrecorder LANGUAGES CXX)

add_executable(htkrecorder main.cpp recorder.cpp)

find_package(k4a CONFIG REQUIRED)

target_link_libraries(htkrecorder PRIVATE k4a::k4a)

find_library(K4ARECORD_LIB NAMES k4arecord REQUIRED)
target_link_libraries(htkrecorder PRIVATE ${K4ARECORD_LIB})

# Setup install
include(GNUInstallDirs)

install(TARGETS htkrecorder RUNTIME DESTINATION bin)