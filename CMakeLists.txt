cmake_minimum_required(VERSION 3.15)
project(abby)

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

set(ROOT_DIR ${PROJECT_SOURCE_DIR})
set(INCLUDE_DIR ${ROOT_DIR}/include)

set(ABBY_LIB_TARGET abby)
set(ABBY_TEST_TARGET abby-test)

set(SOURCE_FILES
        include/abby.hpp)

add_library(${ABBY_LIB_TARGET} INTERFACE)

add_subdirectory(test)