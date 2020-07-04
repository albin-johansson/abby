cmake_minimum_required(VERSION 3.15)
project(abby)

# Creates an interface library target for a header-only library.
#   name: the name of the library target.
#   includeDirectory: the path of the directory that contains the headers of the library.
#   sources: the headers associated with the library.
function(add_header_only_lib [name [includeDirectory [sources]]])
    add_library(${ARGV0} INTERFACE)
    target_sources(${ARGV0} INTERFACE ${ARGV2})
    target_include_directories(${ARGV0} SYSTEM INTERFACE ${ARGV1})
endfunction()

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

# Helper variables for directory paths
set(ROOT_DIR ${PROJECT_SOURCE_DIR})
set(LIBRARY_DIR ${ROOT_DIR}/lib)
set(INCLUDE_DIR ${ROOT_DIR}/include)

add_header_only_lib(libEnTT ${LIBRARY_DIR}/entt ${LIBRARY_DIR}/entt/entt.hpp)

set(BOOST_SOURCES
        ${LIBRARY_DIR}/boost/assert.hpp
        ${LIBRARY_DIR}/boost/config.hpp
        ${LIBRARY_DIR}/boost/current_function.hpp)
add_header_only_lib(libBOOST ${LIBRARY_DIR}/boost ${BOOST_SOURCES})

# The various targets used
set(ABBY_LIB_TARGET abby)
set(ABBY_TEST_TARGET abby-test)

set(SOURCE_FILES
        include/abby.hpp)

add_library(${ABBY_LIB_TARGET} INTERFACE)

add_subdirectory(test)