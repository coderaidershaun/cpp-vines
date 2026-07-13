cmake_minimum_required(VERSION 3.23)

project(cpp-vines LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(Boost REQUIRED)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(cpp-vines INTERFACE)
add_library(cpp-vines::cpp-vines ALIAS cpp-vines)

target_compile_features(cpp-vines INTERFACE cxx_std_23)
target_link_libraries(cpp-vines INTERFACE Boost::headers)

target_include_directories(cpp-vines
    INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

file(GLOB_RECURSE CPP_VINES_HEADERS CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
)

target_sources(cpp-vines
    INTERFACE
        FILE_SET HEADERS
        BASE_DIRS include
        FILES ${CPP_VINES_HEADERS}
)

add_executable(copulas tests/copulas.cpp)
target_link_libraries(copulas PRIVATE cpp-vines::cpp-vines)

add_executable(csv tests/csv.cpp)
target_link_libraries(csv PRIVATE cpp-vines::cpp-vines)

add_executable(ecdf tests/ecdf.cpp)
target_link_libraries(ecdf PRIVATE cpp-vines::cpp-vines)

add_executable(vine tests/vine.cpp)
target_link_libraries(vine PRIVATE cpp-vines::cpp-vines)

add_executable(example-vine examples/vine-example.cpp)
target_link_libraries(example-vine PRIVATE cpp-vines::cpp-vines)

add_executable(optimiser tests/optimiser.cpp)
target_link_libraries(optimiser PRIVATE cpp-vines::cpp-vines)

add_executable(series tests/series.cpp)
target_link_libraries(series PRIVATE cpp-vines::cpp-vines)

add_executable(stats tests/stats.cpp)
target_link_libraries(stats PRIVATE cpp-vines::cpp-vines)
