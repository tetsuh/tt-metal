cmake_minimum_required(VERSION 3.22...3.30)
project(metal_example_add_2_integers_in_compute)

add_executable(metal_example_add_2_integers_in_compute)
target_sources(metal_example_add_2_integers_in_compute PRIVATE add_2_integers_in_compute.cpp)

find_package(TT-Metalium REQUIRED)
target_link_libraries(metal_example_add_2_integers_in_compute PUBLIC TT::Metalium)

# Add include directory for our custom span implementation BEFORE system includes
target_include_directories(metal_example_add_2_integers_in_compute PRIVATE BEFORE ${CMAKE_SOURCE_DIR}/../../tt_stl)
