add_subdirectory(dataset)
add_subdirectory(models)

add_library(cli STATIC cli.cpp)
target_include_directories(cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(ds_load load.cpp)
add_executable(knn knn.cpp)
add_executable(stump stump.cpp)
add_executable(cart cart.cpp)
add_executable(forest forest.cpp)
add_executable(forest_demo forest_demo.cpp)

target_link_libraries(ds_load PRIVATE dataset cli)
target_link_libraries(knn PRIVATE dataset core cli)
target_link_libraries(stump PRIVATE dataset core cli)
target_link_libraries(cart PRIVATE dataset core cli)
target_link_libraries(forest PRIVATE dataset core cli)
target_link_libraries(forest_demo PRIVATE dataset core cli)
