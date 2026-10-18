add_subdirectory(dataset)
add_subdirectory(models)

add_library(cli STATIC cli.cpp)
target_include_directories(cli PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(ds_load load.cpp)
add_executable(knn main.cpp)

target_link_libraries(ds_load PRIVATE dataset cli)
target_link_libraries(knn PRIVATE dataset core cli)
