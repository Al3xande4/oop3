add_subdirectory(dataset)
add_subdirectory(models)

add_executable(ds_load load.cpp)
add_executable(knn main.cpp)

target_link_libraries(ds_load PRIVATE dataset)
target_link_libraries(knn PRIVATE dataset core)
