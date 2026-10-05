add_subdirectory(dataset)

add_executable(ds_load main.cpp)

target_link_libraries(ds_load PRIVATE dataset)
