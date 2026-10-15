add_library(models_api INTERFACE)
target_include_directories(models_api INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(models_api INTERFACE dataset)

add_library(core
    common/distance.cpp
    knn/model.cpp
)
target_link_libraries(core PUBLIC models_api)