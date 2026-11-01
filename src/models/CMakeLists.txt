add_library(models_api INTERFACE)
target_include_directories(models_api INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(models_api INTERFACE dataset)

add_library(core
    common/distance.cpp
    knn/model.cpp
    decision_trees/stump.cpp
    decision_trees/engine/criteria/variance.cpp
    decision_trees/engine/criteria/gini.cpp
    decision_trees/engine/leaf/mean_leaf.cpp
    decision_trees/engine/leaf/proba_leaf.cpp
    decision_trees/cart.cpp
    decision_trees/forest.cpp
)
target_link_libraries(core PUBLIC models_api)