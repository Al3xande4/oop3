add_library(dataset
    dataset.cpp
    transformed.cpp
)

target_include_directories(dataset PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

