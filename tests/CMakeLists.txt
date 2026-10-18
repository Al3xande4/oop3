# Shared header-only support for every test: assertions, fixtures, finish().
add_library(test_support INTERFACE)
target_include_directories(test_support INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(dataset)
add_subdirectory(models)
