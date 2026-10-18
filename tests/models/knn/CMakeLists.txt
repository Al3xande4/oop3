set(KNN_TESTS
    test_weights
    test_knn
)

foreach(test_name IN LISTS KNN_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE core test_support)
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
