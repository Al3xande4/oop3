set(LAB1_TESTS
    test_parse_missing
    test_feature_type
    test_quantiles
    test_encoding
    test_inconsistent_csv
)

foreach(test_name IN LISTS LAB1_TESTS)
    add_executable(${test_name} ${test_name}.cpp)
    target_link_libraries(${test_name} PRIVATE dataset)
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
