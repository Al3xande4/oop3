set(COMMON_TESTS
    common/test_split_finder
    common/test_weighted_criteria
)

set(STUMP_TESTS
    stump/test_stump_edge
    stump/test_stump_exceptions
    stump/test_stump_naive
    stump/test_stump_basic
)

set(CART_TESTS
    cart/test_cart_basic
    cart/test_cart_growth
    cart/test_cart_thresholds
    cart/test_cart_classifier
    cart/test_cart_classifier_labels
    cart/test_cart_features_subset
    cart/test_cart_edge
    cart/test_cart_layout
    cart/test_cart_exceptions
)

set(FOREST_TESTS
    forest/test_bootstrap_sampler
    forest/test_forest_exceptions
    forest/test_forest_equivalence
    forest/test_forest_classifier
    forest/test_forest_parallel
)

foreach(test_path IN LISTS COMMON_TESTS STUMP_TESTS CART_TESTS FOREST_TESTS)
    get_filename_component(test_name ${test_path} NAME)
    add_executable(${test_name} ${test_path}.cpp)
    target_link_libraries(${test_name} PRIVATE core test_support)
    target_include_directories(${test_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/common)
    target_compile_features(${test_name} PRIVATE cxx_std_20)
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach()
