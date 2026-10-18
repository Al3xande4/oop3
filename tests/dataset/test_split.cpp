#include "test_util.hpp"
#include "dataset/transformed.hpp"

#include <filesystem>

int main()
{
    // Arrange: 10 rows x 3 features, targets 0..9
    TransformedDataset tds;
    tds.matrix = Matrix(10, 3);
    for (size_t i = 0; i < 10; i++)
        for (size_t j = 0; j < 3; j++)
            tds.matrix[i][j] = i * 10. + j;
    tds.feature_names = {"f0", "f1", "f2"};
    tds.target_name = "y";
    for (size_t i = 0; i < 10; i++)
        tds.target_values.push_back(static_cast<double>(i));

    // Act: 60/40 split
    auto [train60, test60] = tds.split(0.6);

    // Assert: sizes
    CHECK(train60.matrix.get_rows() == 6);
    CHECK(train60.matrix.get_cols() == 3);
    CHECK(test60.matrix.get_rows() == 4);
    CHECK(test60.matrix.get_cols() == 3);
    CHECK(train60.target_values.size() == 6);
    CHECK(test60.target_values.size() == 4);

    // Assert: rows and targets stay in original order
    for (size_t i = 0; i < 6; i++)
        for (size_t j = 0; j < 3; j++)
            CHECK(train60.matrix[i][j] == tds.matrix[i][j]);
    for (size_t i = 0; i < 4; i++)
        for (size_t j = 0; j < 3; j++)
            CHECK(test60.matrix[i][j] == tds.matrix[i + 6][j]);
    for (size_t i = 0; i < 6; i++)
        CHECK(train60.target_values[i] == tds.target_values[i]);
    for (size_t i = 0; i < 4; i++)
        CHECK(test60.target_values[i] == tds.target_values[i + 6]);

    // Assert: metadata is copied to both halves
    CHECK(train60.feature_names == tds.feature_names);
    CHECK(test60.feature_names == tds.feature_names);
    CHECK(train60.target_name == "y");
    CHECK(test60.target_name == "y");

    // Edges: split(1) puts everything into train, split(0) into test
    auto [train100, test0] = tds.split(1.);
    CHECK(train100.matrix.get_rows() == 10);
    CHECK(train100.target_values.size() == 10);
    CHECK(test0.matrix.get_rows() == 0);
    CHECK(test0.target_values.empty());

    auto [train0, test100] = tds.split(0.);
    CHECK(train0.matrix.get_rows() == 0);
    CHECK(train0.target_values.empty());
    CHECK(test100.matrix.get_rows() == 10);
    CHECK(test100.target_values.size() == 10);

    // Out-of-range ratios must be rejected
    CHECK_THROWS(tds.split(-0.1););
    CHECK_THROWS(tds.split(1.1););

    return finish("split");
}
