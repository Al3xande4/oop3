#include "dataset/matrix.hpp"
#include "test_util.hpp"

void test_empty_slice()
{
    CHECK_THROWS(MatrixSlice<double> slice(nullptr, 12, 1, SliceAxis::Row););

    MatrixSlice<double> slice(nullptr, 0, 1, SliceAxis::Row);
    CHECK(slice.data() == nullptr);
    CHECK(slice.size() == 0);

    double arr[] = {1, 2, 3};
    MatrixSlice<double> slice2(arr, 0, 1, SliceAxis::Column);
    CHECK(slice2.data() == nullptr);
    CHECK(slice2.size() == 0);

    MatrixSlice<double> slice3(arr, 2, 1, SliceAxis::Column);
    CHECK(slice3.data() == arr);
    CHECK(slice3.size() == 2);
}

void test_non_contiguous()
{
    Matrix X(10, 10, Layout::RowMajor);
    auto col = X(std::nullopt, 0);

    CHECK(!col.is_contiguous());
    CHECK_THROWS_AS(*col, NonContiguousSliceError);

    Matrix X2(10, 10, Layout::ColumnMajor);
    auto row = X2(0, std::nullopt);

    CHECK(!row.is_contiguous());
    CHECK_THROWS_AS(*row, NonContiguousSliceError);
}

void test_empty_col_row()
{
    Matrix X(10, 0, Layout::RowMajor);

    auto row = X(1, std::nullopt);
    CHECK(row.as_span().size() == 0);
    CHECK(row.empty());

    Matrix X1(0, 10, Layout::RowMajor);

    auto col = X1(std::nullopt, 1);
    CHECK(col.as_span().size() == 0);
    CHECK(col.empty());
}

void test_basic(Layout layout)
{
    Matrix X(10, 10, layout);
    for (size_t i = 0; i < 10; i++)
        for (size_t j = 0; j < 10; j++)
            X(i, j) = i + j;

    double value = X(1, 1);
    CHECK_NEAR(value, 2., 1e-9);

    X(5, 5) = 2.0;
    CHECK_NEAR(X(5, 5), 2., 1e-9);

    auto row = X(0, std::nullopt);
    CHECK(!row.empty());
    for (size_t j = 0; j < 10; j++)
        CHECK_NEAR(row(j), X(0, j), 1e-9);

    row(5) = 100.;
    CHECK_NEAR(X(0, 5), 100., 1e-9);

    auto col = X(std::nullopt, 0);
    for (size_t i = 0; i < 10; i++)
        CHECK_NEAR(col(i), X(i, 0), 1e-9);

    col(5) = 120.;
    CHECK_NEAR(X(5, 0), 120., 1e-9);
}

void test_rect_matrix(Layout layout)
{
    Matrix X(5, 3, layout);
    for (size_t i = 0; i < X.get_rows(); i++)
        for (size_t j = 0; j < X.get_cols(); j++)
            X(i, j) = i + j;

    for (size_t row = 0; row < X.get_rows(); row++)
        CHECK(X(row, std::nullopt).size() == X.get_cols());

    for (size_t col = 0; col < X.get_cols(); col++)
        CHECK(X(std::nullopt, col).size() == X.get_rows());

    for (size_t i = 0; i < X.get_rows(); i++)
    {
        for (size_t j = 0; j < X.get_cols(); j++)
        {
            CHECK_NEAR(X(i, j), X(i, std::nullopt)(j), 1e-9);
            CHECK_NEAR(X(i, j), X(std::nullopt, j)(i), 1e-9);
        }
    }
    X(2, std::nullopt)(2) = 100.;
    CHECK_NEAR(X(2, 2), 100., 1e-9);

    X(1, std::nullopt)(0) = 200.;
    CHECK_NEAR(X(1, 0), 200., 1e-9);

    auto row = X(4, std::nullopt);
    auto col = X(std::nullopt, 0);
    if (layout == Layout::ColumnMajor)
    {
        CHECK_THROWS_AS(*row, NonContiguousSliceError);
        CHECK((*col).size() == X.get_rows());
    }
    else
    {
        CHECK_THROWS_AS(*col, NonContiguousSliceError);
        CHECK((*row).size() == X.get_cols());
    }
}

void test_slice_at()
{
    double arr[] = {1., 2., 3.};
    MatrixSlice<double> slice(arr, 3, 1, SliceAxis::Column);

    CHECK_NEAR(slice.at(0), 1., 1e-9);
    CHECK_NEAR(slice.at(1), 2., 1e-9);
    CHECK_NEAR(slice.at(2), 3., 1e-9);
    CHECK_THROWS_AS(slice.at(3), std::out_of_range);
}

int main()
{
    test_empty_slice();
    test_slice_at();
    test_non_contiguous();
    test_empty_col_row();
    test_basic(Layout::ColumnMajor);
    test_basic(Layout::RowMajor);
    test_rect_matrix(Layout::ColumnMajor);
    test_rect_matrix(Layout::RowMajor);

    return finish("Matrix Slice tests");
}