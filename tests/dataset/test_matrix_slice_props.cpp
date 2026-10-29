#include <cstddef>
#include <optional>
#include <type_traits>

#include "dataset/matrix.hpp"
#include "test_util.hpp"

// Дополнительные проверки интерфейса MatrixSlice: направление, шаг,
// константность оболочки и элементов, одноэлементный span, as_const.

void test_axis_and_stride(Layout layout)
{
    Matrix X(4, 3, layout);
    for (std::size_t i = 0; i < 4; i++)
        for (std::size_t j = 0; j < 3; j++)
            X(i, j) = static_cast<double>(i * 10 + j);

    for (std::size_t i = 0; i < 4; i++)
    {
        auto row = X(i, std::nullopt);
        CHECK(row.axis() == SliceAxis::Row);
        CHECK(row.size() == 3);
        // RowMajor: строка непрерывна, шаг 1; ColumnMajor: шаг = rows.
        CHECK(row.stride() == (layout == Layout::RowMajor ? 1u : 4u));
        CHECK(row.is_contiguous() == (layout == Layout::RowMajor));
    }

    for (std::size_t j = 0; j < 3; j++)
    {
        auto col = X(std::nullopt, j);
        CHECK(col.axis() == SliceAxis::Column);
        CHECK(col.size() == 4);
        // ColumnMajor: столбец непрерывен, шаг 1; RowMajor: шаг = cols.
        CHECK(col.stride() == (layout == Layout::ColumnMajor ? 1U : 3U));
        CHECK(col.is_contiguous() == (layout == Layout::ColumnMajor));
    }
}

void test_axis_not_swapped_by_layout()
{
    // Направление всегда задаётся аргументами, а не layout.
    for (Layout layout : {Layout::RowMajor, Layout::ColumnMajor})
    {
        Matrix X(5, 4, layout);
        auto row = X(2, std::nullopt);
        auto col = X(std::nullopt, 1);
        CHECK(row.axis() == SliceAxis::Row);
        CHECK(col.axis() == SliceAxis::Column);
        CHECK(row.size() == 4);
        CHECK(col.size() == 5);
    }
}

void test_strided_read_write()
{
    // Неплотные срезы читаются и меняют исходную матрицу при обоих layout.
    for (Layout layout : {Layout::RowMajor, Layout::ColumnMajor})
    {
        Matrix X(3, 4, layout);
        for (std::size_t i = 0; i < 3; i++)
            for (std::size_t j = 0; j < 4; j++)
                X(i, j) = static_cast<double>(i * 4 + j);

        auto row = X(1, std::nullopt);
        for (std::size_t j = 0; j < 4; j++)
            CHECK_NEAR(row(j), X(1, j), 1e-12);
        row(2) = 77.;
        CHECK_NEAR(X(1, 2), 77., 1e-12);

        auto col = X(std::nullopt, 3);
        for (std::size_t i = 0; i < 3; i++)
            CHECK_NEAR(col(i), X(i, 3), 1e-12);
        col(0) = 55.;
        CHECK_NEAR(X(0, 3), 55., 1e-12);
    }
}

void test_const_matrix_returns_const_elements()
{
    // Константная матрица возвращает срезы с константными элементами.
    Matrix X(2, 3, Layout::RowMajor);
    X(0, 0) = 1.;

    const Matrix &view = X;
    auto row = view(0, std::nullopt);
    auto col = view(std::nullopt, 1);

    static_assert(std::is_same_v<decltype(row), MatrixSlice<const double>>);
    static_assert(std::is_same_v<decltype(col), MatrixSlice<const double>>);
    static_assert(std::is_same_v<decltype(view(0, 0)), const double &>);

    CHECK_NEAR(row(0), 1., 1e-12);
    CHECK_NEAR(col(0), 0., 1e-12);
}

void test_slice_shell_constness()
{
    // const MatrixSlice<double> фиксирует оболочку, но не элементы.
    Matrix X(2, 2, Layout::RowMajor);
    X(0, 0) = 1.;
    X(1, 0) = 2.;

    auto mutable_row = X(0, std::nullopt);
    const auto fixed_row = mutable_row;
    fixed_row(0) = 9.; // менять элемент можно
    CHECK_NEAR(X(0, 0), 9., 1e-12);
}

void test_as_const()
{
    Matrix X(2, 3, Layout::ColumnMajor);
    X(1, 2) = 4.;

    auto row = X(1, std::nullopt);
    auto readable = row.as_const();

    static_assert(std::is_same_v<decltype(readable), MatrixSlice<const double>>);
    CHECK(readable.size() == row.size());
    CHECK(readable.stride() == row.stride());
    CHECK(readable.axis() == row.axis());
    CHECK_NEAR(readable(2), 4., 1e-12);

    // Уже читаемый срез: as_const возвращает другой читаемый.
    auto readable2 = readable.as_const();
    CHECK_NEAR(readable2(2), 4., 1e-12);
}

void test_single_element_strided_span()
{
    // Одноэлементный срез даёт span даже при ненулевом шаге.
    Matrix X(3, 3, Layout::ColumnMajor);
    X(0, 0) = 5.;

    // Столбец ColumnMajor непрерывен; строка — нет, но через размер 1 не пройти.
    auto col = X(std::nullopt, 0);
    CHECK((*col).size() == 3);

    // Ручной одноэлементный срез с шагом > 1.
    double arr[] = {1., 2., 3.};
    MatrixSlice<double> strided(arr, 1, 3, SliceAxis::Row);
    CHECK(strided.is_contiguous() == false);
    std::span<double> s = *strided; // size <= 1 — разрешено
    CHECK(s.size() == 1);
    CHECK_NEAR(s[0], 1., 1e-12);

    // Пустой срез также даёт пустой span при любом шаге.
    MatrixSlice<double> empty_slice(arr, 0, 5, SliceAxis::Column);
    std::span<double> empty_span = *empty_slice;
    CHECK(empty_span.empty());
}

void test_empty_slice_stride_kept()
{
    // Пустой срез хранит nullptr независимо от переданного указателя.
    double arr[] = {1., 2.};
    MatrixSlice<double> slice(arr, 0, 2, SliceAxis::Row);
    CHECK(slice.data() == nullptr);
    CHECK(slice.empty());
    CHECK(slice.stride() == 2); // шаг сохраняется
    CHECK(slice.axis() == SliceAxis::Row);
}

int main()
{
    test_axis_and_stride(Layout::RowMajor);
    test_axis_and_stride(Layout::ColumnMajor);
    test_axis_not_swapped_by_layout();
    test_strided_read_write();
    test_const_matrix_returns_const_elements();
    test_slice_shell_constness();
    test_as_const();
    test_single_element_strided_span();
    test_empty_slice_stride_kept();

    return finish("Matrix Slice properties");
}
