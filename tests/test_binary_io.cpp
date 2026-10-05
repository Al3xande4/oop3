#include "dataset/transformed.hpp"
#include "test_util.hpp"

int main()
{
    const std::filesystem::path path = std::filesystem::temp_directory_path() / "lab1_binary_io.bin";

    constexpr int rows = 3;
    constexpr int cols = 4;

    Matrix original(rows, cols, Layout::RowMajor);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            original[{i, j}] = i * cols + j + 0.25;

    original.save(path);
    CHECK(std::filesystem::exists(path));
    CHECK(std::filesystem::file_size(path) > 0);

    Matrix loaded = Matrix::load(path);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            CHECK((loaded[{i, j}] == original[{i, j}]));

    const std::filesystem::path colmajor_path =
        std::filesystem::temp_directory_path() / "lab1_binary_io_colmajor.bin";

    Matrix colmajor(rows, cols, Layout::ColumnMajor);
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            colmajor[{i, j}] = i - 3.5 * j;

    colmajor.save(colmajor_path);
    Matrix colmajor_loaded = Matrix::load(colmajor_path);

    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            CHECK((colmajor_loaded[{i, j}] == colmajor[{i, j}]));

    std::filesystem::remove(path);
    std::filesystem::remove(colmajor_path);

    return finish("binary_io");
}
