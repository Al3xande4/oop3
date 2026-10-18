#include "test_util.hpp"
#include "dataset/transformed.hpp"

#include <filesystem>

int main()
{
    // Arrange: a small dataset with every serialised part filled in
    TransformedDataset original;
    original.matrix = Matrix(3, 2);
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 2; j++)
            original.matrix[i][j] = i * 2. + j + 0.25;
    original.feature_names = {"f0", "f1"};
    original.target_name = "y";
    original.target_values = {10., 0., 1.};

    const std::filesystem::path path =
        std::filesystem::temp_directory_path() / "test_binary_io.bin";

    // Act
    original.save(path);

    // Assert: the file exists and the round-trip preserves every field
    CHECK(std::filesystem::exists(path));
    CHECK(std::filesystem::file_size(path) > 0);

    TransformedDataset loaded = TransformedDataset::load(path);
    CHECK(loaded.matrix.get_rows() == 3);
    CHECK(loaded.matrix.get_cols() == 2);
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 2; j++)
            CHECK(loaded.matrix[i][j] == original.matrix[i][j]);
    CHECK(loaded.feature_names == original.feature_names);
    CHECK(loaded.target_name == original.target_name);
    CHECK(loaded.target_values == original.target_values);

    std::filesystem::remove(path);
    return finish("binary_io");
}
