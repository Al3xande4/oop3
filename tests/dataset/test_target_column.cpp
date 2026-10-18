#include "test_util.hpp"
#include "dataset/dataset.hpp"

#include <filesystem>
#include <string>

int main()
{
    // Arrange: 12 rows so both columns keep >= 8 unique values and
    // stay numeric under datasetconfig::numeric_column_threshold
    std::string numeric_csv = "x,t\n";
    for (int i = 1; i <= 12; i++)
        numeric_csv += std::to_string(i) + ".0," + std::to_string(i * 100) + ".0\n";
    const std::filesystem::path csv = write_temp_file("test_target_default.csv", numeric_csv);

    // Case 1: default target is the last column
    Dataset ds = Dataset::load_dataset(csv);
    DatasetInfo info = ds.analyse();

    CHECK(info.column_names.size() == 2);
    CHECK(info.column_names.back() == "t");
    CHECK(std::holds_alternative<NumericFeatureInfo>(info.feature_info.at("x")));
    CHECK(std::holds_alternative<NumericFeatureInfo>(info.feature_info.at("t")));

    TransformedDataset tds = ds.transform(info, MinMaxScaling{}, CategoryEncoding::ONE_HOT, Layout::RowMajor);
    CHECK(tds.target_name == "t");
    CHECK(tds.feature_names.size() == 1);
    CHECK(tds.feature_names[0] == "x");
    CHECK(tds.matrix.get_rows() == 12);
    CHECK(tds.matrix.get_cols() == 1);
    CHECK(tds.target_values.size() == 12);
    for (int i = 1; i <= 12; i++)
        CHECK_NEAR(tds.target_values[i - 1], i * 100., 1e-12);

    // Case 2: explicit target index 0 swaps the columns
    Dataset ds0 = Dataset::load_dataset(csv, 0);
    DatasetInfo info0 = ds0.analyse();

    CHECK(info0.column_names.size() == 2);
    CHECK(info0.column_names.front() == "t");
    CHECK(info0.column_names.back() == "x");

    TransformedDataset tds0 = ds0.transform(info0, MinMaxScaling{}, CategoryEncoding::ONE_HOT, Layout::RowMajor);
    CHECK(tds0.target_name == "x");
    CHECK(tds0.feature_names.size() == 1);
    CHECK(tds0.feature_names[0] == "t");
    CHECK(tds0.target_values.size() == 12);
    for (int i = 1; i <= 12; i++)
        CHECK_NEAR(tds0.target_values[i - 1], i, 1e-12);

    // Case 3: categorical target is encoded by sorted category order
    const std::filesystem::path csv_cat = write_temp_file(
        "test_target_categorical.csv",
        "x,v\n"
        "1.0,red\n"
        "2.0,blue\n"
        "3.0,red\n"
        "4.0,green\n");

    Dataset ds_cat = Dataset::load_dataset(csv_cat);
    DatasetInfo info_cat = ds_cat.analyse();
    CHECK(std::holds_alternative<CategoricalFeatureInfo>(info_cat.feature_info.at("v")));

    TransformedDataset tds_cat = ds_cat.transform(info_cat, MinMaxScaling{}, CategoryEncoding::ONE_HOT, Layout::RowMajor);
    CHECK(tds_cat.target_name == "v");
    CHECK(tds_cat.target_values.size() == 4);
    // categories sorted: blue = 0, green = 1, red = 2
    CHECK_NEAR(tds_cat.target_values[0], 2., 1e-12); // red
    CHECK_NEAR(tds_cat.target_values[1], 0., 1e-12); // blue
    CHECK_NEAR(tds_cat.target_values[2], 2., 1e-12); // red
    CHECK_NEAR(tds_cat.target_values[3], 1., 1e-12); // green

    std::filesystem::remove(csv);
    std::filesystem::remove(csv_cat);
    return finish("target_column");
}
