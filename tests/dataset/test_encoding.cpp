#include "dataset/dataset.hpp"
#include "test_util.hpp"

namespace
{
    constexpr int kRows = 10;
    constexpr double kMin = 1.5;
    constexpr double kMax = 10.5;

    double scaled(int i)
    {
        return (kMin + i - kMin) / (kMax - kMin);
    }
} // namespace

int main()
{
    const std::filesystem::path csv = write_temp_file(
        "lab1_encoding.csv",
        "val,grp,target\n"
        "1.5,red,10\n"
        "2.5,blue,10\n"
        "3.5,red,10\n"
        "4.5,green,10\n"
        "5.5,blue,10\n"
        "6.5,red,10\n"
        "7.5,green,10\n"
        "8.5,red,10\n"
        "9.5,blue,10\n"
        "10.5,green,10\n");

    Dataset ds = Dataset::load_dataset(csv);
    DatasetInfo info = ds.analyse();

    CHECK(std::holds_alternative<NumericFeatureInfo>(info.feature_info.at("val")));
    CHECK(std::holds_alternative<CategoricalFeatureInfo>(info.feature_info.at("grp")));

    const std::vector<std::string> &cats =
        std::get<CategoricalFeatureInfo>(info.feature_info.at("grp")).categories;
    CHECK(cats.size() == 3);
    CHECK(cats[0] == "blue");
    CHECK(cats[1] == "green");
    CHECK(cats[2] == "red");

    TransformedDataset onehot = ds.transform(info, MinMaxScaling{}, CategoryEncoding::ONE_HOT, Layout::RowMajor);

    CHECK(onehot.feature_names.size() == 4);
    CHECK(onehot.feature_names[0] == "val");
    CHECK(onehot.feature_names[1] == "grp_blue");
    CHECK(onehot.feature_names[2] == "grp_green");
    CHECK(onehot.feature_names[3] == "grp_red");

    const std::string expected_grp[10] = {"red", "blue", "red", "green", "blue",
                                          "red", "green", "red", "blue", "green"};
    const std::string cat_order[3] = {"blue", "green", "red"};

    for (int i = 0; i < kRows; i++)
    {
        const double expected_val = scaled(i);
        CHECK_NEAR((onehot.matrix[{i, 0}]), expected_val, 1e-12);

        int hot = 0;
        for (int c = 0; c < 3; c++)
        {
            const double v = onehot.matrix[{i, 1 + c}];
            CHECK((v == 0.0) || (v == 1.0));
            if (v == 1.0)
            {
                ++hot;
                CHECK(expected_grp[i] == cat_order[c]);
            }
        }
        CHECK(hot == 1);
    }

    TransformedDataset again = ds.transform(info, MinMaxScaling{}, CategoryEncoding::ONE_HOT, Layout::RowMajor);
    for (int i = 0; i < kRows; i++)
        for (int j = 0; j < 4; j++)
            CHECK((again.matrix[{i, j}] == onehot.matrix[{i, j}]));

    TransformedDataset ordinal = ds.transform(info, MinMaxScaling{}, CategoryEncoding::ORDINAL, Layout::RowMajor);

    CHECK(ordinal.feature_names.size() == 2);
    CHECK(ordinal.feature_names[0] == "val");
    CHECK(ordinal.feature_names[1] == "grp");

    const int expected_code[10] = {2, 0, 2, 1, 0, 2, 1, 2, 0, 1};
    for (int i = 0; i < kRows; i++)
    {
        CHECK_NEAR((ordinal.matrix[{i, 0}]), scaled(i), 1e-12);
        CHECK_NEAR((ordinal.matrix[{i, 1}]), static_cast<double>(expected_code[i]), 1e-12);
    }

    std::filesystem::remove(csv);

    return finish("encoding");
}
