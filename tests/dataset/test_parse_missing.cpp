#include "dataset/dataset.hpp"
#include "test_util.hpp"

int main()
{
    const std::filesystem::path csv = write_temp_file(
        "lab1_parse_missing.csv",
        "num,cat,target\n"
        "1,alpha,10\n"
        "2,beta,\n"
        "3,gamma,\n"
        "4,,\n"
        "5,beta,\n"
        "6,gamma,\n"
        "7,alpha,\n"
        "8,beta,\n"
        "9,gamma,\n"
        ",beta,\n");

    Dataset ds = Dataset::load_dataset(csv);
    DatasetInfo info = ds.analyse();

    CHECK(info.total_count == 10);
    CHECK(info.column_names.size() == 3);

    const FeatureInfo &num = info.feature_info.at("num");
    const FeatureInfo &cat = info.feature_info.at("cat");

    CHECK(std::holds_alternative<NumericFeatureInfo>(num));
    CHECK(std::holds_alternative<CategoricalFeatureInfo>(cat));

    CHECK(std::get<NumericFeatureInfo>(num).missing_count == 1);
    CHECK(std::get<CategoricalFeatureInfo>(cat).missing_count == 1);

    CHECK_NEAR(std::get<NumericFeatureInfo>(num).median, 5.0, 1e-12);

    ds.impute(info, MedianImputation{});
    DatasetInfo after = ds.analyse();

    CHECK(std::get<NumericFeatureInfo>(after.feature_info.at("num")).missing_count == 0);
    CHECK(std::get<CategoricalFeatureInfo>(after.feature_info.at("cat")).missing_count == 0);

    CHECK_NEAR(std::get<NumericFeatureInfo>(after.feature_info.at("num")).median, 5.0, 1e-12);
    CHECK(std::get<CategoricalFeatureInfo>(after.feature_info.at("cat"))
              .frequencies.at(datasetconfig::category_imputation) == 1);

    std::filesystem::remove(csv);

    return finish("parse_missing");
}
