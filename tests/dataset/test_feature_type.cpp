#include "dataset/dataset.hpp"
#include "test_util.hpp"

int main()
{
    const std::filesystem::path csv = write_temp_file(
        "lab1_feature_type.csv",
        "small_num,text,wide_num,target\n"
        "1,a,10.5,40\n"
        "2,b,20.5,50\n"
        "3,a,30.5,60\n"
        "4,b,40.5,40\n"
        "5,a,50.5,10\n"
        "6,b,60.5,20\n"
        "7,a,70.5,30\n"
        "1,b,80.5,30\n"
        "2,a,90.5,10\n"
        "3,b,100.5,100\n");

    Dataset ds = Dataset::load_dataset(csv);
    DatasetInfo info = ds.analyse();

    CHECK(info.total_count == 10);
    CHECK(info.column_names.size() == 4);

    const FeatureInfo &small_num = info.feature_info.at("small_num");
    const FeatureInfo &text = info.feature_info.at("text");
    const FeatureInfo &wide_num = info.feature_info.at("wide_num");

    CHECK(std::holds_alternative<CategoricalFeatureInfo>(small_num));
    CHECK(std::holds_alternative<CategoricalFeatureInfo>(text));
    CHECK(std::holds_alternative<NumericFeatureInfo>(wide_num));

    CHECK(std::get<CategoricalFeatureInfo>(small_num).categories.size() == 7);
    CHECK(std::get<CategoricalFeatureInfo>(text).categories.size() == 2);
    CHECK(std::get<NumericFeatureInfo>(wide_num).missing_count == 0);
    CHECK_NEAR(std::get<NumericFeatureInfo>(wide_num).min, 10.5, 1e-12);
    CHECK_NEAR(std::get<NumericFeatureInfo>(wide_num).max, 100.5, 1e-12);

    std::filesystem::remove(csv);

    return finish("feature_type");
}
