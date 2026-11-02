#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <memory>

#include "cli.hpp"
#include "dataset/config.hpp"
#include "dataset/dataset.hpp"
#include "dataset/imputation.hpp"

std::unique_ptr<NumericImputationStrategy> make_imputation(const std::string &name)
{
    if (name == "mean")
        return std::make_unique<MeanImputation>();
    if (name == "zero")
        return std::make_unique<ZeroImputation>();
    if (name == "median")
        return std::make_unique<MedianImputation>();

    std::cout << "Unknown imputation strategy, please use one of: mean, zero, median. Using median as default.\n";
    return std::make_unique<MedianImputation>();
}

std::unique_ptr<ScalingStrategy> make_scaling(const std::string &name)
{
    if (name == "minmax")
        return std::make_unique<MinMaxScaling>();
    if (name == "standart" || name == "standard")
        return std::make_unique<StandartScaling>();
    if (name == "robust")
        return std::make_unique<RobustScaling>();

    std::cout << "Unknown scaling strategy, please use one of: minmax, standart, robust. Using robust as default.\n";
    return std::make_unique<RobustScaling>();
}

CategoryEncoding make_encoding(const std::string &name)
{
    if (name == "ordinal")
        return CategoryEncoding::ORDINAL;
    if (name != "onehot")
        std::cout << "Unknown encoding strategy, please use one of: onehot, ordinal. Using onehot as default.\n";

    return CategoryEncoding::ONE_HOT;
}

int main(int argc, char *argv[])
{
    LoadOptions options = parse_load_cli(argc, argv);

    if (options.help)
    {
        std::cout << options.help_text << std::endl;
        return 0;
    }

    if (!options.dataset)
    {
        std::cout << "You must specify input file, to see options use --help\n";
        return 1;
    }

    std::filesystem::path csvPath = *options.dataset;

    auto imputation = make_imputation(options.impute);
    auto scaling = make_scaling(options.scale);
    auto encoding = make_encoding(options.encode);

    if (options.histogram)
        datasetconfig::histogram_intervals = *options.histogram;

    try
    {
        Dataset ds = Dataset::load_dataset(csvPath, options.target);

        DatasetInfo info = ds.analyse();

        std::cout << "Save the transformed dataset?(y/yes, n/no): ";
        std::string s;
        std::getline(std::cin, s);
        if (s == "no" || s == "n")
            return 0;

        ds.impute(info, *imputation);
        TransformedDataset tds = ds.transform(info, *scaling, encoding, Layout::RowMajor);

        if (!options.dest)
        {
            std::cout << "Didn't specify the path to save processed matrix.\n";
            return 1;
        }
        std::filesystem::path matrix_path = *options.dest;
        tds.save(matrix_path.replace_extension("bin"));
    }
    catch (const std::runtime_error &error)
    {
        std::cout << error.what();
        return 1;
    }

    return 0;
}
