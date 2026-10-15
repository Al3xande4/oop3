#include "dataset/dataset.hpp"
#include "dataset/config.hpp"
#include "dataset/imputation.hpp"
#include <cxxopts.hpp>
#include <iostream>
#include <filesystem>

int main(int argc, char *argv[])
{
    cxxopts::Options options("ds_load", "Load dataset from file and transform it into numeric matrix.");
    options.add_options()("help,h", "Help")("csv,dataset,sourcepath,f", "Path to the dataset", cxxopts::value<std::filesystem::path>())("dest,output,o", "Path to the file with processed data", cxxopts::value<std::filesystem::path>())("histogram", "Number of histogram intervals", cxxopts::value<size_t>())("fill,impute,i", "Imputation strategy", cxxopts::value<std::string>()->default_value("mean"))("s,scale", "Scaling strategy", cxxopts::value<std::string>()->default_value("robust"))("encode,e", "Encoding strategy", cxxopts::value<std::string>()->default_value("onehot"))("t,target", "Target column #", cxxopts::value<std::size_t>());
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (result.count("dataset") == 0)
    {
        std::cout << "You must specify input file, to see options use --help\n";
        exit(0);
    }

    std::filesystem::path csvPath = result["dataset"].as<std::filesystem::path>();

    NumericImputationStrategy *imputation;
    ScalingStrategy *scaling;
    CategoryEncoding encoding;

    std::string impOpt = result["impute"].as<std::string>();
    if (impOpt == "mean")
        imputation = new MeanImputation();
    else if (impOpt == "zero")
        imputation = new ZeroImputation();
    else if (impOpt == "median")
        imputation = new MedianImputation();
    else
    {
        std::cout << "Unknown imputation strategy, please use one of: mean, zero, median. Using median as default.\n";
        imputation = new MedianImputation();
    }

    std::string scalingOpt = result["scale"].as<std::string>();
    if (scalingOpt == "minmax")
        scaling = new MinMaxScaling();
    else if (scalingOpt == "standart")
        scaling = new StandartScaling();
    else if (scalingOpt == "robust")
        scaling = new RobustScaling();
    else
    {
        std::cout << "Unknown scaling strategy, please use one of: minmax, standart, robust. Using minmax as default.\n";
        scaling = new RobustScaling();
    }

    std::string encodingOpt = result["encode"].as<std::string>();
    if (encodingOpt == "onehot")
        encoding = CategoryEncoding::ONE_HOT;
    else if (encodingOpt == "ordinal")
        encoding = CategoryEncoding::ORDINAL;
    else
    {
        std::cout << "Unknown encoding strategy, please use one of: onehot, ordinal. Using onehot as default.\n";
        encoding = CategoryEncoding::ONE_HOT;
    }

    if (result.count("histogram"))
    {
        size_t intervals = result["histogram"].as<size_t>();
        datasetconfig::histogram_intervals = intervals;
    }

    std::optional<size_t> target_column = std::nullopt;
    if (result.count("target"))
        target_column = result["target"].as<size_t>();

    try
    {
        Dataset ds = Dataset::load_dataset(csvPath, target_column);

        DatasetInfo info = ds.analyse();

        std::cout << "Save the transformed dataset?(y/yes, n/no): ";
        std::string s;
        std::getline(std::cin, s);
        if (s == "no" || s == "n")
            return 0;

        ds.impute(info, *imputation);
        TransformedDataset tds = ds.transform(info, *scaling, encoding, Layout::RowMajor);

        if (result.count("dest") == 0)
        {
            std::cout << "Didn't specify the path to save processed matrix.\n";
            return 1;
        }
        std::filesystem::path matrix_path = result["dest"].as<std::filesystem::path>();
        tds.save(matrix_path.replace_extension("bin"));
    }
    catch (const std::runtime_error &error)
    {
        std::cout << error.what();
        return 1;
    }

    return 0;
}
