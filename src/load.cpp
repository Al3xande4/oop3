#include "dataset/dataset.hpp"
#include "dataset/config.hpp"
#include "dataset/imputation.hpp"
#include "cli.hpp"
#include <cstdlib>
#include <iostream>
#include <filesystem>

int main(int argc, char *argv[])
{
    LoadOptions options = parse_load_cli(argc, argv);

    if (options.help)
    {
        std::cout << options.help_text << std::endl;
        exit(0);
    }

    if (!options.dataset)
    {
        std::cout << "You must specify input file, to see options use --help\n";
        exit(0);
    }

    std::filesystem::path csvPath = *options.dataset;

    NumericImputationStrategy *imputation;
    ScalingStrategy *scaling;
    CategoryEncoding encoding;

    const std::string &impOpt = options.impute;
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

    const std::string &scalingOpt = options.scale;
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

    const std::string &encodingOpt = options.encode;
    if (encodingOpt == "onehot")
        encoding = CategoryEncoding::ONE_HOT;
    else if (encodingOpt == "ordinal")
        encoding = CategoryEncoding::ORDINAL;
    else
    {
        std::cout << "Unknown encoding strategy, please use one of: onehot, ordinal. Using onehot as default.\n";
        encoding = CategoryEncoding::ONE_HOT;
    }

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
