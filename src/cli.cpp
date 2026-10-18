#include "cli.hpp"

#include <cxxopts.hpp>

KnnOptions parse_knn_cli(int argc, char *argv[])
{
    KnnOptions parsed;

    cxxopts::Options options("knn", "K nearest neighbours");
    options.add_options()("help,h", "Help")("bin,dataset,f", "Path to the dataset binary file", cxxopts::value<std::filesystem::path>());
    auto result = options.parse(argc, argv);

    parsed.help = result.count("help") > 0;
    parsed.help_text = options.help();

    if (result.count("dataset"))
        parsed.dataset = result["dataset"].as<std::filesystem::path>();

    return parsed;
}

LoadOptions parse_load_cli(int argc, char *argv[])
{
    LoadOptions parsed;

    cxxopts::Options options("ds_load", "Load dataset from file and transform it into numeric matrix.");
    options.add_options()("help,h", "Help")("csv,dataset,sourcepath,f", "Path to the dataset", cxxopts::value<std::filesystem::path>())("dest,output,o", "Path to the file with processed data", cxxopts::value<std::filesystem::path>())("histogram", "Number of histogram intervals", cxxopts::value<size_t>())("fill,impute,i", "Imputation strategy", cxxopts::value<std::string>()->default_value("mean"))("s,scale", "Scaling strategy", cxxopts::value<std::string>()->default_value("robust"))("encode,e", "Encoding strategy", cxxopts::value<std::string>()->default_value("onehot"))("t,target", "Target column #", cxxopts::value<std::size_t>());
    auto result = options.parse(argc, argv);

    parsed.help = result.count("help") > 0;
    parsed.help_text = options.help();

    if (result.count("dataset"))
        parsed.dataset = result["dataset"].as<std::filesystem::path>();

    if (result.count("dest"))
        parsed.dest = result["dest"].as<std::filesystem::path>();

    if (result.count("histogram"))
        parsed.histogram = result["histogram"].as<size_t>();

    if (result.count("target"))
        parsed.target = result["target"].as<std::size_t>();

    parsed.impute = result["impute"].as<std::string>();
    parsed.scale = result["scale"].as<std::string>();
    parsed.encode = result["encode"].as<std::string>();

    return parsed;
}
