#include "dataset/transformed.hpp"
#include "models/knn/model.hpp"
#include "models/api/model.hpp"
#include "models/api/distance.hpp"

#include <cxxopts.hpp>
#include <iostream>
#include <filesystem>
#include <cmath>

int main(int argc, char *argv[])
{
    cxxopts::Options options("knn", "K nearest neighbours");
    options.add_options()("help,h", "Help")("bin,dataset,f", "Path to the dataset binary file", cxxopts::value<std::filesystem::path>());
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        return 0;
    }

    if (result.count("dataset") == 0)
    {
        std::cout << "You must specify input binary file, if you don't have it, run ds_load to convert csv format into binary matrix representations, to see options use --help\n";
        return 1;
    }

    std::filesystem::path path_binary = result["dataset"].as<std::filesystem::path>();

    TransformedDataset dataset = TransformedDataset::load(path_binary);
    auto [train, test] = dataset.split(0.6);

    std::cout << train.target_values.size() << " " << test.target_values.size() << " " << dataset.target_values.size() << '\n';

    auto distance = std::make_shared<EuclidDistance>();
    auto weight = std::make_shared<InverseDistanceWeight>();

    KnnRegressor model = KnnRegressor(5, distance, weight);
    model.fit(train.matrix, train.target_values);

    auto predictions = model.predict(test.matrix);

    std::cout << test.target_name << '\n';
    double Q = 0.;
    for (size_t i = 0; i < predictions.size(); i++)
    {
        std::cout << "Prediction: " << predictions[i] << ", target value: " << test.target_values[i] << '\n';
        Q += std::abs(predictions[i] - test.target_values[i]);
    }
    std::cout << Q / predictions.size() << '\n';

    return 0;
}
