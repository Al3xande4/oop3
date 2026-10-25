#include "dataset/transformed.hpp"
#include "models/knn/model.hpp"
#include "models/api/model.hpp"
#include "models/api/distance.hpp"
#include "cli.hpp"

#include <iostream>
#include <filesystem>
#include <cmath>

std::shared_ptr<IDistance> make_distance(const std::string &name)
{
    if (name == "euclid")
        return std::make_shared<EuclidDistance>();
    if (name == "chebyshev")
        return std::make_shared<ChebyshevDistance>();
    if (name == "manhatten")
        return std::make_shared<ManhattenDistance>();

    throw std::invalid_argument("make_distance: unknown distance " + name);
}

std::shared_ptr<IKnnWeight> make_weight(const std::string &name)
{
    if (name == "inverse")
        return std::make_shared<InverseDistanceWeight>();
    if (name == "uniform")
        return std::make_shared<UniformWeight>(10);

    throw std::invalid_argument("make_weight: unknown weight " + name);
}

int main(int argc, char *argv[])
{
    try
    {
        KnnOptions options = parse_knn_cli(argc, argv);

        if (options.help)
        {
            std::cout << options.help_text << std::endl;
            return 0;
        }

        if (!options.dataset)
        {
            std::cout << "You must specify input binary file, if you don't have it, run ds_load to convert csv format into binary matrix representations, to see options use --help\n";
            return 1;
        }

        std::filesystem::path path_binary = *options.dataset;

        TransformedDataset dataset = TransformedDataset::load(path_binary);
        auto [train, test] = dataset.split(0.6);

        std::cout << train.target_values.size() << " " << test.target_values.size() << " " << dataset.target_values.size() << '\n';

        auto distance = make_distance(options.distance);
        auto weight = make_weight(options.weight);

        KnnRegressor model = KnnRegressor(*options.k, distance, weight);

        model.fit(train.matrix, train.target_values);

        auto predictions = model.predict(test.matrix);

        std::cout << test.target_name << '\n';

        double Q = 0.;
        for (size_t i = 0; i < predictions.size(); i++)
        {
            if (i < 15)
                std::cout << "Prediction: " << predictions[i] << ", target value: " << test.target_values[i] << '\n';

            Q += std::abs(predictions[i] - test.target_values[i]);
        }
        std::cout << Q / predictions.size() << '\n';
    }
    catch (const std::exception &error)
    {
        std::cout << error.what();
        return 1;
    }
    return 0;
}
