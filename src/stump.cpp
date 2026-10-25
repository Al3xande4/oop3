#include "dataset/transformed.hpp"
#include "models/api/model.hpp"
#include "models/decision_trees/stump.hpp"
#include "cli.hpp"

#include <iostream>
#include <filesystem>

int main(int argc, char *argv[])
{
    try
    {
        StumpOptions options = parse_stump_cli(argc, argv);

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

        auto model = DecisionStumpRegressor(*options.min_samples_leaf);
        model.fit(train.matrix, train.target_values);

        if (model.split_feature())
        {
            std::cout << "Chosen feature column: " << train.feature_names[*model.split_feature()] << '\n';
            std::cout << "Chosen threshold: " << *model.split_threshold() << '\n';
            std::cout << "Left leaf: " << model.get_left_leaf() << '\n';
            std::cout << "Right leaf: " << model.get_right_leaf() << '\n';
        }
        else
            std::cout << "Split was not possible, predicting constant: " << model.get_left_leaf() << " .\n";

        auto predictions = model.predict(test.matrix);

        double Q = 0.;
        for (size_t i = 0; i < predictions.size(); i++)
        {
            if (i < 10)
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
