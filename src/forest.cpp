#include "models/decision_trees/forest.hpp"

#include <charconv>
#include <filesystem>
#include <iostream>

#include "cli.hpp"
#include "dataset/transformed.hpp"
#include "models/decision_trees/engine/max_feature.hpp"

std::optional<MaxFeature> make_max_feature(const std::optional<std::string> &name)
{
    if (!name)
        return std::nullopt;

    const std::string &s = name.value();
    size_t value;
    auto [ptr, _] = std::from_chars(s.data(), s.data() + s.length(), value);
    if (ptr == s.data() + s.length())
        return value;

    if (s == "All")
        return MaxFeaturesMode::All;

    if (s == "Sqrt")
        return MaxFeaturesMode::Sqrt;

    throw CliError("Parametr max-feature can not be interperated: " + s);
}

int main(int argc, char *argv[])
{
    try
    {
        ForestOptions options = parse_forest_cli(argc, argv);

        if (options.help)
        {
            std::cout << options.help_text << std::endl;
            return 0;
        }

        if (!options.dataset)
        {
            std::cout << "You must specify input binary file, if you don't have it, run ds_load to convert csv format "
                         "into binary matrix representations, to see options use --help\n";
            return 1;
        }

        std::filesystem::path path_binary = *options.dataset;

        TransformedDataset dataset = TransformedDataset::load(path_binary);
        auto [train, test] = dataset.split(0.6);

        RandomForestParams params{
            .estimators = options.n_estimators.value(),
            .bootstrap = options.bootstrap.value(),
            .max_samples = options.max_samples,
            .max_features = make_max_feature(options.max_features),
            .max_depth = options.max_depth,
            .min_samples_split = options.min_samples_split.value(),
            .min_samples_leaf = options.min_samples_leaf.value(),
            .random_state = options.random_state,
            .n_jobs = options.n_jobs.value()
        };

        ForestRegressor model(params);

        model.fit(train.matrix, train.target_values);

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
