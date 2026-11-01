#include "models/decision_trees/forest.hpp"

#include <charconv>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>

#include "cli.hpp"
#include "dataset/transformed.hpp"

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

        std::cout << "=================================================================\n";
        std::cout << "                   Random Forest Regressor\n";
        std::cout << "=================================================================\n";
        std::cout << "Dataset: " << path_binary.string() << "\n";
        std::cout << "  Train size: " << train.matrix.get_rows() << " objects, " << train.matrix.get_cols()
                  << " features\n";
        std::cout << "  Test size:  " << test.matrix.get_rows() << " objects, " << test.matrix.get_cols()
                  << " features\n\n";

        std::cout << "Hyperparameters:\n";
        std::cout << "  n_estimators:      " << options.n_estimators.value() << "\n";
        std::cout << "  bootstrap:         " << (options.bootstrap.value() ? "true" : "false") << "\n";
        std::cout << "  max_samples:       " << (options.max_samples ? std::to_string(*options.max_samples) : "all")
                  << "\n";
        std::cout << "  max_features:      " << options.max_features.value_or("All") << "\n";
        std::cout << "  max_depth:         " << (options.max_depth ? std::to_string(*options.max_depth) : "none")
                  << "\n";
        std::cout << "  min_samples_split: " << options.min_samples_split.value() << "\n";
        std::cout << "  min_samples_leaf:  " << options.min_samples_leaf.value() << "\n";
        std::cout << "  random_state:      " << options.random_state.value_or(0) << "\n";
        std::cout << "  n_jobs:            " << options.n_jobs.value() << "\n\n";

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

        std::cout << "Training Random Forest...\n";
        auto t_fit_start = std::chrono::high_resolution_clock::now();
        model.fit(train.matrix, train.target_values);
        auto t_fit_end = std::chrono::high_resolution_clock::now();
        double fit_ms = std::chrono::duration<double, std::milli>(t_fit_end - t_fit_start).count();
        std::cout << "  Fit time:     " << std::fixed << std::setprecision(2) << fit_ms << " ms\n\n";

        std::cout << "Predicting on test set...\n";
        auto t_pred_start = std::chrono::high_resolution_clock::now();
        auto predictions = model.predict(test.matrix);
        auto t_pred_end = std::chrono::high_resolution_clock::now();
        double pred_ms = std::chrono::duration<double, std::milli>(t_pred_end - t_pred_start).count();
        std::cout << "  Predict time: " << std::fixed << std::setprecision(2) << pred_ms << " ms\n\n";

        std::cout << "Sample predictions (first 10 test objects):\n";
        std::cout << "   # | Prediction | Target Value | Abs Error\n";
        std::cout << "-----+------------+--------------+-----------\n";

        double Q = 0.;
        for (size_t i = 0; i < predictions.size(); i++)
        {
            double diff = std::abs(predictions[i] - test.target_values[i]);
            if (i < 10)
            {
                std::cout << "  #" << std::setw(2) << i << " |   " << std::setw(8) << std::fixed << std::setprecision(6)
                          << predictions[i] << " |     " << std::setw(8) << std::fixed << std::setprecision(6)
                          << test.target_values[i] << " |  " << std::setw(8) << std::fixed << std::setprecision(6)
                          << diff << "\n";
            }
            Q += diff;
        }

        double mae = Q / static_cast<double>(predictions.size());
        std::cout << "-----------------------------------------------------------------\n";
        std::cout << "Test MAE (Mean Absolute Error): " << std::fixed << std::setprecision(6) << mae << "\n";
        std::cout << "=================================================================\n";
    }
    catch (const std::exception &error)
    {
        std::cout << "Error: " << error.what() << "\n";
        return 1;
    }

    return 0;
}
