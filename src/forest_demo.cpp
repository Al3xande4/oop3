#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include "dataset/matrix.hpp"
#include "models/decision_trees/forest.hpp"

namespace
{

struct SyntheticDataset
{
    Matrix X_train;
    std::vector<double> y_train;
    Matrix X_test;
    std::vector<double> y_test;
};

SyntheticDataset generate_synthetic_3class(size_t n_per_class = 60, uint32_t seed = 0)
{
    std::mt19937 gen(seed);
    std::normal_distribution<double> dist(0.0, 0.45);

    // 3 центра кластеров в вершинах равностороннего треугольника
    const double centers[3][2] = {{0.0, 0.0}, {2.0, 0.0}, {1.0, 1.732}};

    std::vector<std::vector<double>> train_rows, test_rows;
    std::vector<double> train_y, test_y;

    for (size_t c = 0; c < 3; ++c)
    {
        for (size_t i = 0; i < n_per_class; ++i)
        {
            double x1 = centers[c][0] + dist(gen);
            double x2 = centers[c][1] + dist(gen);

            if (i < static_cast<size_t>(n_per_class * 0.7))
            {
                train_rows.push_back({x1, x2});
                train_y.push_back(static_cast<double>(c));
            }
            else
            {
                test_rows.push_back({x1, x2});
                test_y.push_back(static_cast<double>(c));
            }
        }
    }

    Matrix X_tr(train_rows.size(), 2);
    for (size_t r = 0; r < train_rows.size(); ++r)
    {
        X_tr(r, 0) = train_rows[r][0];
        X_tr(r, 1) = train_rows[r][1];
    }

    Matrix X_te(test_rows.size(), 2);
    for (size_t r = 0; r < test_rows.size(); ++r)
    {
        X_te(r, 0) = test_rows[r][0];
        X_te(r, 1) = test_rows[r][1];
    }

    return {std::move(X_tr), std::move(train_y), std::move(X_te), std::move(test_y)};
}

double compute_accuracy(const std::vector<double> &pred, const std::vector<double> &target)
{
    size_t correct = 0;
    for (size_t i = 0; i < pred.size(); ++i)
    {
        if (std::abs(pred[i] - target[i]) < 1e-6)
            correct++;
    }
    return static_cast<double>(correct) / static_cast<double>(pred.size());
}

} // namespace

int main()
{
    std::cout << "=================================================================\n";
    std::cout << "  Random Forest Demonstration (Synthetic 3-Class Classification)\n";
    std::cout << "=================================================================\n\n";

    SyntheticDataset data = generate_synthetic_3class(70, 42);
    std::cout << "Generated 2D dataset with 3 Gaussian clusters:\n";
    std::cout << "  Train objects: " << data.X_train.get_rows() << " (" << data.X_train.get_cols() << " features)\n";
    std::cout << "  Test objects:  " << data.X_test.get_rows() << " (" << data.X_test.get_cols() << " features)\n";
    std::cout << "  Classes:       {0, 1, 2}\n\n";

    std::cout << "-----------------------------------------------------------------\n";
    std::cout << "1. Impact of Number of Trees (n_estimators) with max_features = Sqrt\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::cout << std::left << std::setw(16) << "n_estimators" << std::setw(16) << "Train Acc" << std::setw(16)
              << "Test Acc\n";
    std::cout << "-----------------------------------------------------------------\n";

    std::vector<size_t> tree_counts = {1, 3, 5, 10, 25, 50, 100};
    for (size_t n : tree_counts)
    {
        RandomForestParams params{
            .estimators = n, .bootstrap = true, .max_features = MaxFeaturesMode::Sqrt, .random_state = 0, .n_jobs = 4
        };
        ForestClassifier clf(params);
        clf.fit(data.X_train, data.y_train);

        auto train_pred = clf.predict(data.X_train);
        auto test_pred = clf.predict(data.X_test);

        double train_acc = compute_accuracy(train_pred, data.y_train);
        double test_acc = compute_accuracy(test_pred, data.y_test);

        std::cout << std::left << std::setw(16) << n << std::setw(16) << std::fixed << std::setprecision(4) << train_acc
                  << std::setw(16) << std::fixed << std::setprecision(4) << test_acc << "\n";
    }

    std::cout << "\n-----------------------------------------------------------------\n";
    std::cout << "2. Comparison of Feature Search Policies (max_features) at 50 Trees\n";
    std::cout << "-----------------------------------------------------------------\n";

    {
        ForestClassifier clf_all(
            RandomForestParams{.estimators = 50, .max_features = MaxFeaturesMode::All, .random_state = 0, .n_jobs = 4}
        );
        clf_all.fit(data.X_train, data.y_train);
        double acc_all = compute_accuracy(clf_all.predict(data.X_test), data.y_test);

        ForestClassifier clf_sqrt(
            RandomForestParams{.estimators = 50, .max_features = MaxFeaturesMode::Sqrt, .random_state = 0, .n_jobs = 4}
        );
        clf_sqrt.fit(data.X_train, data.y_train);
        double acc_sqrt = compute_accuracy(clf_sqrt.predict(data.X_test), data.y_test);

        std::cout << "  max_features = All  (d = 2): Test Accuracy = " << acc_all << "\n";
        std::cout << "  max_features = Sqrt (q = 1): Test Accuracy = " << acc_sqrt << "\n";
    }

    std::cout << "\n-----------------------------------------------------------------\n";
    std::cout << "3. Predicted Probabilities (predict_proba) on Sample Test Objects\n";
    std::cout << "-----------------------------------------------------------------\n";

    ForestClassifier final_model(
        RandomForestParams{.estimators = 50, .max_features = MaxFeaturesMode::All, .random_state = 0, .n_jobs = 4}
    );
    final_model.fit(data.X_train, data.y_train);

    Matrix proba = final_model.predict_proba(data.X_test);
    auto preds = final_model.predict(data.X_test);

    std::cout << "Showing first 8 test objects with class probabilities:\n\n";
    std::cout << "  Obj  | Coordinates (x1, x2) |  P(c=0)   P(c=1)   P(c=2)  | Sum   | Pred | True\n";
    std::cout << "-------+----------------------+----------------------------+-------+------+------\n";

    for (size_t i = 0; i < std::min<size_t>(8, data.X_test.get_rows()); ++i)
    {
        double p0 = proba(i, 0);
        double p1 = proba(i, 1);
        double p2 = proba(i, 2);
        double p_sum = p0 + p1 + p2;

        std::cout << "  #" << std::setw(3) << i << " | (" << std::setw(6) << std::fixed << std::setprecision(2)
                  << data.X_test(i, 0) << ", " << std::setw(6) << std::fixed << std::setprecision(2)
                  << data.X_test(i, 1) << ") |  " << std::setw(6) << std::fixed << std::setprecision(3) << p0 << "   "
                  << std::setw(6) << std::fixed << std::setprecision(3) << p1 << "   " << std::setw(6) << std::fixed
                  << std::setprecision(3) << p2 << "  | " << std::setw(5) << std::fixed << std::setprecision(3) << p_sum
                  << " |   " << static_cast<int>(preds[i]) << "  |  " << static_cast<int>(data.y_test[i]) << "\n";
    }

    std::cout << "\nAll probabilities sum to 1.0; ensemble produces smooth posterior probabilities.\n";
    return 0;
}
