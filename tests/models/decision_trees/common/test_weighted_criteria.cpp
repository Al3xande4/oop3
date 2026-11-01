#include <cmath>
#include <span>
#include <vector>

#include "cart_test_data.hpp"
#include "models/decision_trees/cart.hpp"
#include "models/decision_trees/engine/criteria/gini.hpp"
#include "models/decision_trees/engine/criteria/variance.hpp"
#include "models/decision_trees/engine/split_finder.hpp"
#include "test_util.hpp"

// Взвешенные критерии сравниваются с медленным/наивным подсчетом.
// Ограничения размеров проверяются по числу исходных индексов (s), а не по сумме кратностей (W).

void test_weighted_gini_vs_slow()
{
    // 4 объекта, 2 класса (0 и 1), кратности {2, 3, 1, 4}
    Matrix X = make_matrix({{1.0}, {2.0}, {3.0}, {4.0}});
    std::vector<double> y = {0.0, 0.0, 1.0, 1.0};
    std::vector<size_t> counts = {2, 3, 1, 4};
    std::vector<size_t> object_ids = {0, 1, 2, 3};
    std::vector<size_t> feature_ids = {0};

    SplitFinder finder;
    GiniCriteria crit{2};
    auto split = finder.find_best(X, y, object_ids, feature_ids, crit, 1, counts);

    CHECK(split.has_value());
    // Ручной подсчет формулы A_L/W_L + A_R/W_R:
    // Порог 1.5: (4/2) + (34/8) = 2.0 + 4.25 = 6.25
    // Порог 2.5: (25/5) + (25/5) = 5.0 + 5.0 = 10.0 (лучший!)
    // Порог 3.5: (26/6) + (16/4) = 4.333 + 4.0 = 8.333
    CHECK_NEAR(split->threshold, 2.5, 1e-9);
    CHECK_NEAR(split->score, 10.0, 1e-9);
}

void test_weighted_variance_vs_slow()
{
    // Матрица 4x1, отклик y = {0, 10, 20, 30}, кратности counts = {2, 1, 3, 2}
    Matrix X = make_matrix({{1.0}, {2.0}, {3.0}, {4.0}});
    std::vector<double> y = {0.0, 10.0, 20.0, 30.0};
    std::vector<size_t> counts = {2, 1, 3, 2};
    std::vector<size_t> object_ids = {0, 1, 2, 3};
    std::vector<size_t> feature_ids = {0};

    SplitFinder finder;
    VarianceCriteria crit;
    auto split = finder.find_best(X, y, object_ids, feature_ids, crit, 1, counts);

    CHECK(split.has_value());

    // Медленный ручной подсчет критерия:
    // Порог 1.5:
    // Left: id 0 (w=2, y=0) -> S_L = 0, W_L = 2 -> S_L^2/W_L = 0
    // Right: id 1,2,3 -> w={1,3,2}, y={10,20,30} -> S_R = 10+60+60 = 130, W_R = 6 -> S_R^2/W_R = 16900/6 = 2816.666
    // Total score 1.5 = 2816.666
    //
    // Порог 2.5:
    // Left: id 0,1 -> S_L = 10, W_L = 3 -> S_L^2/W_L = 100/3 = 33.333
    // Right: id 2,3 -> S_R = 120, W_R = 5 -> S_R^2/W_R = 14400/5 = 2880.0
    // Total score 2.5 = 2913.333 -> лучший!
    //
    // Порог 3.5:
    // Left: id 0,1,2 -> S_L = 70, W_L = 6 -> S_L^2/W_L = 4900/6 = 816.666
    // Right: id 3 -> S_R = 60, W_R = 2 -> S_R^2/W_R = 3600/2 = 1800.0
    // Total score 3.5 = 2616.666

    CHECK_NEAR(split->threshold, 2.5, 1e-9);
    CHECK_NEAR(split->score, 100.0 / 3.0 + 14400.0 / 5.0, 1e-9);
}

void test_size_constraints_use_distinct_objects_not_weights()
{
    // 2 различных объекта, но с большими кратностями: count = {10, 10}
    // При min_samples_leaf = 2, если бы считали по сумме весов (W=10),
    // разбиение было бы разрешено. Но по числу исходных индексов в листе будет только 1 объект!
    // Поэтому разбиение должно быть ЗАПРЕЩЕНО (leaf_size < 2).
    Matrix X = make_matrix({{1.0}, {2.0}});
    std::vector<double> y = {0.0, 10.0};
    std::vector<size_t> counts = {10, 10};
    std::vector<size_t> object_ids = {0, 1};
    std::vector<size_t> feature_ids = {0};

    SplitFinder finder;
    VarianceCriteria crit;
    auto split = finder.find_best(X, y, object_ids, feature_ids, crit, 2, counts);

    // Должен быть nullopt, так как каждый потомок получает только по 1 уникальному объекту
    CHECK(!split.has_value());
}

// При первых константных кандидатах проверяются дополнительные признаки.
void test_constant_candidates_continue_search()
{
    // Матрица где признак 0 - константный, а признак 1 - информативный
    Matrix X = make_matrix({{1.0, 0.0}, {1.0, 0.0}, {1.0, 10.0}, {1.0, 10.0}});
    std::vector<double> y = {0.0, 0.0, 100.0, 100.0};
    std::vector<size_t> counts = {1, 1, 1, 1};
    std::vector<size_t> object_ids = {0, 1, 2, 3};
    std::vector<size_t> feature_ids = {0, 1};

    SplitFinder finder;
    VarianceCriteria crit;
    // Ищем разбиение по признаку 0 (константному)
    auto res0 = finder.find_for_feature(X, y, object_ids, 0, crit, 1, counts);
    CHECK(res0.is_constant == true);
    CHECK(!res0.split.has_value());

    // Ищем лучшее по всем признакам: должно выбрать признак 1, не остановившись на 0
    auto best = finder.find_best(X, y, object_ids, feature_ids, crit, 1, counts);
    CHECK(best.has_value());
    CHECK(best->feature == 1);
    CHECK_NEAR(best->threshold, 5.0, 1e-9);
}

// Проверка пункта ТЗ:
// "классификация проверяет усреднение вероятностей, отсутствующие классы,
// ничьи, один класс и неравные кратности"
void test_unequal_weights_classification()
{
    // Два объекта с одинаковым признаком (нельзя разделить):
    // объект 0: класс 0, вес 1
    // объект 1: класс 1, вес 9
    Matrix X = make_matrix({{0.0}, {0.0}});
    std::vector<double> y = {0.0, 1.0};
    std::vector<size_t> counts = {1, 9};

    CartClassifier classifier(TreeParams{.max_depth = 1, .min_samples_split = 2, .min_samples_leaf = 1});
    classifier.fit_with_counts(X, y, counts);

    Matrix proba = classifier.predict_proba(X);
    CHECK(proba.get_cols() == 2);
    // Доля класса 0 = 1 / (1 + 9) = 0.1
    CHECK_NEAR(proba(0, 0), 0.1, 1e-9);
    // Доля класса 1 = 9 / (1 + 9) = 0.9
    CHECK_NEAR(proba(0, 1), 0.9, 1e-9);

    // predict выбирает класс с большей вероятностью (класс 1)
    auto pred = classifier.predict(X);
    CHECK_NEAR(pred[0], 1.0, 1e-9);
}

int main()
{
    test_weighted_variance_vs_slow();
    test_weighted_gini_vs_slow();
    test_size_constraints_use_distinct_objects_not_weights();
    test_constant_candidates_continue_search();
    test_unequal_weights_classification();
    return 0;
}
