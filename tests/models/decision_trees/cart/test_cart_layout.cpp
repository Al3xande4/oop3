#include <cstddef>
#include <span>
#include <vector>

#include "cart_test_data.hpp"
#include "models/decision_trees/cart.hpp"
#include "models/decision_trees/criteria/variance.hpp"
#include "models/decision_trees/leaf/mean_leaf.hpp"
#include "models/decision_trees/tree_builder.hpp"
#include "test_util.hpp"

namespace
{

std::vector<std::size_t> iota_ids(std::size_t n)
{
    std::vector<std::size_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    return ids;
}

} // namespace

void test_regression_predictions_match()
{
    // Логически одинаковые матрицы при обоих layout дают одинаковые предсказания.
    LabeledDataset row_major = make_nested_regression(Layout::RowMajor);
    LabeledDataset col_major = make_nested_regression(Layout::ColumnMajor);

    CartRegressor rm_model({4, 2, 1});
    CartRegressor cm_model({4, 2, 1});

    rm_model.fit(row_major.X, row_major.y);
    cm_model.fit(col_major.X, col_major.y);

    std::vector<double> pred_rm = rm_model.predict(row_major.X);
    std::vector<double> pred_cm = cm_model.predict(col_major.X);

    CHECK(pred_rm.size() == pred_cm.size());
    for (std::size_t i = 0; i < pred_rm.size(); i++)
        CHECK_NEAR(pred_rm[i], pred_cm[i], 1e-12);

    // Предсказания одного и того же обученного дерева на матрице другого layout.
    std::vector<double> mixed = rm_model.predict(col_major.X);
    for (std::size_t i = 0; i < pred_rm.size(); i++)
        CHECK_NEAR(pred_rm[i], mixed[i], 1e-12);
}

void test_classification_matches()
{
    LabeledDataset row_major = make_three_class(Layout::RowMajor);
    LabeledDataset col_major = make_three_class(Layout::ColumnMajor);

    CartClassifier rm_model({4, 2, 1});
    CartClassifier cm_model({4, 2, 1});

    rm_model.fit(row_major.X, row_major.y);
    cm_model.fit(col_major.X, col_major.y);

    std::span<const double> classes_rm = rm_model.classes();
    std::span<const double> classes_cm = cm_model.classes();
    CHECK(classes_rm.size() == classes_cm.size());
    for (std::size_t c = 0; c < classes_rm.size(); c++)
        CHECK_NEAR(classes_rm[c], classes_cm[c], 1e-12);

    Matrix proba_rm = rm_model.predict_proba(row_major.X);
    Matrix proba_cm = cm_model.predict_proba(col_major.X);

    CHECK(proba_rm.get_rows() == proba_cm.get_rows());
    CHECK(proba_rm.get_cols() == proba_cm.get_cols());
    for (std::size_t i = 0; i < proba_rm.get_rows(); i++)
        for (std::size_t c = 0; c < proba_rm.get_cols(); c++)
            CHECK_NEAR(proba_rm(i, c), proba_cm(i, c), 1e-12);

    std::vector<double> pred_rm = rm_model.predict(row_major.X);
    std::vector<double> pred_cm = cm_model.predict(col_major.X);
    for (std::size_t i = 0; i < pred_rm.size(); i++)
        CHECK_NEAR(pred_rm[i], pred_cm[i], 1e-12);
}

void test_identical_splits()
{
    // Структура дерева, построенного по одним и тем же данным, не зависит от layout.
    LabeledDataset row_major = make_nested_regression(Layout::RowMajor);
    LabeledDataset col_major = make_nested_regression(Layout::ColumnMajor);

    std::vector<std::size_t> feature_ids = {0};

    std::vector<std::size_t> ids_rm = iota_ids(row_major.y.size());
    std::vector<std::size_t> ids_cm = iota_ids(col_major.y.size());

    TreeBuilder<VarianceCriteria, MeanLeaf> builder(VarianceCriteria{}, MeanLeaf{}, 4, 2, 1);
    Node<double> root_rm = builder.build(row_major.X, row_major.y, ids_rm, feature_ids);
    Node<double> root_cm = builder.build(col_major.X, col_major.y, ids_cm, feature_ids);

    TreeStats stats_rm = compute_tree_stats(root_rm);
    TreeStats stats_cm = compute_tree_stats(root_cm);
    CHECK(stats_rm.nodes == stats_cm.nodes);
    CHECK(stats_rm.leaves == stats_cm.leaves);
    CHECK(stats_rm.depth == stats_cm.depth);

    std::vector<Split> splits_rm;
    std::vector<Split> splits_cm;
    collect_splits(root_rm, splits_rm);
    collect_splits(root_cm, splits_cm);

    CHECK(splits_rm.size() == splits_cm.size());
    for (std::size_t i = 0; i < splits_rm.size(); i++)
    {
        CHECK(splits_rm[i].feature == splits_cm[i].feature);
        CHECK(splits_rm[i].threshold == splits_cm[i].threshold);
        CHECK(splits_rm[i].score == splits_cm[i].score);
    }
}

void test_wide_matrix_both_layouts()
{
    // Прямоугольная матрица 3 × 4: срезы и обучение работают при обоих layout.
    const std::vector<std::vector<double>> rows = {
        {0., 1., 2., 3.},
        {1., 0., 3., 2.},
        {2., 3., 0., 1.},
    };
    std::vector<double> y = {0., 1., 2.};

    for (Layout layout : {Layout::RowMajor, Layout::ColumnMajor})
    {
        Matrix X = make_matrix(rows, layout);
        CHECK(X.get_rows() == 3);
        CHECK(X.get_cols() == 4);

        // Доступ к элементам и срезы согласованы.
        for (std::size_t i = 0; i < 3; i++)
        {
            auto row = X(i, std::nullopt);
            CHECK(row.size() == 4);
            for (std::size_t j = 0; j < 4; j++)
                CHECK_NEAR(row(j), rows[i][j], 1e-12);
        }
        for (std::size_t j = 0; j < 4; j++)
        {
            auto col = X(std::nullopt, j);
            CHECK(col.size() == 3);
            for (std::size_t i = 0; i < 3; i++)
                CHECK_NEAR(col(i), rows[i][j], 1e-12);
        }

        CartClassifier model({3, 2, 1});
        model.fit(X, y);

        Matrix proba = model.predict_proba(X);
        CHECK(proba.get_rows() == 3);
        CHECK(proba.get_cols() == 3);

        double sum = 0.;
        for (std::size_t c = 0; c < 3; c++)
            sum += proba(0, c);
        CHECK_NEAR(sum, 1., 1e-12);
    }
}

int main()
{
    test_regression_predictions_match();
    test_classification_matches();
    test_identical_splits();
    test_wide_matrix_both_layouts();

    return finish("Cart layout equivalence");
}
