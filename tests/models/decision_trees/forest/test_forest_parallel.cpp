#include "cart_test_data.hpp"
#include "models/decision_trees/forest.hpp"
#include "test_util.hpp"

// Одинаковый seed и разные n_jobs (1, 2, 4) дают побитово идентичные результаты.
// Порядок потоков не влияет на результат.
void test_reproducibility_across_n_jobs_regressor()
{
    LabeledDataset data = make_nested_regression();

    RandomForestParams params_j1{
        .estimators = 8,
        .max_depth = 4,
        .random_state = 777,
        .n_jobs = 1,
    };
    ForestRegressor model_j1(params_j1);
    model_j1.fit(data.X, data.y);
    auto pred_j1 = model_j1.predict(data.X);

    RandomForestParams params_j2 = params_j1;
    params_j2.n_jobs = 2;
    ForestRegressor model_j2(params_j2);
    model_j2.fit(data.X, data.y);
    auto pred_j2 = model_j2.predict(data.X);

    RandomForestParams params_j4 = params_j1;
    params_j4.n_jobs = 4;
    ForestRegressor model_j4(params_j4);
    model_j4.fit(data.X, data.y);
    auto pred_j4 = model_j4.predict(data.X);

    CHECK(pred_j1.size() == pred_j2.size());
    CHECK(pred_j1.size() == pred_j4.size());

    for (size_t i = 0; i < pred_j1.size(); ++i)
    {
        CHECK_NEAR(pred_j1[i], pred_j2[i], 1e-12);
        CHECK_NEAR(pred_j1[i], pred_j4[i], 1e-12);
    }
}

void test_reproducibility_across_n_jobs_classifier()
{
    LabeledDataset data = make_three_class();

    RandomForestParams params_j1{
        .estimators = 8,
        .max_depth = 4,
        .random_state = 888,
        .n_jobs = 1,
    };
    ForestClassifier model_j1(params_j1);
    model_j1.fit(data.X, data.y);
    auto proba_j1 = model_j1.predict_proba(data.X);

    RandomForestParams params_j4 = params_j1;
    params_j4.n_jobs = 4;
    ForestClassifier model_j4(params_j4);
    model_j4.fit(data.X, data.y);
    auto proba_j4 = model_j4.predict_proba(data.X);

    CHECK(proba_j1.get_rows() == proba_j4.get_rows());
    CHECK(proba_j1.get_cols() == proba_j4.get_cols());

    for (size_t r = 0; r < proba_j1.get_rows(); ++r)
        for (size_t c = 0; c < proba_j1.get_cols(); ++c)
            CHECK_NEAR(proba_j1(r, c), proba_j4(r, c), 1e-12);
}

// Проверка пункта 10:
// Повторный fit с тем же seed воспроизводит одинаковый результат.
void test_repeated_fit()
{
    LabeledDataset data = make_step_regression();

    ForestRegressor model(RandomForestParams{
        .estimators = 5,
        .random_state = 42,
        .n_jobs = 2,
    });

    model.fit(data.X, data.y);
    auto pred1 = model.predict(data.X);

    model.fit(data.X, data.y);
    auto pred2 = model.predict(data.X);

    for (size_t i = 0; i < pred1.size(); ++i)
        CHECK_NEAR(pred1[i], pred2[i], 1e-12);
}

// Проверка пункта 10:
// Оба layout (RowMajor и ColumnMajor) дают одинаковые предсказания.
void test_layout_invariance()
{
    LabeledDataset rm_data = make_nested_regression(Layout::RowMajor);
    LabeledDataset cm_data = make_nested_regression(Layout::ColumnMajor);

    ForestRegressor rm_model(RandomForestParams{.estimators = 6, .random_state = 42, .n_jobs = 2});
    ForestRegressor cm_model(RandomForestParams{.estimators = 6, .random_state = 42, .n_jobs = 2});

    rm_model.fit(rm_data.X, rm_data.y);
    cm_model.fit(cm_data.X, cm_data.y);

    auto pred_rm = rm_model.predict(rm_data.X);
    auto pred_cm = cm_model.predict(cm_data.X);

    for (size_t i = 0; i < pred_rm.size(); ++i)
        CHECK_NEAR(pred_rm[i], pred_cm[i], 1e-12);

    // Предсказание на матрице другого layout
    auto pred_mixed = rm_model.predict(cm_data.X);
    for (size_t i = 0; i < pred_rm.size(); ++i)
        CHECK_NEAR(pred_rm[i], pred_mixed[i], 1e-12);
}

// Проверка пункта ТЗ: "исключение рабочей задачи"
// Ошибка внутри рабочего потока перехватывается через std::current_exception()
// и передаётся вызывающему потоку через std::rethrow_exception().
void test_worker_task_exception()
{
    LabeledDataset data = make_step_regression();

    // Запуск параллельного обучения с n_jobs = 4, где max_samples заведомо превышает число строк
    ForestRegressor model(RandomForestParams{
        .estimators = 8,
        .max_samples = 1000, // rows = 8, sample() бросит HyperParameterError внутри потоков
        .n_jobs = 4,
    });

    CHECK_THROWS_AS(model.fit(data.X, data.y), HyperParameterError);
}

int main()
{
    test_reproducibility_across_n_jobs_regressor();
    test_reproducibility_across_n_jobs_classifier();
    test_repeated_fit();
    test_layout_invariance();
    test_worker_task_exception();
    return 0;
}
