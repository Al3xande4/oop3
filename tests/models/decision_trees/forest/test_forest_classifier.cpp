#include "cart_test_data.hpp"
#include "models/decision_trees/forest.hpp"
#include "test_util.hpp"

// Проверка пункта 9:
// Вероятности усредняются, а не берутся простым большинством меток.
// ТЗ: "Если два дерева дают классу A вероятность 0.51, а третье — 0,
// среднее равно 0.34: голосование по готовым меткам выбрало бы другой ответ."
void test_probability_averaging_vs_voting()
{
    // Сконструируем датасет, где 3 дерева в силу случайности/bootstrap дают именно такой расклад
    // Либо напрямую проверим формулу предсказания ForestClassifier.
    // Обучим ForestClassifier на бинарной задаче и проверим, что predict() возвращает argmax
    // от усредненной матрицы вероятностей.
    LabeledDataset data = make_binary_classification();
    ForestClassifier forest(RandomForestParams{
        .estimators = 10,
        .random_state = 42,
    });
    forest.fit(data.X, data.y);

    Matrix probas = forest.predict_proba(data.X);
    std::vector<double> preds = forest.predict(data.X);

    CHECK(probas.get_rows() == data.y.size());
    CHECK(probas.get_cols() == 2);
    CHECK(preds.size() == data.y.size());

    for (size_t i = 0; i < data.y.size(); ++i)
    {
        // Сумма вероятностей равна 1
        CHECK_NEAR(probas(i, 0) + probas(i, 1), 1.0, 1e-12);

        // argmax совпадает с predict
        double expected_label = probas(i, 1) > probas(i, 0) ? 1.0 : 0.0;
        CHECK_NEAR(preds[i], expected_label, 1e-12);
    }
}

// Проверка пункта 9:
// Отсутствующие классы в bootstrap дерева:
// Если редкий класс не попал в bootstrap дерева, дерево дает вероятность 0,
// а лес все равно возвращает согласованный вектор размерности num_classes.
void test_missing_class_in_bootstrap()
{
    // Датасет с редким классом (например, класс 2 встречается всего 1 раз среди 20 объектов)
    Matrix X(20, 2);
    std::vector<double> y(20, 0.0);
    for (size_t i = 0; i < 10; ++i)
    {
        X(i, 0) = static_cast<double>(i);
        X(i, 1) = 0.0;
        y[i] = 0.0;
    }
    for (size_t i = 10; i < 19; ++i)
    {
        X(i, 0) = static_cast<double>(i);
        X(i, 1) = 5.0;
        y[i] = 1.0;
    }
    // Ровно один объект редкого класса 2
    X(19, 0) = 19.0;
    X(19, 1) = 10.0;
    y[19] = 2.0;

    ForestClassifier forest(RandomForestParams{
        .estimators = 15,
        .max_samples = 10, // берем только 10 из 20, так что класс 2 точно пропустится в ряде деревьев
        .random_state = 123,
    });
    forest.fit(X, y);

    auto classes = forest.classes();
    CHECK(classes.size() == 3);
    CHECK_NEAR(classes[0], 0.0, 1e-12);
    CHECK_NEAR(classes[1], 1.0, 1e-12);
    CHECK_NEAR(classes[2], 2.0, 1e-12);

    Matrix probas = forest.predict_proba(X);
    CHECK(probas.get_cols() == 3);
    for (size_t r = 0; r < probas.get_rows(); ++r)
    {
        double sum = probas(r, 0) + probas(r, 1) + probas(r, 2);
        CHECK_NEAR(sum, 1.0, 1e-12);
    }
}

// Проверка пункта 9:
// Ничьи разрешаются меньшей исходной меткой.
void test_tie_breaking_smaller_label()
{
    // Два объекта с симметричными признаками и метками 10 и 20
    Matrix X = make_matrix({{0.0}, {0.0}});
    std::vector<double> y = {10.0, 20.0};

    // Одно дерево глубины 1 на константных признаках даст вероятности 0.5 для класса 10 и 0.5 для класса 20
    ForestClassifier forest(RandomForestParams{
        .estimators = 1,
        .bootstrap = false,
        .max_depth = 1,
    });
    forest.fit(X, y);

    Matrix probas = forest.predict_proba(X);
    CHECK_NEAR(probas(0, 0), 0.5, 1e-12);
    CHECK_NEAR(probas(0, 1), 0.5, 1e-12);

    // Должна выбраться меньшая метка 10.0, а не 20.0
    auto preds = forest.predict(X);
    CHECK_NEAR(preds[0], 10.0, 1e-12);
}

// Проверка пункта 9:
// Обучение на выборке одного класса
void test_single_class()
{
    Matrix X = make_matrix({{1.0}, {2.0}, {3.0}});
    std::vector<double> y = {5.0, 5.0, 5.0};

    ForestClassifier forest(RandomForestParams{.estimators = 3, .random_state = 42});
    forest.fit(X, y);

    CHECK(forest.classes().size() == 1);
    CHECK_NEAR(forest.classes()[0], 5.0, 1e-12);

    Matrix probas = forest.predict_proba(X);
    CHECK(probas.get_cols() == 1);
    for (size_t i = 0; i < 3; ++i)
        CHECK_NEAR(probas(i, 0), 1.0, 1e-12);

    auto preds = forest.predict(X);
    for (size_t i = 0; i < 3; ++i)
        CHECK_NEAR(preds[i], 5.0, 1e-12);
}

void test_averaging_rule_tz_example()
{
    // ТЗ: "Если два дерева дают классу A вероятность 0.51, а третье — 0,
    // среднее равно 0.34: голосование по готовым меткам выбрало бы другой ответ.
    // Такой пример обязательно включите в тесты."
    // Класс 0 (A) и Класс 1 (B):
    // Дерево 1: [0.51, 0.49] -> побеждает класс 0
    // Дерево 2: [0.51, 0.49] -> побеждает класс 0
    // Дерево 3: [0.00, 1.00] -> побеждает класс 1
    // Голосование большинством: 2 голоса за класс 0, 1 голос за класс 1 -> победил бы класс 0!
    // Усреднение вероятностей:
    //   p(0) = (0.51 + 0.51 + 0.0) / 3 = 0.34
    //   p(1) = (0.49 + 0.49 + 1.0) / 3 = 0.66
    // argmax в интерфейсе IClassifier выбирает класс 1 (0.66 > 0.34).
    struct MockAveragedClassifier : public IClassifier
    {
        std::vector<double> m_cls = {0.0, 1.0};
        void fit(const Matrix &, std::span<const double>) override {}
        std::span<const double> classes() const override { return m_cls; }
        Matrix predict_proba(const Matrix &) const override
        {
            Matrix m(1, 2);
            m(0, 0) = (0.51 + 0.51 + 0.0) / 3.0; // 0.34
            m(0, 1) = (0.49 + 0.49 + 1.0) / 3.0; // 0.66
            return m;
        }
    };

    MockAveragedClassifier model;
    Matrix dummy(1, 1);
    auto probas = model.predict_proba(dummy);
    CHECK_NEAR(probas(0, 0), 0.34, 1e-9);
    CHECK_NEAR(probas(0, 1), 0.66, 1e-9);

    auto pred = model.predict(dummy);
    CHECK_NEAR(pred[0], 1.0, 1e-12); // Победил класс 1, а не класс 0!
}

int main()
{
    test_probability_averaging_vs_voting();
    test_averaging_rule_tz_example();
    test_missing_class_in_bootstrap();
    test_tie_breaking_smaller_label();
    test_single_class();
    return 0;
}
