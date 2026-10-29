#include <cstddef>
#include <span>
#include <vector>

#include "cart_test_data.hpp"
#include "models/decision_trees/cart.hpp"
#include "test_util.hpp"

// Исходные метки классов не обязаны быть номерами 0..C-1.
// Дерево обязано работать с любыми возрастающими метками, например 10, 42, 100.

void test_depth_zero_labels()
{
    // Без разбиений частоты считаются по всем классам, включая нули.
    Matrix X = make_matrix({{0.}, {1.}, {2.}, {3.}});
    std::vector<double> y = {10., 10., 42., 100.};

    CartClassifier model(0, 2, 1);
    model.fit(X, y);

    std::span<const double> classes = model.classes();
    CHECK(classes.size() == 3);
    CHECK_NEAR(classes[0], 10., 1e-12);
    CHECK_NEAR(classes[1], 42., 1e-12);
    CHECK_NEAR(classes[2], 100., 1e-12);

    Matrix proba = model.predict_proba(X);
    CHECK(proba.get_rows() == 4);
    CHECK(proba.get_cols() == 3);

    for (std::size_t row = 0; row < 4; row++)
    {
        CHECK_NEAR(proba(row, 0), 0.5, 1e-12);
        CHECK_NEAR(proba(row, 1), 0.25, 1e-12);
        CHECK_NEAR(proba(row, 2), 0.25, 1e-12);
    }

    // Ничья 0.25: предсказывается меньшая исходная метка 42.
    std::vector<double> predictions = model.predict(X);
    for (double p : predictions)
        CHECK_NEAR(p, 10., 1e-12); // 0.5 — максимум однозначно
}

void test_split_with_arbitrary_labels()
{
    // Разбиение по признаку при метках 10, 42, 100.
    Matrix X = make_matrix({{0.}, {1.}, {2.}, {7.}, {8.}, {9.}});
    std::vector<double> y = {10., 10., 10., 42., 42., 100.};

    CartClassifier model(3, 2, 1);
    model.fit(X, y);

    std::span<const double> classes = model.classes();
    CHECK(classes.size() == 3);
    CHECK_NEAR(classes[0], 10., 1e-12);
    CHECK_NEAR(classes[1], 42., 1e-12);
    CHECK_NEAR(classes[2], 100., 1e-12);

    Matrix proba = model.predict_proba(X);
    CHECK(proba.get_rows() == 6);
    CHECK(proba.get_cols() == 3);

    for (std::size_t row = 0; row < 6; row++)
    {
        double sum = 0.;
        for (std::size_t c = 0; c < 3; c++)
        {
            CHECK(proba(row, c) >= 0.);
            sum += proba(row, c);
        }
        CHECK_NEAR(sum, 1., 1e-12);
    }

    // Сепарируемая часть: первые три объекта класса 10, последние два — 42.
    std::vector<double> predictions = model.predict(X);
    for (std::size_t i = 0; i < 3; i++)
        CHECK_NEAR(predictions[i], 10., 1e-12);
    for (std::size_t i = 3; i < 5; i++)
        CHECK_NEAR(predictions[i], 42., 1e-12);
    CHECK_NEAR(predictions[5], 100., 1e-12);

    // Каждая предсказанная метка принадлежит набору классов.
    for (double p : predictions)
    {
        bool known = false;
        for (std::size_t c = 0; c < classes.size(); c++)
            if (p == classes[c])
                known = true;
        CHECK(known);
    }
}

int main()
{
    test_depth_zero_labels();
    // Далее дерево строит разбиения при несмежных метках.
    test_split_with_arbitrary_labels();

    return finish("Cart classifier labels");
}
