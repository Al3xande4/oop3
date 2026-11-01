#pragma once

#include <span>
#include <vector>

#include "dataset/matrix.hpp"

/**
 * @brief Интерфейс модели машинного обучения ядра.
 *
 * Единый контракт для всех моделей: обучение на матрице признаков
 * и векторе целей, затем предсказание на матрице с тем же числом столбцов.
 * Наследники (k-NN, решающий пень) реализуют fit/predict по-своему,
 * но вызывающий код работает только через этот интерфейс.
 */
class IModel
{
public:
    virtual ~IModel() = default;

    /**
     * @brief Обучает модель на выборке.
     *
     * После успешного вызова @c predict становится доступен;
     * повторный вызов переобучает модель заново.
     *
     * @param features матрица признаков: строка — объект, столбец — признак
     * @param target вектор целевой переменной, по одному значению на строку
     *               матрицы @p features
     * @throw DimensionMismatch число строк @p features не равно размеру @p target
     * @throw EmptyData обучение на пустой выборке
     */
    virtual void fit(const Matrix &features, std::span<const double> target) = 0;

    /**
     * Возвращает предсказание для каждой строки матрицы.
     *
     * @param features Матрица признаков. Число столбцов должно совпадать
     *                 с числом столбцов обучающей выборки.
     * @return Вектор предсказаний длиной `features.get_rows()`.
     *
     * @throw NotFitted Если метод вызван до вызова `fit`.
     * @throw DimensionMismatch Если число столбцов матрицы не совпадает с обучающей.
     */
    virtual std::vector<double> predict(const Matrix &features) const = 0;
};

class IRegressor : public IModel
{
};

class IClassifier : public IModel
{
public:
    virtual Matrix predict_proba(const Matrix &) const = 0;
    virtual std::span<const double> classes() const = 0;

    std::vector<double> predict(const Matrix &features) const override
    {
        Matrix probas = predict_proba(features);

        std::vector<double> result(features.get_rows());

        for (size_t row = 0; row < features.get_rows(); row++)
        {
            size_t best_index = 0;
            double best_probability = -1.;

            for (size_t p = 0; p < probas.get_cols(); p++)
            {
                if (probas(row, p) > best_probability)
                {
                    best_index = p;
                    best_probability = probas(row, p);
                }
            }
            result[row] = classes()[best_index];
        }

        return result;
    }
};
