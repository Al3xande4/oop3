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
    virtual void fit(
        const Matrix &features,
        std::span<const double> target) = 0;

    /**
     * @brief Возвращает предсказание для каждой строки матрицы.
     *
     * @param features матрица признаков; число столбцов должно совпадать
     *                 с числом столбцов обучающей матрицы
     * @return вектор предсказаний длиной @c features.get_rows()
     * @throw NotFitted @c predict вызван до @c fit
     * @throw DimensionMismatch число столбцов не совпадает с обучающим
     */
    virtual std::vector<double> predict(
        const Matrix &features) const = 0;
};
