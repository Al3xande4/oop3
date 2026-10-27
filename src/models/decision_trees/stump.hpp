#pragma once

#include "models/api/model.hpp"
#include "models/api/errors.hpp"
#include "split.hpp"

#include <span>
#include <memory>
#include <optional>
#include <unordered_map>

/**
 * @file stump.hpp
 * @brief Реализация решающего пня (decision stump).
 *
 * Пень — дерево глубины 1: выбирается один признак и один порог,
 * значения меньше или равны порогу отправляются в левый лист,
 * остальные — в правый. Оба листа выдают константу:
 * среднее (регрессия) или наиболее частую категорию (классификация).
 */

/**
 * @brief Общий интерфейс решающего пня.
 *
 * Алгоритм обучения (@c fit): для каждого признака строки сортируются
 * по значению признака (O(n log n)), затем один проход по отсортированному
 * порядку с инкрементальным критерием перебирает все пороги между
 * соседними различающимися значениями. Порог — среднее арифметическое
 * двух соседних значений (@c std::midpoint). Итоговая сложность —
 * O(d·n log n) на d признаков.
 *
 * Ограничение @c min_samples_leaf не даёт образовать лист меньше этого
 * числа объектов; если допустимого разбиения нет (например, все значения
 * признаков одинаковы), модель вырождается в константную:
 * @c split_feature и @c split_threshold возвращают @c std::nullopt,
 * а @c predict выдаёт общую константу.
 *
 * @note При равных оценках побеждает первая найденная пара
 * (меньший индекс признака, затем меньший порог) — обход детерминирован.
 */
class IDecisionStump : public IModel
{
protected:
    std::optional<Split> m_split;

    size_t m_min_samples_leaf; ///< Минимальный размер листа

    bool m_fitted;             ///< Признак успешного fit
    size_t m_fit_feature_size; ///< Число столбцов обучающей матрицы

    SplitFinder m_split_finder;

public:
    /**
     * @brief Создаёт пень с заданным критерием и ограничением на лист.
     * @param criteria критерий качества разбиения (владеющий указатель)
     * @param min_samples_leaf минимальное число объектов в листе, >= 1
     * @throw HyperParameterError @p min_samples_leaf меньше 1
     */
    IDecisionStump(size_t min_samples_leaf) : m_split{std::nullopt}, m_min_samples_leaf{min_samples_leaf}, m_fitted{false}, m_fit_feature_size{0}
    {
        if (min_samples_leaf < 1)
            throw HyperParameterError("DecisionStump: min samples leaf must be greater than 0");
    };

    /**
     * @brief Ищет лучший признак и порог за O(d·n log n).
     * @param features матрица признаков обучающей выборки
     * @param targets вектор целей, по значению на строку
     * @throw DimensionMismatch число строк не равно размеру @p targets
     * @throw EmptyData пустая выборка
     * @note Переобучение сбрасывает предыдущие признак и порог.
     *       Допустимого разбиения может не оказаться — тогда признак
     *       и порог остаются @c std::nullopt (см. класс).
     */
    virtual void fit(
        const Matrix &features,
        std::span<const double> targets) = 0;

    /**
     * @brief Индекс выбранного признака.
     * @return индекс столбца либо @c std::nullopt, если разбиение вырождено
     * @throw NotFitted до вызова @c fit
     */
    std::optional<size_t> split_feature() const
    {
        if (!m_fitted)
            throw NotFitted("DesionStump: split_feature before fit");

        if (m_split)
            return m_split.value().feature;
        return std::nullopt;
    };

    /**
     * @brief Найденный порог: значение <= порога идёт в левый лист.
     * @return порог либо @c std::nullopt, если разбиение вырождено
     * @throw NotFitted до вызова @c fit
     */
    std::optional<double> split_threshold() const
    {
        if (!m_fitted)
            throw NotFitted("DesionStump: split_threshold before fit");

        if (m_split)
            return m_split.value().threshold;
        return std::nullopt;
    };

    /**
     * @brief Константа левого листа (среднее или мода).
     * @throw NotFitted до вызова @c fit
     */
    virtual double get_left_leaf() const = 0;

    /**
     * @brief Константа правого листа (среднее или мода).
     * @throw NotFitted до вызова @c fit
     */
    virtual double get_right_leaf() const = 0;

    virtual ~IDecisionStump() = default;
};

/**
 * @brief Решающий пень для регрессии.
 *
 * Использует @c VarianceCriteria (критерий G = S_L²/m_L + S_R²/m_R).
 * В листьях — среднее значение целей этого листа; при вырожденном
 * разбиении оба листа равны среднему по всей выборке.
 */
class DecisionStumpRegressor : public IDecisionStump
{
protected:
    double m_left_mean;  ///< Среднее целей левого листа
    double m_right_mean; ///< Среднее целей правого листа
    double m_constant;   ///< Общее среднее (вырожденное разбиение)

public:
    /**
     * @brief Создаёт регрессионный пень.
     * @param min_samples_leaf минимальный размер листа, >= 1
     * @throw HyperParameterError @p min_samples_leaf меньше 1
     */
    DecisionStumpRegressor(size_t min_samples_leaf) : IDecisionStump(min_samples_leaf), m_left_mean{0.}, m_right_mean{0.}, m_constant{0.} {};

    /**
     * @brief Подбирает порог по критерию дисперсии и считает средние листьев.
     * @param features матрица признаков обучающей выборки
     * @param targets вектор целей
     * @throw DimensionMismatch несовпадение числа строк и размера целей
     * @throw EmptyData пустая выборка
     */
    void fit(
        const Matrix &features,
        std::span<const double> targets) override;

    /**
     * @brief Предсказывает среднее целей листа, в который попадает объект.
     * @param features матрица признаков с тем же числом столбцов, что и при обучении
     * @return вектор предсказаний длиной @c features.get_rows()
     * @throw NotFitted до @c fit
     * @throw DimensionMismatch число столбцов изменилось
     */
    std::vector<double> predict(
        const Matrix &features) const override;

    /**
     * @brief Среднее целей левого листа.
     * @throw NotFitted до @c fit
     */
    double get_left_leaf() const override
    {
        if (!m_fitted)
            throw NotFitted("DesionStumpRegressor: get_left_leaf before fit");

        return m_left_mean;
    }

    /**
     * @brief Среднее целей правого листа.
     * @throw NotFitted до @c fit
     */
    double get_right_leaf() const override
    {
        if (!m_fitted)
            throw NotFitted("DesionStumpRegressor: get_right_leaf before fit");

        return m_right_mean;
    }
};

/**
 * @brief Решающий пень для классификации.
 *
 * Использует @c GiniCriteria (критерий H = A_L/m_L + A_R/m_R).
 * В листьях — наиболее частая категория этого листа; при вырожденном
 * разбиении оба листа равны моде по всей выборке.
 */
class DecisionStumpClassifier : public IDecisionStump
{
protected:
    double m_left_most_common;  ///< Мода левого листа
    double m_right_most_common; ///< Мода правого листа
    double m_constant;          ///< Общая мода (вырожденное разбиение)

public:
    /**
     * @brief Создаёт классификационный пень.
     * @param min_samples_leaf минимальный размер листа, >= 1
     * @throw HyperParameterError @p min_samples_leaf меньше 1
     */
    DecisionStumpClassifier(size_t min_samples_leaf) : IDecisionStump(min_samples_leaf), m_left_most_common{0.}, m_right_most_common{0.}, m_constant{0.} {};

    /**
     * @brief Подбирает порог по критерию Джини и считает моды листьев.
     * @param features матрица признаков обучающей выборки
     * @param targets вектор категорий (числа типа double)
     * @throw DimensionMismatch несовпадение числа строк и размера целей
     * @throw EmptyData пустая выборка
     */
    void fit(
        const Matrix &features,
        std::span<const double> targets) override;

    /**
     * @brief Предсказывает моду листа, в который попадает объект.
     * @param features матрица признаков с тем же числом столбцов, что и при обучении
     * @return вектор предсказанных категорий длиной @c features.get_rows()
     * @throw NotFitted до @c fit
     * @throw DimensionMismatch число столбцов изменилось
     */
    std::vector<double> predict(
        const Matrix &features) const override;

    /**
     * @brief Мода левого листа.
     * @throw NotFitted до @c fit
     */
    double get_left_leaf() const override
    {
        if (!m_fitted)
            throw NotFitted("DesionStumpClassifier: get_left_leaf before fit");

        return m_left_most_common;
    }

    /**
     * @brief Мода правого листа.
     * @throw NotFitted до @c fit
     */
    double get_right_leaf() const override
    {
        if (!m_fitted)
            throw NotFitted("DesionStumpClassifier: get_right_leaf before fit");

        return m_right_most_common;
    }
};
