#pragma once

#include "models/api/model.hpp"
#include "models/api/errors.hpp"

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
 * @brief Интерфейс критерия качества разбиения.
 *
 * Критерий ведёт инкрементный учёт разбиения «слева/справа», что позволяет
 * вычислять оценку нового порога за O(1) вместо пересчёта по всем объектам:
 * @code
 * calculate_total(targets);   // агрегаты по всей выборке (один раз)
 * init();                     // все объекты в правой стороне
 * update(target);             // перенос объекта в левую сторону
 * score();                    // оценка текущего разбиения
 * @endcode
 *
 * Такой порядок делает полный перебор порогов за O(n) на признак
 * после сортировки.
 *
 * @note Порядок вызовов обязателен: @c score определён только когда
 * обе стороны непусты (гарантируется вызывающим кодом через
 * @c min_samples_leaf >= 1).
 */
class ISearchCriteria
{
public:
    /**
     * @brief Накапливает агрегаты по всем целям выборки.
     * @param targets целевая переменная обучающей выборки
     */
    virtual void calculate_total(std::span<const double> targets) = 0;

    /**
     * @brief Сбрасывает разбиение: все объекты находятся справа.
     *
     * Вызывается один раз на признак перед перебором порогов.
     */
    virtual void init() = 0;

    /**
     * @brief Переносит один объект из правой стороны в левую.
     * @param target целевое значение переносимого объекта
     * @note Обновления ведутся по формулам O(1):
     * суммы/счётчики для регрессии, частоты категорий для классификации.
     */
    virtual void update(double target) = 0;

    /**
     * @brief Оценка текущего разбиения: больше — лучше.
     * @return значение критерия; при равенстве побеждает порог,
     *         найденный первым (обход признаков и порогов по возрастанию)
     */
    virtual double score() const = 0;

    virtual ~ISearchCriteria() = default;
};

/**
 * @brief Критерий для регрессии: G = S_L²/m_L + S_R²/m_R.
 *
 * S — сумма целей стороны, m — число объектов стороны.
 * Максимизация G (эквивалентно максимизации межгрупповой дисперсии)
 * минимизирует суммарную внутригрупповую дисперсию разбиения.
 *
 * Хранит только две суммы и два счётчика, поэтому @c update работает за O(1).
 */
class VarianceCriteria : public ISearchCriteria
{
private:
    double m_sum_left, m_sum_right, m_total_sum;
    size_t m_count_left, m_count_right, m_total_count;

public:
    /**
     * @brief Считает общую сумму и число целей.
     * @param targets полный вектор целей выборки
     */
    void calculate_total(std::span<const double> targets) override;

    /**
     * @brief Обнуляет левую сторону, перенося всю сумму в правую.
     */
    void init() override;

    /**
     * @brief Переносит одно значение из правой стороны в левую.
     * @param target целевое значение объекта
     */
    void update(double target) override;

    /**
     * @brief Возвращает G = S_L²/m_L + S_R²/m_R.
     * @return оценка текущего разбиения (больше — лучше)
     */
    double score() const override;
};

/**
 * @brief Критерий для классификации: H = A_L/m_L + A_R/m_R.
 *
 * A = Σ f_i² — сумма квадратов частот категорий (нормированная
 * нечистота Джини без вычитания единицы: максимизация H эквивалентна
 * максимизации чистоты листьев). Категории представляются значениями
 * типа @c double, как и в остальной части ядра.
 *
 * Обновления ведутся по обратным формулам O(1): A_L += 2a+1,
 * A_R += 1-2b, где a — частота категории до переноса в левой стороне,
 * b — в правой.
 */
class GiniCriteria : public ISearchCriteria
{
private:
    std::unordered_map<double, size_t> m_category_count;
    std::unordered_map<double, size_t> m_category_count_right;
    size_t m_count_left, m_count_right, m_total_count;
    double m_left, m_right;

public:
    /**
     * @brief Считает частоты всех категорий.
     * @param targets полный вектор целей выборки
     */
    void calculate_total(std::span<const double> targets) override;

    /**
     * @brief Копирует частоты в правую сторону и обнуляет левую.
     */
    void init() override;

    /**
     * @brief Переносит объект одной категории из правой стороны в левую.
     * @param target категория переносимого объекта
     */
    void update(double target) override;

    /**
     * @brief Возвращает H = A_L/m_L + A_R/m_R.
     * @return оценка текущего разбиения (больше — лучше)
     */
    double score() const override;
};

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
    std::optional<size_t> m_feature_index; ///< Индекс выбранного признака; nullopt — вырожденное разбиение
    std::optional<double> m_threshold;     ///< Порог разбиения; nullopt — вырожденное разбиение

    std::unique_ptr<ISearchCriteria> m_criteria; ///< Критерий качества (variance или Gini)
    size_t m_min_samples_leaf;                   ///< Минимальный размер листа

    bool m_fitted;             ///< Признак успешного fit
    size_t m_fit_feature_size; ///< Число столбцов обучающей матрицы

public:
    /**
     * @brief Создаёт пень с заданным критерием и ограничением на лист.
     * @param criteria критерий качества разбиения (владеющий указатель)
     * @param min_samples_leaf минимальное число объектов в листе, >= 1
     * @throw HyperParameterError @p min_samples_leaf меньше 1
     */
    IDecisionStump(std::unique_ptr<ISearchCriteria> criteria, size_t min_samples_leaf) : m_feature_index{std::nullopt}, m_threshold{std::nullopt}, m_criteria(std::move(criteria)), m_min_samples_leaf{min_samples_leaf}, m_fitted{false}, m_fit_feature_size{0}
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
        std::span<const double> targets);

    /**
     * @brief Индекс выбранного признака.
     * @return индекс столбца либо @c std::nullopt, если разбиение вырождено
     * @throw NotFitted до вызова @c fit
     */
    std::optional<size_t> split_feature() const
    {
        if (!m_fitted)
            throw NotFitted("DesionStump: split_feature before fit");

        return m_feature_index;
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

        return m_threshold;
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
    DecisionStumpRegressor(size_t min_samples_leaf) : IDecisionStump(std::make_unique<VarianceCriteria>(), min_samples_leaf), m_left_mean{0.}, m_right_mean{0.}, m_constant{0.} {};

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
    DecisionStumpClassifier(size_t min_samples_leaf) : IDecisionStump(std::make_unique<GiniCriteria>(), min_samples_leaf), m_left_most_common{0.}, m_right_most_common{0.}, m_constant{0.} {};

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
