#pragma once

#include "dataset/matrix.hpp"

#include <optional>
#include <span>
#include <unordered_map>
#include <memory>

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
    virtual void calculate_total(std::span<const double> targets, std::span<const size_t> object_ids) = 0;

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
    void calculate_total(std::span<const double> targets, std::span<const size_t> object_ids) override;

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
    void calculate_total(std::span<const double> targets, std::span<const size_t> object_ids) override;

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

struct Split
{
    std::size_t feature;
    double threshold;
    double score;
};

class SplitFinder
{
public:
    std::optional<Split> find_best(
        const Matrix &X,
        std::span<const double> y,
        std::span<const std::size_t> object_ids,
        std::span<const std::size_t> feature_ids,
        std::unique_ptr<ISearchCriteria> criterion,
        std::size_t min_samples_leaf) const;
};