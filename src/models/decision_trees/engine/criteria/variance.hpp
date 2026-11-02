#pragma once

#include <span>

class VarianceCriteria
{
private:
    double m_sum_left{0}, m_sum_right{0}, m_total_sum{0};
    double m_count_left{0}, m_count_right{0}, m_total_count{0};

public:
    /**
     * @brief Считает общую сумму и число целей.
     * @param targets полный вектор целей выборки
     */
    void calculate_total(
        std::span<const double> targets,
        std::span<const size_t> object_ids,
        std::span<const size_t> counts
    );

    /**
     * @brief Обнуляет левую сторону, перенося всю сумму в правую.
     */
    void init();

    /**
     * @brief Переносит одно значение из правой стороны в левую.
     * @param target целевое значение объекта
     */
    void update(double target, double weight);

    /**
     * @brief Возвращает G = S_L²/m_L + S_R²/m_R.
     * @return оценка текущего разбиения (больше — лучше)
     */
    double score() const;

    size_t total_count() const { return m_total_count; }
};