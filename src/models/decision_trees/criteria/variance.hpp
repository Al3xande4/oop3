#pragma once

#include <span>

class VarianceCriteria
{
private:
    double m_sum_left, m_sum_right, m_total_sum;
    size_t m_count_left, m_count_right, m_total_count;

public:
    /**
     * @brief Считает общую сумму и число целей.
     * @param targets полный вектор целей выборки
     */
    void calculate_total(std::span<const double> targets, std::span<const size_t> object_ids);

    /**
     * @brief Обнуляет левую сторону, перенося всю сумму в правую.
     */
    void init();

    /**
     * @brief Переносит одно значение из правой стороны в левую.
     * @param target целевое значение объекта
     */
    void update(double target);

    /**
     * @brief Возвращает G = S_L²/m_L + S_R²/m_R.
     * @return оценка текущего разбиения (больше — лучше)
     */
    double score() const;
};