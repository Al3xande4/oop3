#pragma once

#include <span>
#include <vector>

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
class GiniCriteria
{
private:
    std::vector<size_t> m_category_count;
    std::vector<size_t> m_category_count_right;
    double m_initial_right;
    size_t m_count_left, m_count_right, m_total_count;
    double m_left, m_right;

    size_t m_classes_count;

public:
    GiniCriteria(size_t classes_count) : m_classes_count{classes_count}
    {
        m_category_count.resize(classes_count);
        m_category_count_right.resize(classes_count);
    };

    /**
     * @brief Считает частоты всех категорий.
     * @param targets полный вектор целей выборки
     */
    void calculate_total(
        std::span<const double> targets,
        std::span<const size_t> object_ids,
        std::span<const size_t> counts
    );

    /**
     * @brief Копирует частоты в правую сторону и обнуляет левую.
     */
    void init();

    /**
     * @brief Переносит объект одной категории из правой стороны в левую.
     * @param target категория переносимого объекта
     */
    void update(double target, double weight);

    /**
     * @brief Возвращает H = A_L/m_L + A_R/m_R.
     * @return оценка текущего разбиения (больше — лучше)
     */
    double score() const;
};