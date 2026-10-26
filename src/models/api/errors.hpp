#pragma once

#include <stdexcept>
#include <string>

/**
 * @file errors.hpp
 * @brief Иерархия исключений ядра.
 *
 * Все ошибки ядра — производные от классов стандартной библиотеки
 * (@c std::runtime_error, @c std::invalid_argument, ...), поэтому вызывающий
 * код может ловить как типы ядра, так и их стандартные базы.
 */

/**
 * @brief Базовая ошибка выполнения ядра.
 *
 * Предназначена для сбоев, не подпадающих под остальные категории
 * (ошибки ввода-вывода, неожиданные состояния). В текущем коде не бросается:
 * зарезервирована для будущих подсистем ядра.
 *
 * База: @c std::runtime_error.
 */
class CoreError : public std::runtime_error
{
public:
    CoreError(const std::string &message) : std::runtime_error{message} {};
};

/**
 * @brief Выход индекса за допустимые границы.
 *
 * Ошибки обращения по индексу к контейнерам ядра. В текущем коде не бросается:
 * @c Matrix использует стандартный @c std::out_of_range в @c at,
 * тип зарезервирован для остальных контейнеров.
 *
 * База: @c std::out_of_range.
 */
class IndexOutOfBounds : public std::out_of_range
{
public:
    IndexOutOfBounds(const std::string &message) : std::out_of_range{message} {};
};

/**
 * @brief Несовместимые размеры данных.
 *
 * Бросается, когда размеры входных данных противоречат друг другу:
 * число строк матрицы признаков не равно размеру вектора целей
 * или число столбцов при обучении и предсказании различается
 * (см. @c IModel::fit, @c IModel::predict), а также при несовпадении
 * длин векторов в функциях расстояния.
 *
 * База: @c std::invalid_argument.
 */
class DimensionMismatch : public std::invalid_argument
{
public:
    DimensionMismatch(const std::string &message) : std::invalid_argument{message} {};
};

/**
 * @brief Пустая выборка.
 *
 * Бросается, когда модель обучают на нулевом числе объектов
 * (например, @c DecisionStump::fit или @c KnnModel::fit с пустым вектором целей).
 *
 * База: @c std::invalid_argument.
 */
class EmptyData : public std::invalid_argument
{
public:
    EmptyData(const std::string &message) : std::invalid_argument{message} {};
};

/**
 * @brief Обращение к модели до обучения.
 *
 * Бросается при вызове @c predict, @c split_feature, @c split_threshold,
 * @c get_left_leaf, @c get_right_leaf до успешного @c fit.
 *
 * База: @c std::logic_error.
 */
class NotFitted : public std::logic_error
{
public:
    NotFitted(const std::string &message) : std::logic_error{message} {};
};

/**
 * @brief Недопустимое значение гиперпараметра.
 *
 * Бросается в конструкторах моделей: например, @c k <= 0 у k-NN
 * или @c min_samples_leaf == 0 у решающего пня.
 *
 * База: @c std::invalid_argument.
 */
class HyperParameterError : public std::invalid_argument
{
public:
    HyperParameterError(const std::string &message) : std::invalid_argument{message} {};
};
