#pragma once

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <variant>
#include <vector>

#include "dataset/matrix.hpp"
#include "models/decision_trees/split.hpp"
#include "models/decision_trees/tree_node.hpp"

/// Общие фикстуры для тестов дерева CART.

inline Matrix make_matrix(const std::vector<std::vector<double>> &rows, Layout layout = Layout::RowMajor)
{
    const std::size_t n = rows.size();
    const std::size_t d = n == 0 ? 0 : rows[0].size();

    Matrix X(n, d, layout);
    for (std::size_t i = 0; i < n; i++)
        for (std::size_t j = 0; j < d; j++)
            X(i, j) = rows[i][j];

    return X;
}

struct LabeledDataset
{
    Matrix X;
    std::vector<double> y;
};

/// Ступенчатый отклик: одно разбиение по признаку 0 между 3 и 4.
inline LabeledDataset make_step_regression(Layout layout = Layout::RowMajor)
{
    return {
        make_matrix({{0.}, {1.}, {2.}, {3.}, {4.}, {5.}, {6.}, {7.}}, layout),
        {0., 0., 0., 0., 10., 10., 10., 10.},
    };
}

/// Вложенные пороги: одного разбиения не хватает, нужно хотя бы два.
inline LabeledDataset make_nested_regression(Layout layout = Layout::RowMajor)
{
    return {
        make_matrix({{0.}, {1.}, {2.}, {3.}, {4.}, {5.}, {6.}, {7.}}, layout),
        {0., 0., 10., 10., 5., 5., 15., 15.},
    };
}

/// Бинарная классификация, сепарируемая по признаку 0.
inline LabeledDataset make_binary_classification(Layout layout = Layout::RowMajor)
{
    return {
        make_matrix({{0.}, {1.}, {2.}, {3.}, {4.}, {5.}}, layout),
        {0., 0., 0., 1., 1., 1.},
    };
}

/// Три класса, сепарируемые по одному признаку.
inline LabeledDataset make_three_class(Layout layout = Layout::RowMajor)
{
    return {
        make_matrix({{0.}, {1.}, {2.}, {3.}, {4.}, {5.}}, layout),
        {0., 0., 1., 1., 2., 2.},
    };
}

struct TreeStats
{
    std::size_t nodes{0};   ///< Всего узлов
    std::size_t leaves{0};  ///< Листьев
    std::size_t branches{0}; ///< Внутренних узлов
    std::size_t depth{0};   ///< Высота: лист имеет глубину 0
};

/// Обходит дерево и считает число узлов, листьев и высоту.
template <class T>
inline TreeStats compute_tree_stats(const Node<T> &node)
{
    return std::visit(
        [](const auto &arg) -> TreeStats {
            using Arg = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<Arg, Leaf<T>>)
            {
                return TreeStats{.nodes = 1, .leaves = 1, .branches = 0, .depth = 0};
            }
            else
            {
                TreeStats left = compute_tree_stats(arg->left());
                TreeStats right = compute_tree_stats(arg->right());
                return TreeStats{
                    .nodes = 1 + left.nodes + right.nodes,
                    .leaves = left.leaves + right.leaves,
                    .branches = 1 + left.branches + right.branches,
                    .depth = 1 + std::max(left.depth, right.depth),
                };
            }
        },
        node);
}

/// Собирает разбиения внутренних узлов в обходе preorder.
template <class T>
inline void collect_splits(const Node<T> &node, std::vector<Split> &out)
{
    std::visit(
        [&](const auto &arg) {
            using Arg = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<Arg, Leaf<T>>)
            {
                return;
            }
            else
            {
                out.push_back(arg->split());
                collect_splits(arg->left(), out);
                collect_splits(arg->right(), out);
            }
        },
        node);
}
