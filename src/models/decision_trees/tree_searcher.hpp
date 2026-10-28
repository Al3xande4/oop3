#pragma once

#include <variant>

#include "dataset/matrix_slice.hpp"
#include "tree_node.hpp"

template <class... Ts>
struct Overloaded : Ts...
{
    using Ts::operator()...;
};

template <class... Ts>
Overloaded(Ts...) -> Overloaded<Ts...>;

class TreeSearcher
{
public:
    template <class T>
    Leaf<T> search(const Node<T> &node, const MatrixSlice<const double> &row) const
    {
        return std::visit(
            Overloaded{
                [](const Leaf<T> &leaf) { return leaf; },
                [&](const std::unique_ptr<Branch<T>> &branch)
                {
                    Split split = branch->split();
                    if (row(split.feature) <= split.threshold)
                        return search(branch->left(), row);

                    return search(branch->right(), row);
                },
            },
            node
        );
    }
};
