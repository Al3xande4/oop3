#pragma once

#include <memory>
#include <variant>

#include "dataset/matrix_slice.hpp"
#include "tree_node.hpp"

class TreeSearcher
{
public:
    template <class T>
    const Leaf<T> &search(const Node<T> &node, const MatrixSlice<const double> &row) const
    {
        const Node<T> *curr = &node;
        while (std::holds_alternative<std::unique_ptr<Branch<T>>>(*curr))
        {
            const auto &branch = std::get<std::unique_ptr<Branch<T>>>(*curr);
            const Split &split = branch->split();
            if (row(split.feature) <= split.threshold)
                curr = &branch->left();
            else
                curr = &branch->right();
        }
        return std::get<Leaf<T>>(*curr);
    }
};
