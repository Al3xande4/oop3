#pragma once

#include <algorithm>
#include <memory>
#include <variant>

#include "models/common/overloaded.hpp"
#include "models/decision_trees/engine/tree_node.hpp"

struct TreeInfo
{
    size_t leaves;
    size_t branches;
    size_t depth;
};

class TreeInspector
{
public:
    template <class T>
    TreeInfo info(const Node<T> &node) const
    {
        return std::visit(
            Overloaded{
                [](const Leaf<T> &) -> TreeInfo { return {.leaves = 1, .branches = 0, .depth = 0}; },
                [this](const std::unique_ptr<Branch<T>> &branch) -> TreeInfo
                {
                    TreeInfo left = info(branch->left());
                    TreeInfo right = info(branch->right());

                    return {
                        .leaves = left.leaves + right.leaves,
                        .branches = 1 + left.branches + right.branches,
                        .depth = std::max(left.depth, right.depth) + 1
                    };
                }
            },
            node
        );
    }
};