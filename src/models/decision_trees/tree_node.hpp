#pragma once

#include <memory>
#include <variant>

#include "split.hpp"

template <class T>
class Leaf
{
private:
    T m_value;

public:
    Leaf(const T &value) : m_value(value){};
    Leaf(T &&value) : m_value(std::move(value)){};

    const T &get_value() const { return m_value; };
};

template <class T>
class Branch;

template <class T>
using Node = std::variant<Leaf<T>, std::unique_ptr<Branch<T>>>;

template <class T>
class Branch
{
private:
    Node<T> m_left;
    Node<T> m_right;
    Split m_split;

public:
    Branch(Node<T> left, Node<T> right, Split split)
        : m_left(std::move(left)), m_right(std::move(right)), m_split(std::move(split)){};

    const Node<T> &left() const { return m_left; };
    const Node<T> &right() const { return m_right; };
    Split split() const { return m_split; };
};
