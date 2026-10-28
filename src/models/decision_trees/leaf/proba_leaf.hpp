#pragma once

#include <span>
#include <vector>

class ProbaLeaf
{
private:
    std::span<const double> m_classes;

public:
    ProbaLeaf(std::span<const double> classes) : m_classes{classes} {};

    using value_type = std::vector<double>;

    value_type build(std::span<const double> targets, std::span<const size_t> object_ids) const;
};