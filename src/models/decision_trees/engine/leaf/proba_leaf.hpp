#pragma once

#include <span>
#include <vector>

class ProbaLeaf
{
private:
    size_t m_classses_count{0};

public:
    ProbaLeaf(size_t classes_count) : m_classses_count{classes_count} {};

    using value_type = std::vector<double>;

    value_type build(
        std::span<const double> targets,
        std::span<const size_t> object_ids,
        std::span<const size_t> counts
    ) const;
};