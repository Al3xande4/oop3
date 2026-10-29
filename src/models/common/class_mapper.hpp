#pragma once

#include <algorithm>
#include <span>
#include <vector>

class ClassMapper
{
private:
    std::vector<double> m_classes;

public:
    ClassMapper() = default;

    void fit(std::span<const double> targets)
    {
        m_classes.assign(targets.begin(), targets.end());
        std::ranges::sort(m_classes);
        auto [drop_begin, drop_end] = std::ranges::unique(m_classes);
        m_classes.erase(drop_begin, drop_end);
    }

    std::vector<double> encode(std::span<const double> targets) const
    {
        std::vector<double> mapped(targets.size());
        for (std::size_t i = 0; i < targets.size(); ++i)
            mapped[i] = static_cast<double>(encode_single(targets[i]));
        return mapped;
    }

    std::size_t encode_single(double label) const
    {
        auto it = std::ranges::lower_bound(m_classes, label);
        return it - m_classes.begin();
    }

    double decode(std::size_t class_index) const { return m_classes[class_index]; }

    std::span<const double> classes() const noexcept { return m_classes; }
    std::size_t num_classes() const noexcept { return m_classes.size(); }
};