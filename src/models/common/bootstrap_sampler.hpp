#pragma once

#include <cstddef>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include "models/api/errors.hpp"

struct BootstrapResult
{
    std::vector<size_t> object_ids;
    std::vector<size_t> counts;
};

class BootstrapSampler
{
private:
    bool m_bootstrap;
    std::optional<size_t> m_max_samples;

public:
    BootstrapSampler(bool bootstrap = true, std::optional<size_t> max_samples = std::nullopt)
        : m_bootstrap(bootstrap), m_max_samples(max_samples)
    {
        if (!m_bootstrap && m_max_samples.has_value())
            throw HyperParameterError("max_samples must not be provided when bootstrap is false");
    };

    BootstrapResult sample(size_t rows, std::mt19937 &gen) const
    {
        if (!m_bootstrap)
        {
            BootstrapResult res;
            res.counts.assign(rows, 1);
            res.object_ids.resize(rows);
            std::iota(res.object_ids.begin(), res.object_ids.end(), 0ULL);
            return res;
        }
        size_t m = m_max_samples ? m_max_samples.value() : rows;
        if (m > rows || m < 1)
            throw HyperParameterError("Max samples must be in range [1; " + std::to_string(rows) + "]");

        BootstrapResult res;
        res.counts.resize(rows);
        std::uniform_int_distribution<size_t> dist(0, rows - 1);
        while (m-- > 0)
        {
            size_t index = dist(gen);
            res.counts[index]++;
        }

        for (size_t i = 0; i < rows; i++)
            if (res.counts[i] > 0)
                res.object_ids.push_back(i);

        return res;
    }
};