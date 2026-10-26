#pragma once

#include <vector>
#include <optional>
#include <variant>
#include <string>

struct NumericColumn
{
    std::vector<std::optional<double>> values;

    auto begin() const { return values.begin(); }
    auto end() const { return values.end(); }
};

struct CategoricalColumn
{
    std::vector<std::optional<std::string>> values;

    auto begin() const { return values.begin(); }
    auto end() const { return values.end(); }
};

using Column = std::variant<NumericColumn, CategoricalColumn>;