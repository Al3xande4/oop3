#pragma once

#include <string>

namespace datasetconfig
{
    inline constexpr char separator = ',';
    inline constexpr std::string missing_field = "";

    inline size_t histogram_intervals = 5;

    inline constexpr size_t numeric_column_threshold = 8;

    inline constexpr std::string category_imputation = "__MISSING__";
};