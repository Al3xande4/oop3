#pragma once

#include <ranges>
#include "column.hpp"
#include "feature_info.hpp"
#include "imputation.hpp"
#include "transformed.hpp"
#include "category_encoding.hpp"
#include "scaling.hpp"

class Dataset
{
private:
    size_t m_count;
    std::vector<std::string> m_column_names;
    std::unordered_map<std::string, Column> m_columns;

    std::string m_target_name;
    Column m_target_column;

public:
    DatasetInfo analyse() const;

    void impute(const DatasetInfo &, const NumericImputationStrategy &);

    TransformedDataset transform(
        const DatasetInfo &info,
        const ScalingStrategy &scaling,
        CategoryEncoding encoding,
        Layout layout) const;

    static Dataset load_dataset(std::filesystem::path, std::optional<size_t> = std::nullopt);
};
