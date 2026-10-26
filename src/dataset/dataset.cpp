#include "dataset.hpp"
#include "config.hpp"
#include "helpers.hpp"

#include <set>
#include <string>
#include <algorithm>
#include <charconv>
#include <fstream>

std::vector<std::string> parse_header(std::string_view header)
{
    std::vector<std::string> result;
    std::set<std::string> names;

    for (auto s : header | std::views::split(datasetconfig::separator) |
                      std::views::transform([](auto s)
                                            { return trim(std::string_view(s.begin(), s.end())); }))
    {
        if (s.empty())
            throw std::runtime_error{"Empty header field."};

        if (names.find(s) != names.end())
            throw std::runtime_error{"Several columns with a same name"};

        result.push_back(s);
        names.insert(s);
    }

    return result;
}

using Cell = std::optional<std::variant<double, std::string>>;
using Row = std::vector<Cell>;
Row parse_row(std::string_view row)
{
    Row result;
    for (auto token : row | std::views::split(datasetconfig::separator) | std::views::transform([](auto s)
                                                                                                { return trim(std::string_view(s.begin(), s.end())); }))
    {
        if (std::ranges::equal(token, datasetconfig::missing_field))
        {
            result.push_back(std::nullopt);
            continue;
        }

        double value;
        auto [ptr, ec] = std::from_chars(token.data(), token.data() + token.size(), value);
        if (ptr != token.data() + token.size())
        {
            result.push_back(token);
            continue;
        }
        result.push_back(value);
    }

    return result;
}

Dataset Dataset::load_dataset(std::filesystem::path path, std::optional<size_t> target_column)
{
    Dataset dataset;

    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error{"Cannot find a file: " + std::string(path)};

    std::string header;
    std::getline(file, header);
    if (header.size() == 0)
        throw std::runtime_error{"Empty header."};

    std::vector<std::string> column_names = parse_header(header);
    if (!target_column)
        target_column = column_names.size() - 1;

    std::string row;
    size_t row_index = 0;

    std::vector<bool> determined_columns(column_names.size(), false);

    while (std::getline(file, row))
    {
        Row r = parse_row(row);
        if (r.size() != column_names.size())
            throw std::runtime_error{
                std::string{"Size of header and row are not compatible."
                            " At row: "} +
                std::to_string(row_index + 1)};

        for (size_t i = 0; i < r.size(); i++)
        {
            Column &col = dataset.m_columns[column_names[i]];
            Cell c = r[i];
            if (c && determined_columns[i] && std::holds_alternative<NumericColumn>(col) ^ std::holds_alternative<double>(*c))
                throw std::runtime_error{
                    std::string{"Type of fields are incompatible. "
                                " At row: "} +
                    std::to_string(row_index + 1) + std::string{", at column: "} + column_names[i]};

            if (!c)
            {
                std::visit([](auto &vec)
                           { vec.values.push_back(std::nullopt); }, col);
                continue;
            }

            if (determined_columns[i])
            {
                if (std::holds_alternative<NumericColumn>(col))
                    std::get<NumericColumn>(col).values.push_back(std::get<double>(*c));
                else
                    std::get<CategoricalColumn>(col).values.push_back(std::get<std::string>(*c));

                continue;
            }

            if (std::holds_alternative<double>(*c))
            {
                std::vector<std::optional<double>> prev(row_index + 1, std::nullopt);
                prev[row_index] = std::get<double>(*c);
                col = NumericColumn{std::move(prev)};
            }
            else
            {
                std::vector<std::optional<std::string>> prev(row_index + 1, std::nullopt);
                prev[row_index] = std::get<std::string>(*c);
                col = CategoricalColumn{std::move(prev)};
            }
            determined_columns[i] = true;
        }

        row_index++;
    }

    for (size_t i = 0; i < determined_columns.size(); i++)
        if (!determined_columns[i])
            throw std::runtime_error{
                "Couldn't deduce type of column: " + column_names[i]};

    std::swap(column_names[*target_column], column_names.back());
    dataset.m_target_name = column_names.back();
    dataset.m_target_column = dataset.m_columns[dataset.m_target_name];

    dataset.m_columns.erase(dataset.m_target_name);
    column_names.pop_back();

    for (auto &[name, column] : dataset.m_columns)
    {
        if (std::holds_alternative<CategoricalColumn>(column))
            continue;

        std::set<double> unique;
        for (auto elem : std::get<NumericColumn>(column))
            if (elem)
                unique.insert(*elem);

        if (unique.size() < datasetconfig::numeric_column_threshold)
        {
            CategoricalColumn categorical_column;
            for (auto val : std::get<NumericColumn>(column))
            {
                if (val)
                    categorical_column.values.push_back(std::to_string(*val));
                else
                    categorical_column.values.push_back(std::nullopt);
            }
            column = std::move(categorical_column);
        }
    }

    dataset.m_column_names = std::move(column_names);
    dataset.m_count = row_index;
    return dataset;
}

DatasetInfo Dataset::analyse() const
{
    DatasetInfo info;
    info.column_names = m_column_names;
    info.column_names.push_back(m_target_name);
    info.total_count = m_count;

    for (const auto &name : m_column_names)
    {
        const Column &column = m_columns.at(name);
        if (std::holds_alternative<NumericColumn>(column))
            info.feature_info[name] = analyze_numeric_column(std::get<NumericColumn>(column));
        else
            info.feature_info[name] = analyze_categorical_column(std::get<CategoricalColumn>(column));
    }

    if (std::holds_alternative<NumericColumn>(m_target_column))
        info.feature_info[m_target_name] = analyze_numeric_column(std::get<NumericColumn>(m_target_column));
    else
        info.feature_info[m_target_name] = analyze_categorical_column(std::get<CategoricalColumn>(m_target_column));

    return info;
}

void Dataset::impute(const DatasetInfo &info, const NumericImputationStrategy &strategy)
{
    for (auto &[name, column] : m_columns)
    {
        if (std::holds_alternative<NumericColumn>(column))
        {
            for (auto &value : std::get<NumericColumn>(column).values)
                if (!value)
                    value = strategy.impute(std::get<NumericFeatureInfo>(info.feature_info.at(name)));
        }
        else
        {
            for (auto &value : std::get<CategoricalColumn>(column).values)
                if (!value)
                    value = datasetconfig::category_imputation;
        }
    }
}

TransformedDataset Dataset::transform(
    const DatasetInfo &info,
    const ScalingStrategy &scaling,
    CategoryEncoding encoding,
    Layout layout) const
{
    size_t cols = m_column_names.size();
    size_t rows = info.total_count;

    size_t additional = 0;
    if (encoding == CategoryEncoding::ONE_HOT)
    {
        for (const auto &[name, feature_info] : info.feature_info)
        {
            if (std::holds_alternative<CategoricalFeatureInfo>(feature_info))
                additional += std::get<CategoricalFeatureInfo>(feature_info).categories.size() - 1;
        }
    }

    Matrix matrix = Matrix{rows, cols + additional, layout};

    std::vector<std::string> names;

    for (size_t col = 0; col < cols; col++)
    {
        const std::string &name = m_column_names[col];
        const Column &column = m_columns.at(name);

        if (std::holds_alternative<CategoricalColumn>(column) && encoding == CategoryEncoding::ONE_HOT)
        {
            const auto &categories = std::get<CategoricalFeatureInfo>(info.feature_info.at(name)).categories;
            for (size_t idx = 0; idx < categories.size(); idx++)
                names.push_back(name + "_" + categories[idx]);

            continue;
        }

        names.push_back(name);
    }

    for (size_t i = 0; i < rows; i++)
    {
        size_t offset = 0;
        for (size_t j = 0; j < cols; j++)
        {
            const std::string &name = m_column_names[j];
            const Column &column = m_columns.at(name);

            if (std::holds_alternative<NumericColumn>(column))
            {
                auto value = std::get<NumericColumn>(column).values[i];
                matrix[{i, j + offset}] = scaling.scale(*value, std::get<NumericFeatureInfo>(info.feature_info.at(name)));

                continue;
            }

            auto category = std::get<CategoricalColumn>(column).values[i];
            const auto &categories = std::get<CategoricalFeatureInfo>(info.feature_info.at(name)).categories;

            if (encoding == CategoryEncoding::ONE_HOT)
            {
                for (size_t idx = 0; idx < categories.size(); idx++)
                    matrix[{i, j + offset + idx}] = *category == categories[idx];

                offset += categories.size() - 1;

                continue;
            }

            auto it = std::ranges::find(categories, *category);
            size_t index = it - categories.begin();

            matrix[{i, j}] = index;
        }
    }

    TransformedDataset tds;
    tds.matrix = std::move(matrix);
    tds.feature_names = std::move(names);
    tds.target_name = m_target_name;

    for (size_t i = 0; i < rows; i++)
    {
        if (std::holds_alternative<NumericColumn>(m_target_column))
        {
            auto value = std::get<NumericColumn>(m_target_column).values[i];
            tds.target_values.push_back(*value);

            continue;
        }

        auto category = std::get<CategoricalColumn>(m_target_column).values[i];
        const auto &categories = std::get<CategoricalFeatureInfo>(info.feature_info.at(m_target_name)).categories;

        auto it = std::ranges::find(categories, *category);
        size_t index = it - categories.begin();

        tds.target_values.push_back(index);
    }

    return tds;
}