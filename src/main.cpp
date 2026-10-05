#include "dataset/dataset.hpp"
#include "dataset/config.hpp"
#include "dataset/imputation.hpp"
#include <cxxopts.hpp>
#include <iostream>
#include <filesystem>

void print_dataset_summary(const DatasetInfo &info)
{
    printf("Objects: %zu\n", info.total_count);
    printf("Features: %zu\n\n", info.column_names.size());

    printf("%5s | %-25.22s | %-20.15s | %15s\n", "#", "Feature", "Type", "Missing");
    for (int j = 0; j < 75; j++)
        printf("-");
    printf("\n");

    size_t total_missing = 0;
    size_t count_missing_features = 0;
    size_t i = 0;
    for (const auto &name : info.column_names)
    {
        const FeatureInfo &feature = info.feature_info.at(name);
        std::string type;
        size_t missing;

        if (std::holds_alternative<NumericFeatureInfo>(feature))
        {
            type = "numeric";
            missing = std::get<NumericFeatureInfo>(feature).missing_count;
        }
        else
        {
            type = "categorical";
            missing = std::get<CategoricalFeatureInfo>(feature).missing_count;
        }
        total_missing += missing;
        std::string miss_str = std::to_string(missing);
        if (missing != 0)
        {
            double percantage = 100 * missing * 1. / info.total_count;
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(2) << percantage;
            miss_str += " (" + oss.str() + "%)";
            count_missing_features++;
        }

        printf("%5zu | %-25.22s | %-20.15s | %15s\n", i, name.c_str(), type.c_str(), miss_str.c_str());

        ++i;
    }
    printf("Total: %zu missing values in %zu distinct features.", total_missing, count_missing_features);
}

void print_numeric_feature_statistics(size_t no, std::string_view name, const NumericFeatureInfo &info)
{
    printf("Feature %zu: %10s (numeric)\n\n", no, name.data());

    printf("%3sMissing values: %zu\n", "", info.missing_count);
    printf("%3smin: %.3lf %3s max: %.3lf\n", "", info.min, "", info.max);
    printf("%3smean: %.3lf %3s var: %.3lf\n\n", "", info.mean, "", info.variance);

    printf("%3sq05: %.3lf\n", "", info.q05);
    printf("%3sq25: %.3lf\n", "", info.q25);
    printf("%3sq50: %.3lf\n", "", info.median);
    printf("%3sq75: %.3lf\n", "", info.q75);
    printf("%3sq95: %.3lf\n\n", "", info.q95);

    const size_t histogram_width = 100;
    for (const auto &[range, freq] : info.histogram)
    {
        const auto &[start, end] = range;

        char buffer[100];
        sprintf(buffer, "[%.1lf, %.1lf)", start, end);
        printf("%3s%-20s", "", buffer);
        for (size_t i = 0; i < histogram_width; i++)
            if (i <= histogram_width * freq / info.max_histogram_freq)
                putchar('#');
            else
                putchar(' ');
        printf("%3s%zu\n", "", freq);
    }
}

void print_categorical_feature_statistics(size_t no, std::string_view name, const CategoricalFeatureInfo &info)
{
    printf("Feature %zu: %10s (categorical)\n\n", no, name.data());

    printf("%3sMissing values: %zu\n", "", info.missing_count);
    printf("%3sCategories list:", "");
    for (auto category : info.categories)
        printf(" %s;", category.c_str());
    putchar('\n');

    const size_t histogram_width = 100;
    for (const auto &[category, freq] : info.frequencies)
    {
        printf("%3s%-20s", "", category.c_str());
        for (size_t i = 0; i < histogram_width; i++)
            if (i <= histogram_width * freq / info.max_freq)
                putchar('#');
            else
                putchar(' ');
        printf("%3s%zu\n", "", freq);
    }
}

void print_statistics(const DatasetInfo &info)
{
    std::cout << "Database Statictics.\n\n";
    std::string s;

    while (true)
    {
        std::cout << "Print the column number or column name of the feature(or s/summary to print summary or q/quit to exit statistics): \n";
        std::getline(std::cin, s);

        if (s == "q" || s == "quit")
            break;
        if (s == "s" || s == "summary")
        {
            print_dataset_summary(info);
            continue;
        }

        long long value = -1;
        auto [ptr, err] = std::from_chars(s.data(), s.data() + s.size(), value);
        if (ptr != s.data() + s.size())
        {
            if (info.feature_info.contains(s))
            {
                size_t feat_no;
                for (size_t i = 0; i < info.column_names.size(); i++)
                    if (info.column_names[i] == s)
                    {
                        feat_no = i;
                        break;
                    }

                if (std::holds_alternative<NumericFeatureInfo>(info.feature_info.at(s)))
                    print_numeric_feature_statistics(feat_no, s, std::get<NumericFeatureInfo>(info.feature_info.at(s)));
                else
                    print_categorical_feature_statistics(feat_no, s, std::get<CategoricalFeatureInfo>(info.feature_info.at(s)));
            }
            else
                std::cout << "No such feature name. Try again\n";
        }
        else
        {
            if (value >= 0 && (size_t)value < info.column_names.size())
            {
                if (std::holds_alternative<NumericFeatureInfo>(info.feature_info.at(info.column_names[value])))
                    print_numeric_feature_statistics(value, info.column_names[value], std::get<NumericFeatureInfo>(info.feature_info.at(info.column_names[value])));
                else
                    print_categorical_feature_statistics(value, info.column_names[value], std::get<CategoricalFeatureInfo>(info.feature_info.at(info.column_names[value])));
            }
            else
                std::cout << "No such feature index, only from 0 to " << info.column_names.size() << " are allowed. Try again\n";
        }
    }
}

int main(int argc, char *argv[])
{
    cxxopts::Options options("ds_load", "Load dataset from file and transform it into numeric matrix.");
    options.add_options()("help,h", "Help")("csv,dataset,sourcepath,f", "Path to the dataset", cxxopts::value<std::filesystem::path>())("dest,output,o", "Path to the file with processed data", cxxopts::value<std::filesystem::path>())("histogram", "Number of histogram intervals", cxxopts::value<size_t>())("fill,impute,i", "Imputation strategy", cxxopts::value<std::string>()->default_value("mean"))("s,scale", "Scaling strategy", cxxopts::value<std::string>()->default_value("minmax"))("encode,e", "Encoding strategy", cxxopts::value<std::string>()->default_value("onehot"));
    auto result = options.parse(argc, argv);

    if (result.count("help"))
    {
        std::cout << options.help() << std::endl;
        exit(0);
    }

    if (result.count("dataset") == 0)
    {
        std::cout << "You must specify input file, to see options use --help\n";
        exit(0);
    }

    std::filesystem::path csvPath = result["dataset"].as<std::filesystem::path>();

    NumericImputationStrategy *imputation;
    ScalingStrategy *scaling;
    CategoryEncoding encoding;

    std::string impOpt = result["impute"].as<std::string>();
    if (impOpt == "mean")
        imputation = new MeanImputation();
    else if (impOpt == "zero")
        imputation = new ZeroImputation();
    else if (impOpt == "median")
        imputation = new MedianImputation();
    else
    {
        std::cout << "Unknown imputation strategy, please use one of: mean, zero, median. Using median as default.\n";
        imputation = new MedianImputation();
    }

    std::string scalingOpt = result["scale"].as<std::string>();
    if (scalingOpt == "minmax")
        scaling = new MinMaxScaling();
    else if (scalingOpt == "standart")
        scaling = new StandartScaling();
    else if (scalingOpt == "robust")
        scaling = new RobustScaling();
    else
    {
        std::cout << "Unknown scaling strategy, please use one of: minmax, standart, robust. Using minmax as default.\n";
        scaling = new MinMaxScaling();
    }

    std::string encodingOpt = result["encode"].as<std::string>();
    if (encodingOpt == "onehot")
        encoding = CategoryEncoding::ONE_HOT;
    else if (encodingOpt == "ordinal")
        encoding = CategoryEncoding::ORDINAL;
    else
    {
        std::cout << "Unknown encoding strategy, please use one of: onehot, ordinal. Using onehot as default.\n";
        encoding = CategoryEncoding::ONE_HOT;
    }

    if (result.count("histogram"))
    {
        size_t intervals = result["histogram"].as<size_t>();
        datasetconfig::histogram_intervals = intervals;
    }

    try
    {
        Dataset ds = Dataset::load_dataset(csvPath);

        DatasetInfo info = ds.analyse();

        print_dataset_summary(info);
        print_statistics(info);

        std::cout << "Save the transformed dataset?(y/yes, n/no): ";
        std::string s;
        std::getline(std::cin, s);
        if (s == "no" || s == "n")
            return 0;

        ds.impute(info, *imputation);
        TransformedDataset tds = ds.transform(info, *scaling, encoding, Layout::RowMajor);

        if (result.count("dest") == 0)
        {
            std::cout << "Didn't specify the path to save processed matrix.\n";
            return 1;
        }
        std::filesystem::path matrix_path = result["dest"].as<std::filesystem::path>();
        tds.save(matrix_path.replace_extension("bin"));

        std::ifstream csv_file{csvPath, std::ios::binary | std::ios::ate};
        size_t csv_bytes = csv_file.tellg();

        std::ifstream matrix_file{matrix_path.replace_extension("bin"), std::ios::binary | std::ios::ate};
        size_t matrix_bytes = matrix_file.tellg();

        std::cout << "CSV size: " << csv_bytes << "\n";
        std::cout << "Binary size: " << matrix_bytes << "\n";
    }
    catch (const std::runtime_error &error)
    {
        std::cout << error.what();
        return 1;
    }

    return 0;
}
