#include "dataset/dataset.hpp"
#include "test_util.hpp"

namespace
{
template <typename Fn>
bool throws_runtime_error_with(Fn &&fn, const std::string &needle)
{
    try
    {
        fn();
    }
    catch (const std::runtime_error &error)
    {
        const std::string message = error.what();
        if (message.find(needle) == std::string::npos)
        {
            std::printf("FAIL: message \"%s\" does not contain \"%s\"\n", message.c_str(), needle.c_str());
            ++g_failures;
            return true;
        }
        return true;
    }
    catch (const std::exception &error)
    {
        std::printf("FAIL: wrong exception type: %s\n", error.what());
        ++g_failures;
        return false;
    }
    std::printf("FAIL: no exception thrown (expected message with \"%s\")\n", needle.c_str());
    ++g_failures;
    return false;
}
} // namespace

int main()
{
    const std::filesystem::path short_row = write_temp_file(
        "lab1_inconsistent_short.csv",
        "a,b,c\n"
        "1,2,3\n"
        "4,5\n");
    throws_runtime_error_with([&] { Dataset::load_dataset(short_row); }, "At row: 2");
    std::filesystem::remove(short_row);

    const std::filesystem::path long_row = write_temp_file(
        "lab1_inconsistent_long.csv",
        "a,b\n"
        "1,2,3\n");
    throws_runtime_error_with([&] { Dataset::load_dataset(long_row); }, "At row: 1");
    std::filesystem::remove(long_row);

    const std::filesystem::path duplicate_header = write_temp_file(
        "lab1_duplicate_header.csv",
        "a,a\n"
        "1,2\n");
    throws_runtime_error_with([&] { Dataset::load_dataset(duplicate_header); }, "same name");
    std::filesystem::remove(duplicate_header);

    const std::filesystem::path empty_header = write_temp_file("lab1_empty_header.csv", "");
    throws_runtime_error_with([&] { Dataset::load_dataset(empty_header); }, "Empty header");
    std::filesystem::remove(empty_header);

    const std::filesystem::path mixed_type = write_temp_file(
        "lab1_mixed_type.csv",
        "a,b\n"
        "1,2\n"
        "x,3\n");
    throws_runtime_error_with([&] { Dataset::load_dataset(mixed_type); }, "At row: 2");
    std::filesystem::remove(mixed_type);

    throws_runtime_error_with(
        [] { Dataset::load_dataset(std::filesystem::temp_directory_path() / "lab1_does_not_exist.csv"); },
        "Cannot find a file");

    return finish("inconsistent_csv");
}
