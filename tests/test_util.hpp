#pragma once

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>

inline int g_failures = 0;

#define CHECK(cond)                                                     \
    do                                                                  \
    {                                                                   \
        if (!(cond))                                                    \
        {                                                               \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
            ++g_failures;                                               \
        }                                                               \
    } while (0)

#define CHECK_NEAR(a, b, eps)                                                    \
    do                                                                           \
    {                                                                            \
        double check_near_a = (a);                                               \
        double check_near_b = (b);                                               \
        if (!(std::fabs(check_near_a - check_near_b) <= (eps)))                  \
        {                                                                        \
            std::printf("FAIL %s:%d: %s = %.17g, expected %s = %.17g\n",         \
                        __FILE__, __LINE__, #a, check_near_a, #b, check_near_b); \
            ++g_failures;                                                        \
        }                                                                        \
    } while (0)

#define CHECK_THROWS(code)  \
    do                      \
    {                       \
        bool threw = false; \
        try                 \
        {                   \
            code            \
        }                   \
        catch (...)         \
        {                   \
            threw = true;   \
        }                   \
        CHECK(threw);       \
    } while (0)

inline std::filesystem::path
write_temp_file(std::string_view name, std::string_view content)
{
    const std::filesystem::path path = std::filesystem::temp_directory_path() / name;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out << content;
    out.close();
    return path;
}

inline int finish(const char *test_name)
{
    if (g_failures == 0)
    {
        std::printf("PASS: %s\n", test_name);
        return 0;
    }
    std::printf("FAILED: %s (%d check(s))\n", test_name, g_failures);
    return 1;
}
