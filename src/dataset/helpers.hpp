#pragma once

#include <ranges>
#include <string>

inline std::string trim(std::string_view s)
{
    auto start = s.find_first_not_of(" \t\r\n");
    auto end = s.find_last_not_of(" \t\r\n");
    return (start == std::string_view::npos) ? "" : std::string(s.substr(start, end - start + 1));
}