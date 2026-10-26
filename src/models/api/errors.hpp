#pragma once

#include <stdexcept>
#include <string>

class CoreError : public std::runtime_error
{
public:
    CoreError(const std::string &message) : std::runtime_error{message} {};
};

class IndexOutOfBounds : public std::out_of_range
{
public:
    IndexOutOfBounds(const std::string &message) : std::out_of_range{message} {};
};

class DimensionMismatch : public std::invalid_argument
{
public:
    DimensionMismatch(const std::string &message) : std::invalid_argument{message} {};
};

class EmptyData : public std::invalid_argument
{
public:
    EmptyData(const std::string &message) : std::invalid_argument{message} {};
};

class NotFitted : public std::logic_error
{
public:
    NotFitted(const std::string &message) : std::logic_error{message} {};
};

class HyperParameterError : public std::invalid_argument
{
public:
    HyperParameterError(const std::string &message) : std::invalid_argument{message} {};
};
