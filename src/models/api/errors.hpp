#pragma once

#include <stdexcept>
#include <string>

class CoreError : public std::runtime_error
{
public:
    CoreError(const std::string &message) : std::runtime_error{message} {};
};

class IndexOutOfBounds : public CoreError
{
public:
    IndexOutOfBounds(const std::string &message) : CoreError{message} {};
};

class DimensionMismatch : public CoreError
{
public:
    DimensionMismatch(const std::string &message) : CoreError{message} {};
};

class EmptyData : public CoreError
{
public:
    EmptyData(const std::string &message) : CoreError{message} {};
};

class NotFitted : public CoreError
{
public:
    NotFitted(const std::string &message) : CoreError{message} {};
};

class HyperParameterError : public CoreError
{
public:
    HyperParameterError(const std::string &message) : CoreError{message} {};
};
