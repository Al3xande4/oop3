#pragma once

#include <cstddef>
#include <span>
#include <stdexcept>
#include <string>

class NonContiguousSliceError : public std::logic_error
{
public:
    NonContiguousSliceError(const std::string &msg) : std::logic_error(msg){};
};

enum class SliceAxis
{
    Row,
    Column
};

template <class T>
class MatrixSlice
{
public:
    MatrixSlice() noexcept = default;
    MatrixSlice(T *data, std::size_t size, std::size_t stride, SliceAxis axis);

    std::size_t size() const noexcept;
    bool empty() const noexcept;
    std::size_t stride() const noexcept;
    SliceAxis axis() const noexcept;
    bool is_contiguous() const noexcept;
    T *data() const noexcept;

    T &operator()(std::size_t index) const noexcept;
    T &at(std::size_t index) const;
    std::span<T> operator*() const;
    std::span<T> as_span() const;

    MatrixSlice<const T> as_const() const noexcept;

private:
    T *m_data = nullptr;
    std::size_t m_size = 0;
    std::size_t m_stride = 1;
    SliceAxis m_axis = SliceAxis::Row;
};

template <class T>
MatrixSlice<T>::MatrixSlice(T *data, std::size_t size, std::size_t stride, SliceAxis axis)
{
    if (size != 0 && stride == 0)
        throw std::invalid_argument("MatrixSlice: stride must be greater than 0");

    if (size != 0 && data == nullptr)
        throw std::invalid_argument("MatrixSlice: data is nullptr, but size is not 0");

    m_size = size;
    m_stride = stride;
    m_axis = axis;

    m_data = data;
    if (size == 0)
        m_data = nullptr;
}

template <class T>
std::size_t MatrixSlice<T>::size() const noexcept
{
    return m_size;
}

template <class T>
bool MatrixSlice<T>::empty() const noexcept
{
    return m_data == nullptr || m_size == 0;
}

template <class T>
std::size_t MatrixSlice<T>::stride() const noexcept
{
    return m_stride;
}

template <class T>
SliceAxis MatrixSlice<T>::axis() const noexcept
{
    return m_axis;
}

template <class T>
bool MatrixSlice<T>::is_contiguous() const noexcept
{
    return m_stride == 1;
}

template <class T>
T *MatrixSlice<T>::data() const noexcept
{
    return m_data;
}

template <class T>
T &MatrixSlice<T>::operator()(std::size_t index) const noexcept
{
    return m_data[index * m_stride];
}

template <class T>
T &MatrixSlice<T>::at(std::size_t index) const
{
    if (index >= m_size)
        throw std::out_of_range(
            "MatrixSlice.at: index out of range " + std::to_string(index * m_stride) +
            " size: " + std::to_string(m_size)
        );

    return m_data[index * m_stride];
}

template <class T>
std::span<T> MatrixSlice<T>::operator*() const
{
    if (m_stride == 1 || m_size <= 1)
        return std::span(m_data, m_size);

    throw NonContiguousSliceError("MatrixSlice.operator*: non contiguous");
}

template <class T>
std::span<T> MatrixSlice<T>::as_span() const
{
    return this->operator*();
}

template <class T>
MatrixSlice<const T> MatrixSlice<T>::as_const() const noexcept
{
    return MatrixSlice<const T>(m_data, m_size, m_stride, m_axis);
}