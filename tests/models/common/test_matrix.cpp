#include "dataset/matrix.hpp"
#include "test_util.hpp"

#include <stdexcept>

void test_matrix_at_out_of_bounds_throws()
{
    Matrix m(3, 2);
    const Matrix &cm = m;

    m.at(0)[0] = 1.5;
    m.at(2)[1] = 3.5;
    CHECK_NEAR(cm.at(0)[0], 1.5, 1e-12);
    CHECK_NEAR(cm.at(2)[1], 3.5, 1e-12);

    CHECK_THROWS_AS(m.at(3), std::out_of_range);
    CHECK_THROWS_AS(m.at(999), std::out_of_range);
    CHECK_THROWS_AS(cm.at(3), std::out_of_range);
    CHECK_THROWS_AS(cm.at(999), std::out_of_range);
}

int main()
{
    test_matrix_at_out_of_bounds_throws();

    return finish("Matrix tests");
}