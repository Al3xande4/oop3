#include "test_util.hpp"
#include "models/knn/model.hpp"

int main()
{
    // Arrange: strategies under test
    UniformWeight uniform{2.5};
    InverseDistanceWeight inverse;

    // Act + Assert: uniform weight ignores the distance
    CHECK_NEAR(uniform.weight(0.), 2.5, 1e-12);
    CHECK_NEAR(uniform.weight(100.), 2.5, 1e-12);

    // Inverse distance weight follows its formula...
    CHECK_NEAR(inverse.weight(1.), 1. / (1e-9 + 1.), 1e-12);
    CHECK_NEAR(inverse.weight(10.), 1. / (1e-9 + 10.), 1e-12);

    // ...and decreases as the neighbour gets farther away
    CHECK(inverse.weight(1.) > inverse.weight(10.));

    return finish("weights");
}
