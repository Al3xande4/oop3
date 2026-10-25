#include "models/decision_trees/stump.hpp"
#include "test_util.hpp"

class TestDecisionStumpEdgeCases
{
private:
public:
    void one_column_equal()
    {
    }

    void all_columns_equal()
    {
    }

    void min_samples_blocks()
    {
    }

    void determenistic_split()
    {
    }

    void only_one_class() {}

    void run_all()
    {
        one_column_equal();
        all_columns_equal();
        min_samples_blocks();
        only_one_class();
    }
};

int main()
{
    TestDecisionStumpEdgeCases tests;

    tests.run_all();

    return finish("Decision Stump tests");
}