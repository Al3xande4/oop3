#include <vector>

#include "models/api/errors.hpp"
#include "models/api/model.hpp"
#include "tree_node.hpp"

class ICartModel
{
protected:
    size_t m_max_depth;
    size_t m_min_samples_split;
    size_t m_min_samples_leaf;

    size_t m_fit_feature_size{0}; ///< Число столбцов обучающей матрицы
    bool m_fitted{false};         ///< Признак успешного fit

public:
    ICartModel(size_t max_depth, size_t min_samples_split, size_t min_samples_leaf)
        : m_max_depth{max_depth}, m_min_samples_split{min_samples_split}, m_min_samples_leaf{min_samples_leaf}
    {
        if (min_samples_split < 2)
            throw HyperParameterError("CardModel: min samples split must be greater than 1");

        if (min_samples_leaf < 1)
            throw HyperParameterError("CardModel: min samples leaf must be greater than 0");
    }
};

class CartRegressor : public IRegressor, public ICartModel
{
private:
    Node<double> m_root{nullptr};

public:
    using ICartModel::ICartModel;

    void fit(const Matrix &features, std::span<const double> targets) override;

    std::vector<double> predict(const Matrix &features) const override;
};

class CartClassifier : public IClassifier, public ICartModel
{
private:
    Node<std::vector<double>> m_root{nullptr};
    std::vector<double> m_classes;

public:
    using ICartModel::ICartModel;

    void fit(const Matrix &features, std::span<const double> targets) override;

    std::vector<double> predict(const Matrix &features) const override;

    Matrix predict_proba(const Matrix &features) const override;

    std::span<const double> classes() const override
    {
        if (!m_fitted)
            throw NotFitted("CartClassifier.classes: called before fit.");
        return m_classes;
    }
};