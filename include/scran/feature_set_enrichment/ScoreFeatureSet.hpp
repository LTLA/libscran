#ifndef SCRAN_SCORE_FEATURE_SET_HPP
#define SCRAN_SCORE_FEATURE_SET_HPP

#include "../utils/macros.hpp"

#include <algorithm>
#include <vector>
#include "tatami/tatami.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Dense"

#include "../dimensionality_reduction/SimplePca.hpp"
#include "../dimensionality_reduction/ResidualPca.hpp"

/**
 * @file ScoreFeatureSet.hpp
 * @brief Compute per-cell scores for a feature set.
 */

namespace scran {

/**
 * @brief Compute per-cell scores for a given feature set.
 *
 * Per-cell scores are defined as the column means of the rank-1 approximation of the input matrix for the subset of features in the set.
 * The assumption here is that the primary activity of the feature set can be quantified by the largest component of variance amongst its features.
 * (If this was not the case, one could argue that this feature set is not well-suited to capture the biology attributed to it.)
 * In effect, the rotation vector defines weights for all features in the set, focusing on genes that contribute to the primary activity.
 * This is based on the [**GSDecon**](https://github.com/JasonHackney/GSDecon) package from Jason Hackney and is implemented using `SimplePca`.
 *
 * For multi-block analyses, we extend this approach by performing the PCA on the residuals generated after centering each block in `ResidualPca`.
 * Each cell is projected onto this rotation vector, and the feature set score for each cell is then defined from the ensuing rank-1 approximation.
 * This approach avoids comparing cells between blocks and favors blocks where the feature set has greater (relative) activity.
 * In addition, blocks can be weighted so that they contribute equally to the rotation vector, regardless of the number of cells -
 * see `ResidualPca::set_weight_policy()` for more details.
 *
 * For a typical log-expression input matrix, the per-cell scores can be interpreted as the mean log-expression across all features in the set.
 * The importance of each feature is quantified by the absolute value of the weights.
 */
class ScoreFeatureSet {
public:
    /**
     * @brief Default parameters.
     */
    struct Defaults {
        /**
         * See `set_block_weight_policy()` for more details.
         */
        static constexpr WeightPolicy block_weight_policy = WeightPolicy::VARIABLE;

        /**
         * See `set_num_threads()` for more details.
         */
        static constexpr int num_threads = 1;

        /**
         * See `set_scale()` for more details.
         */
        static constexpr bool scale = false;
    };

private:
    WeightPolicy block_weight_policy = Defaults::block_weight_policy;
    int nthreads = Defaults::num_threads;
    bool scale = Defaults::scale;

public:
    /**
     * @param b Policy to use when dealing with multiple blocks containing different numbers of cells.
     * @return A reference to this `ScoreFeatureSet` instance.
     */
    ScoreFeatureSet& set_block_weight_policy(WeightPolicy b = Defaults::block_weight_policy) {
        block_weight_policy = b;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     * @return A reference to this `ScoreFeatureSet` object.
     */
    ScoreFeatureSet& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

    /**
     * @param s Whether to scale features to unit variance.
     * @return A reference to this `ScoreFeatureSet` object.
     */
    ScoreFeatureSet& set_scale(bool s = Defaults::scale) {
        scale = s;
        return *this;
    }

public:
    /**
     * @brief Feature set scoring results.
     */
    struct Results {
        /**
         * Vector of per-cell scores for this feature set.
         * This has length equal to the number of scores in the dataset.
         */
        std::vector<double> scores;

        /**
         * Vector of weights of length equal to the number of features in the set.
         * Each entry contains the weight of each successive feature in the feature set.
         * Weights may be negative.
         */
        std::vector<double> weights;
    };

private:
    /*
     * We have the first PC 'P' and the first rotation vector 'R', plus a centering vector 'C' 
     * and scaling vector 'S'. The low-rank approximation is defined as (using R syntax):
     *
     *     L = outer(R, P) * S + C 
     *       = outer(R * S, P) + C
     *
     * Remember that we want the column means of the rank-1 approximation, so:
     *
     *     colMeans(L) = mean(R * S) * P + colMeans(C)
     *
     * If scale = false, then S can be dropped from the above expression.
     */
    double compute_multiplier(const Eigen::MatrixXd& rotation, const Eigen::VectorXd& scale_v) const {
        auto first_rot = rotation.col(0);

        double multiplier = 0;
        if (scale) {
            for (Eigen::Index i = 0, end = first_rot.size(); i < end; ++i) {
                multiplier += scale_v.coeff(i) * first_rot.coeff(i);
            }
        } else {
            multiplier = std::accumulate(first_rot.begin(), first_rot.end(), 0.0);
        }

        // no need to protect against zero rows, as that should already be caught.
        return multiplier / first_rot.size();
    }

    void transfer_rotation(const Eigen::MatrixXd& rotation, std::vector<double>& weights) const {
        auto first_rot = rotation.col(0);
        weights.insert(weights.end(), first_rot.begin(), first_rot.end());
        return;
    }

public:
    /**
     * @tparam T Floating point type for the data.
     * @tparam IDX Integer type for the indices.
     * @tparam X Integer type for the feature filter.
     * @tparam Block Integer type for the block assignments.
     *
     * @param[in] mat Pointer to the input matrix.
     * Columns should contain cells while rows should contain genes.
     * @param[in] features Pointer to an array of length equal to the number of rows in `mat`, specifying the features in the set of interest.
     * Non-zero values indicate that the corresponding row is part of the feature set.
     * @param[in] block Pointer to an array of length equal to the number of columns in `mat`.
     * This should contain the blocking factor as 0-based block assignments 
     * (i.e., for `n` blocks, block identities should run from 0 to `n-1` with at least one entry for each block.)
     * If this is `NULL`, all cells are assumed to belong to the same block.
     *
     * @return A `Results` object containing the per-cell scores and per-feature weights.
     */
    template<typename T, typename IDX, typename X, typename Block>
    Results run_blocked(const tatami::Matrix<T, IDX>* mat, const X* features, const Block* block) const {
        std::shared_ptr<const tatami::Matrix<T, IDX> > subsetted = pca_utils::subset_matrix_by_features(mat, features);
        auto NR = subsetted->nrow();
        auto NC = subsetted->ncol();

        // Catching edge cases.
        if (NR == 0) {
            Results output;
            output.scores.resize(NC);
            return output;
        } else if (NR == 1) {
            Results output;
            output.weights.push_back(1);
            output.scores = subsetted->dense_row()->fetch(0);
            return output;
        } else if (NC == 0) {
            Results output;
            output.weights.resize(NR);
            return output;
        }

        Results output;

        if (block == NULL) {
            SimplePca runner;
            runner.set_rank(1);
            runner.set_scale(scale);
            runner.set_num_threads(nthreads);
            runner.set_return_rotation(true).set_return_scale(scale).set_return_center(true);

            auto temp = runner.run(subsetted.get());
            transfer_rotation(temp.rotation, output.weights);
            double multiplier = compute_multiplier(temp.rotation, temp.scale);
            double shift = std::accumulate(temp.center.begin(), temp.center.end(), 0.0) / temp.center.size();

            output.scores.resize(temp.pcs.cols());
            for (Eigen::Index c = 0, end = temp.pcs.cols(); c < end; ++c) {
                output.scores[c] = temp.pcs.coeff(0, c) * multiplier + shift;
            }

        } else {
            ResidualPca runner;
            runner.set_rank(1);
            runner.set_scale(scale);
            runner.set_num_threads(nthreads);
            runner.set_return_rotation(true).set_return_scale(scale).set_return_center(true);
            runner.set_block_weight_policy(block_weight_policy);

            auto temp = runner.run(subsetted.get(), block);
            transfer_rotation(temp.rotation, output.weights);
            double multiplier = compute_multiplier(temp.rotation, temp.scale);

            // Here, we restore the block-specific centers. Don't be tempted into
            // using MultiBatchPca, as that doesn't yield a rank-1 approximation
            // that preserves global shifts between blocks.
            Eigen::VectorXd shift = temp.center.colwise().sum() / temp.center.rows();
            output.scores.resize(temp.pcs.cols());
            for (Eigen::Index c = 0, end = temp.pcs.cols(); c < end; ++c) {
                output.scores[c] = temp.pcs.coeff(0, c) * multiplier + shift.coeff(block[c]);
            }
        }

        return output;
    }

    /**
     * An overload of `run()` where all cells are assumed to belong to the same block.
     * 
     * @tparam T Floating point type for the data.
     * @tparam IDX Integer type for the indices.
     * @tparam X Integer type for the feature filter.
     *
     * @param[in] mat Pointer to the input matrix.
     * Columns should contain cells while rows should contain genes.
     * @param[in] features Pointer to an array of length equal to the number of rows in `mat`, specifying the features in the set of interest.
     * Non-zero values indicate that the corresponding row is part of the feature set.
     *
     * @return A `Results` object containing the per-cell scores and per-feature weights.
     */
    template<typename T, typename IDX, typename X>
    Results run(const tatami::Matrix<T, IDX>* mat, const X* features) const {
        return run_blocked(mat, features, static_cast<unsigned char*>(NULL));
    }
};

}

#endif
