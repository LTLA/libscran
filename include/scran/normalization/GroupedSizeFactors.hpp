#ifndef SCRAN_CLUSTERED_SIZE_FACTORS_HPP
#define SCRAN_CLUSTERED_SIZE_FACTORS_HPP

#include "../utils/macros.hpp"

#include "MedianSizeFactors.hpp"
#include "../aggregation/AggregateAcrossCells.hpp"

#include "tatami/tatami.hpp"

#include <memory>
#include <vector>
#include <algorithm>

/**
 * @file GroupedSizeFactors.hpp
 *
 * @brief Compute size factors for groups of cells.
 */

namespace scran {

/**
 * @brief Compute grouped size factors to handle composition bias.
 *
 * This implements the grouping approach described in Lun et al. (2016) whereby groups/clusters of cells are used to construct pseudo-cells.
 * These pseudo-cells are normalized against each other using median-based size factors (see `MedianSizeFactors`) to obtain group-specific scaling factors.
 * Each cell is then normalized against its pseudo-cell using the library size; each cell's size factor is defined as product of its library size-based factor and the median-based factor for its group.
 *
 * This strategy leverages the reduced sparsity of the pseudo-cells to obtain sensible median-based size factors for removing composition biases,
 * while still generating per-cell factors for computing a normalized single-cell expression matrix in `LogNormCounts`.
 * The assumption is that there are no composition biases within each group; thus, the supplied groupings should correspond to subpopulations in the data, typically generated by clustering.
 *
 * @see
 * Lun ATL, Bach K and Marioni JC (2016).
 * Pooling across cells to normalize single-cell RNA sequencing data with many zero counts.
 * _Genome Biol._ 17:75
 */
class GroupedSizeFactors {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /** 
         * See `set_center()` for more details.
         */
        static constexpr bool center = true;

        /**
         * See `set_num_threads()`.
         */
        static constexpr int num_threads = 1;
    };

    /**
     * @param c Whether to center the size factors to have a mean of unity.
     * This is usually desirable for interpretation of relative values.
     * 
     * @return A reference to this `GroupedSizeFactors` object.
     *
     * For more control over centering, this can be set to `false` and the resulting size factors can be passed to `CenterSizeFactors`.
     */
    GroupedSizeFactors& set_center(bool c = Defaults::center) {
        center = c;
        return *this;
    }

    /**
     * @param p Prior count for the library size shrinkage, see `MedianSizeFactors::set_prior_count()` for details.
     *
     * @return A reference to this `GroupedSizeFactors` object.
     */
    GroupedSizeFactors& set_prior_count(double p = MedianSizeFactors::Defaults::prior_count) {
        prior_count = p;
        return *this;
    }

    /**
     * @param n Number of threads to use. 
     * @return A reference to this `AggregateAcrossCells` object.
     */
    GroupedSizeFactors& set_num_threads(int n = Defaults::num_threads) {
        num_threads = n;
        return *this;
    }

private:
    bool center = Defaults::center;
    double prior_count = MedianSizeFactors::Defaults::prior_count;
    int num_threads = Defaults::num_threads;

public:
    /**
     * Compute per-cell size factors based on user-supplied groupings.
     * The reference group is automatically determined from the pseudo-cell with the highest sum of root-counts;
     * this favors higher-coverage libraries with decent transcriptome complexity.
     *
     * @tparam T Numeric data type of the input matrix.
     * @tparam IDX Integer index type of the input matrix.
     * @tparam Group Integer type for the groupings.
     * @tparam Out Numeric data type of the output vector.
     *
     * @param mat Matrix containing non-negative expression data, usually counts.
     * Rows should be genes and columns should be cells.
     * @param[in] group Pointer to an array of group identifiers, of length equal to the number of columns in `mat`.
     * Values should be integers in \f$[0, N)\f$ where \f$N\f$ is the total number of groups.
     * @param[out] output Pointer to an array to use to store the output size factors.
     * This should be of length equal to the number of columns in `mat`.
     *
     * @return `output` is filled with the size factors for all cells in `mat`.
     */
    template<typename T, typename IDX, typename Group, typename Out>
    void run(const tatami::Matrix<T, IDX>* mat, const Group* group, Out* output) const {
        run_internal(mat, group, false, output);
    }

    /**
     * Compute per-cell size factors based on user-supplied groupings and a user-specified reference group.
     *
     * @tparam T Numeric data type of the input matrix.
     * @tparam IDX Integer index type of the input matrix.
     * @tparam Group Integer type for the groupings.
     * @tparam Out Numeric data type of the output vector.
     *
     * @param mat Matrix containing non-negative expression data, usually counts.
     * Rows should be genes and columns should be cells.
     * @param[in] group Pointer to an array of group identifiers, of length equal to the number of columns in `mat`.
     * Values should be integers in \f$[0, N)\f$ where \f$N\f$ is the total number of groups.
     * @param reference Identifier of the group to use as the reference.
     * This should be an integer in \f$[0, N)\f$.
     * @param[out] output Pointer to an array to use to store the output size factors.
     * This should be of length equal to the number of columns in `mat`.
     *
     * @return `output` is filled with the size factors for all cells in `mat`.
     */
    template<typename T, typename IDX, typename Group, typename Out>
    void run(const tatami::Matrix<T, IDX>* mat, const Group* group, size_t reference, Out* output) const {
        run_internal(mat, group, reference, output);
    }

private:
    template<typename T, typename IDX, typename Group, typename Ref, typename Out>
    void run_internal(const tatami::Matrix<T, IDX>* mat, const Group* group, Ref reference, Out* output) const {
        size_t NR = mat->nrow(), NC = mat->ncol();
        if (!NC) {
            return;
        }

        // Aggregating each group to get a pseudo-bulk sample.
        auto ngroups = *std::max_element(group, group + NC) + 1;
        std::vector<double> combined(ngroups * NR);
        {
            std::vector<double*> sums(ngroups);
            for (size_t i = 0; i < ngroups; ++i) {
                sums[i] = combined.data() + i * NR;
            }
            AggregateAcrossCells aggregator;
            aggregator.set_num_threads(num_threads).run(mat, group, std::move(sums), std::vector<int*>());
        }

        size_t ref = 0;
        if constexpr(std::is_same<Ref, size_t>::value) {
            ref = reference;
        } else {
            // Choosing one of them to be the reference. Here, we borrow some logic
            // from edgeR and use the one with the largest sum of square roots,
            // which provides a compromise between transcriptome coverage and
            // complexity. The root ensures that we don't pick a sample that just
            // has very high expression in a small subset of genes, while still
            // remaining responsive to the overall coverage level.
            double best = 0;

            for (size_t i = 0; i < ngroups; ++i) {
                auto start = combined.data() + i * NR;
                double current = 0;
                for (size_t j = 0; j < NR; ++j) {
                    current += std::sqrt(start[j]);
                }
                if (current > best) {
                    ref = i;
                    best = current;
                }
            }
        }

        // Computing median-based size factors. No need to center
        // here as we'll be recentering afterwards anyway.
        tatami::ArrayView view(combined.data(), combined.size());
        tatami::DenseColumnMatrix<T, IDX, decltype(view)> aggmat(NR, ngroups, std::move(view));
        MedianSizeFactors med;
        med.set_num_threads(num_threads).set_center(false).set_prior_count(prior_count);
        auto mres = med.run(&aggmat, combined.data() + ref * NR);

        // Propagating to each cell via library size-based normalization.
        auto aggcolsums = tatami::column_sums(&aggmat, num_threads);
        auto colsums = tatami::column_sums(mat, num_threads);
        for (size_t i = 0; i < NC; ++i) {
            auto curgroup = group[i];
            auto scale = static_cast<double>(colsums[i])/static_cast<double>(aggcolsums[curgroup]);
            output[i] = scale * mres.factors[curgroup];
        }

        // Throwing in some centering.
        if (center) {
            CenterSizeFactors centerer;
            centerer.run(NC, output);
        }
    }

public:
    /**
     * @brief Result of the size factor calculations.
     * @tparam Out Numeric type for the size factors.
     */
    template<typename Out>
    struct Results {
        /**
         * @cond
         */
        Results(size_t NC) : factors(NC) {}
        /**
         * @endcond
         */

        /**
         * Vector of length equal to the number of cells,
         * containing the size factor for each cell.
         */
        std::vector<Out> factors;
    };

    /**
     * Compute per-cell size factors based on user-supplied groupings.
     * The reference sample is automatically chosen, see `run()` for details.
     *
     * @tparam T Numeric data type of the input matrix.
     * @tparam IDX Integer index type of the input matrix.
     * @tparam Group Integer type for the groupings.
     *
     * @param mat Matrix containing non-negative expression data, usually counts.
     * Rows should be genes and columns should be cells.
     * @param[in] group Pointer to an array of group identifiers, of length equal to the number of columns in `mat`.
     * Values should be integers in \f$[0, N)\f$ where \f$N\f$ is the total number of groups.
     *
     * @return A `Results` object is returned containing the size factors.
     */
    template<typename Out = double, typename T, typename IDX, typename Group>
    Results<Out> run(const tatami::Matrix<T, IDX>* mat, const Group* group) const {
        Results<Out> output(mat->ncol());
        run(mat, group, output.factors.data());
        return output;
    }

    /**
     * Compute per-cell size factors based on user-supplied groupings and a user-specified grouping.
     *
     * @tparam T Numeric data type of the input matrix.
     * @tparam IDX Integer index type of the input matrix.
     * @tparam Group Integer type for the groupings.
     *
     * @param mat Matrix containing non-negative expression data, usually counts.
     * Rows should be genes and columns should be cells.
     * @param[in] group Pointer to an array of group identifiers, of length equal to the number of columns in `mat`.
     * Values should be integers in \f$[0, N)\f$ where \f$N\f$ is the total number of groups.
     * @param reference Identifier of the group to use as the reference.
     * This should be an integer in \f$[0, N)\f$.
     *
     * @return A `Results` object is returned containing the size factors.
     */
    template<typename Out = double, typename T, typename IDX, typename Group>
    Results<Out> run(const tatami::Matrix<T, IDX>* mat, const Group* group, size_t reference) const {
        Results<Out> output(mat->ncol());
        run(mat, group, reference, output.factors.data());
        return output;
    }
};

}

#endif
