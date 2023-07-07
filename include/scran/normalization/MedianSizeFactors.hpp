#ifndef SCRAN_MEDIAN_SIZE_FACTORS_HPP
#define SCRAN_MEDIAN_SIZE_FACTORS_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <limits>
#include "tatami/tatami.hpp"
#include "CenterSizeFactors.hpp"

/**
 * @file MedianSizeFactors.hpp
 *
 * @brief Compute median-based size factors. 
 */

namespace scran {

/**
 * @brief Compute median-based size factors to handle composition bias.
 *
 * This is roughly equivalent to the DESeq2-based approach where the size factor for each library is defined as the median ratio against a reference profile.
 * The aim is to account for composition biases from differential expression between libraries, which would not be handled properly by library size normalization.
 * The main differences from DESeq2 are:
 *
 * - The row means are used as the default reference, instead of the geometric mean.
 *   This avoids problems with reference values of zero in sparse data.
 * - The median-based size factors are slightly shrunk towards the library size-derived factors.
 *   This ensures that the reported factors are never zero.
 *
 * In practice, this tends to work poorly for actual single-cell data due to its sparsity.
 * Nonetheless, we provide it here because it can be helpful for removing composition biases between clusters based on their averaged pseudo-bulk profiles.
 */
class MedianSizeFactors {
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
         * See `set_prior_count()` for more details.
         */
        static constexpr double prior_count = 10;

        /**
         * See `set_num_threads()`.
         */
        static constexpr int num_threads = 1;
    };

    /**
     * @param c Whether to center the size factors to have a mean of unity.
     * This is usually desirable for interpretation of relative values.
     * 
     * @return A reference to this `MedianSizeFactors` object.
     *
     * For more control over centering, this can be set to `false` and the resulting size factors can be passed to `CenterSizeFactors`.
     */
    MedianSizeFactors& set_center(bool c = Defaults::center) {
        center = c;
        return *this;
    }

    /**
     * @param p Prior count to use for shrinking median-based size factors towards their library size-based counterparts.
     * Larger values result in more shrinkage, while a value of zero will disable shrinkage altogether.
     *
     * @return A reference to this `MedianSizeFactors` object.
     *
     * When using shrinkage, we add a scaled version of the reference profile to each expression profile before computing the ratios.
     * The scaling of the reference profile varies for each profile and is proportional to the (relative) total count of that profile.
     * This implicitly pushes the median-based size factor towards a value that is proportional to the library size of the profile,
     * given that the median of the ratio of the reference against a scaled version of itself is just the scaling factor, i.e., the library size.
     *
     * The amount of shrinkage depends on the magnitude of the reference scaling.
     * The prior count should be interpreted as the number of extra reads from the reference profile that is added to each profile.
     * For example, the default of 10 means that the equivalent of 10 reads are added to each profile, distributed according to the reference profile.
     * Increasing the prior count will increase the strength of the shrinkage as the reference profile has a greater contribution to the ratios.
     */
    MedianSizeFactors& set_prior_count(double p = Defaults::prior_count) {
        prior_count = p;
        return *this;
    }

    /**
     * @param n Number of threads to use. 
     * @return A reference to this `MedianSizeFactors` object.
     */
    MedianSizeFactors& set_num_threads(int n = Defaults::num_threads) {
        num_threads = n;
        return *this;
    }

private:
    bool center = Defaults::center;
    double prior_count = Defaults::prior_count;
    int num_threads = Defaults::num_threads;

public:
    /**
     * Compute per-column size factors against a user-supplied reference profile.
     *
     * @tparam T Numeric data type of the input matrix.
     * @tparam IDX Integer index type of the input matrix.
     * @tparam Ref Numeric data type of the reference profile.
     * @tparam Out Numeric data type of the output vector.
     *
     * @param mat Matrix containing non-negative expression data, usually counts.
     * Rows should be genes; columns may be cells, but are more typically some kind of aggregated pseudo-bulk profile.
     * @param[in] ref Pointer to an array containing the reference expression profile to normalize against.
     * This should be of length equal to the number of rows in `mat` and should contain non-negative values.
     * @param[out] output Pointer to an array to use to store the output size factors.
     * This should be of length equal to the number of columns in `mat`.
     */
    template<typename T, typename IDX, typename Ref, typename Out>
    void run(const tatami::Matrix<T, IDX>* mat, const Ref* ref, Out* output) const {
        auto NR = mat->nrow(), NC = mat->ncol();

        std::vector<T> sums(NC);
        tatami::parallelize([&](size_t, IDX start, IDX length) -> void {
            auto ext = tatami::consecutive_extractor<false, false>(mat, start, length);
            std::vector<T> buffer(NR);

            for (IDX c = start, end = start + length; c < end; ++c) { 
                auto ptr = ext->fetch(c, buffer.data());
                sums[c] = std::accumulate(ptr, ptr + NR, static_cast<T>(0));

                size_t sofar = 0;
                for (IDX r = 0; r < NR; ++r) {
                    if (ref[r] == 0 && ptr[r] == 0) {
                        continue;
                    }

                    // potential overwriting of 'buffer' should be safe, as 'sofar' is always behind 'i'.
                    if (ref[r] == 0) { 
                        buffer[sofar] = std::numeric_limits<T>::infinity();
                    } else {
                        buffer[sofar] = ptr[r] / ref[r];
                    }

                    ++sofar;
                }

                // TODO: convince tatami maintainers to document this.
                output[c] = tatami::stats::compute_median<Out>(buffer.data(), sofar);
            }
        }, NC, num_threads);

        /* Mild squeezing towards library size-derived factors. Basically,
         * we're adding a scaled version of the reference profile to each
         * column, before normalizing against the reference profile. Given gene
         * i and column j:
         *
         *   ratio_{ij} = (y_{ij} + ref_i * extra_j) / ref_i
         *              = (y_{ij} / ref_i) + extra_j
         *
         * which means that the "shrunken" size factor is:
         *
         *   median(ratio_{ij}) = median(y_{ij} / ref_i) + extra_j
         *
         * This allows us to avoid the actual addition, as we can compute the
         * unshrunken size factor first and then add the j-specific scaling
         * later. This is important as otherwise we'd need to make two passes
         * over the matrix; once to get the mean library size to compute
         * extra_j, and then again to compute the shrunken factors.
         *
         * Incidentally, extra_j is defined as:
         *
         *   extra_j = p * t_j / T / R
         *
         * where p is the constant prior count, t_j is the library size for j,
         * T is the mean library size across all j, and R is the library size
         * for the reference profile.  Basically, p * ref_i / R is how we
         * "spread out" the prior count across all genes based on their
         * relative abundance in the reference profile, while t_j / T
         * represents the library size factor that we are shrinking towards.
         *
         * The addition of extra_j means that the shrunken size factor is
         * slightly too big to normalize against the reference. To adjust for
         * this, imagine that we have some unknown position-invariant function
         * f() such that E[f(y_{ij} / ref_i)] yields the true size factor x_j.
         * For example, f() would be the mean if there wasn't any differential
         * expression between columns; or the median, if the counts were large
         * enough. Our shrunken size factor can then be written as
         *
         *   f(ratio_{ij}) = x_j + extra_j
         * 
         * To correct our shrunken size factor to x_j, we need to divide it by:
         *
         *   (x_j + extra_j) / x_j = 1 + (extra_j / x_j)
         * 
         * Of course, we don't actually know x_j, so we need to approximate by
         * assuming x_j =~ t_j / R, i.e., library size normalization against
         * the reference. This simplifies the correction:
         *
         *   1 + (p / T)
         *
         * which is what we use to divide our median-based shrunken factor.
         * It's not exactly correct but it be good enough when T >> p, and
         * it at least ensures that the normalization is accurate in the
         * simplest case where the only difference is due to library size.
         */
        if (prior_count && NR && NC) {
            double mean = std::accumulate(sums.begin(), sums.end(), static_cast<T>(0));
            mean /= NC;

            double reftotal = std::accumulate(ref, ref + NR, static_cast<double>(0));

            if (mean && reftotal) {
                double scaling = prior_count / mean;
                for (size_t i = 0; i < NC; ++i) {
                    output[i] += sums[i] * scaling / reftotal;
                    output[i] /= 1.0 + scaling; 
                }
            }
        }

        // Throwing in some centering.
        if (center) {
            CenterSizeFactors centerer;
            centerer.run(NC, output);
        }

        return;
    }

    /**
     * Compute per-column size factors against an average pseudo-sample constructed from the row means of the input matrix.
     *
     * @tparam T Numeric data type of the input matrix.
     * @tparam IDX Integer index type of the input matrix.
     * @tparam Out Numeric data type of the output vector.
     *
     * @param mat Matrix containing non-negative expression data, usually counts.
     * Rows should be genes; columns may be cells, but are more typically some kind of aggregated pseudo-bulk profile.
     * @param[out] output Pointer to an array to use to store the output size factors.
     * This should be of length equal to the number of columns in `mat`.
     */
    template<typename T, typename IDX, typename Out>
    void run_with_mean(const tatami::Matrix<T, IDX>* mat, Out* output) const {
        auto ref = tatami::row_sums(mat, num_threads);
        if (ref.size()) {
            double NC = mat->ncol();
            for (auto& r : ref) {
                r /= NC;
            }
        }
        run(mat, ref.data(), output);
        return;
    }

public:
    /**
     * @brief Result of the size factor calculation.
     *
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
         * Vector of length equal to the number of columns,
         * containing the size factor for each column.
         */
        std::vector<Out> factors;
    };

    /**
     * Compute per-column size factors against a user-supplied reference profile.
     *
     * @tparam Out Numeric type for the size factors.
     * @tparam T Numeric data type of the input matrix.
     * @tparam IDX Integer index type of the input matrix.
     * @tparam Ref Numeric data type of the reference profile.
     *
     * @param mat Matrix containing non-negative expression data, usually counts.
     * Rows should be genes; columns may be cells, but are more typically some kind of aggregated pseudo-bulk profile.
     * @param[in] ref Pointer to an array containing the reference expression profile to normalize against.
     * This should be of length equal to the number of rows in `mat` and should contain non-negative values.
     *
     * @return A `Results` containing the size factors for each column in `mat`.
     */
    template<typename Out = double, typename T, typename IDX, typename Ref>
    Results<Out> run(const tatami::Matrix<T, IDX>* mat, const Ref* ref) const {
        Results<Out> output(mat->ncol());
        run(mat, ref, output.factors.data());
        return output;
    }

    /**
     * Compute per-column size factors against an average pseudo-sample constructed from the row means of the input matrix.
     *
     * @tparam Out Numeric type for the size factors.
     * @tparam T Numeric data type of the input matrix.
     * @tparam IDX Integer index type of the input matrix.
     *
     * @param mat Matrix containing non-negative expression data, usually counts.
     * Rows should be genes; columns may be cells, but are more typically some kind of aggregated pseudo-bulk profile.
     *
     * @return A `Results` containing the size factors for each column in `mat`.
     */
    template<typename Out = double, typename T, typename IDX>
    Results<Out> run_with_mean(const tatami::Matrix<T, IDX>* mat) const {
        Results<Out> output(mat->ncol());
        run_with_mean(mat, output.factors.data());
        return output;
    }
};

}

#endif
