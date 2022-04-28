#ifndef SCRAN_MEDIAN_SIZE_FACTORS_HPP
#define SCRAN_MEDIAN_SIZE_FACTORS_HPP

#include <vector>
#include <limits>
#include "tatami/stats/medians.hpp"
#include "tatami/stats/sums.hpp"

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
 * Nonetheless, we provide it here because it can be helpful for removing composition biases between clusters.
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
    };

    /**
     * @param c Whether to center the size factors to have a mean of unity.
     * This is usually desirable for interpretation of relative values.
     * 
     * @return A reference to this `MedianSizeFactors` object.
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
     * When using shrinkage, we add a scaled version of the reference profile to each cell before computing the ratios.
     * The scaling of the reference profile varies for each cell and is proportional to the (relative) total count of that cell.
     * This implicitly pushes the median-based size factor towards a value that is proportional to the library size of the cell,
     * given that the median of the ratio of the reference against a scaled version of itself is just the scaling factor, i.e., the library size.
     *
     * The amount of shrinkage depends on the magnitude of the reference scaling.
     * The prior count should be interpreted as the number of extra reads from the reference profile that is added to each cell.
     * For example, the default of 10 means that the equivalent of 10 reads are added to each cell, distributed according to the reference profile.
     * Increasing the prior count will increase the strength of the shrinkage as the reference profile has a greater contribution to the ratios.
     */
    MedianSizeFactors& set_prior_count(double p = Defaults::prior_count) {
        prior_count = p;
        return *this;
    }

private:
    bool center = Defaults::center;
    double prior_count = Defaults::prior_count;

private:
    template<typename T, typename Ref, typename Out>
    struct Factory {
        Factory(size_t nr, size_t nc, const Ref* ref_, Out* fac) : NR(nr), sums(nc), ref(ref_), factors(fac) {}
        size_t NR;
        std::vector<T> sums;
        const Ref* ref;
        Out* factors;
    public:
        struct DenseDirect {
            DenseDirect(size_t nr, T* s, const Ref* ref_, Out* fac) : NR(nr), sums(s), ref(ref_), factors(fac), buffer(NR) {}

            void compute(size_t c, const T* ptr) {
                sums[c] = std::accumulate(ptr, ptr + NR, static_cast<T>(0));

                size_t sofar = 0;
                for (size_t i = 0; i < NR; ++i) {
                    if (ref[i] == 0 && ptr[i] == 0) {
                        continue;
                    }

                    if (ref[i] == 0) {
                        buffer[sofar] = std::numeric_limits<T>::infinity();
                    } else {
                        buffer[sofar] = ptr[i] / ref[i];
                    }

                    ++sofar;
                }

                // TODO: convince tatami maintainers to document this.
                factors[c] = tatami::stats::compute_median<Out>(buffer.data(), sofar);
            }

            size_t NR;
            T* sums;
            const Ref* ref;
            Out* factors;
            std::vector<Out> buffer;
        };

        DenseDirect dense_direct() {
            return DenseDirect(NR, sums.data(), ref, factors);
        }
    };

public:
    /**
     * Compute per-cell size factors against a user-supplied reference profile.
     *
     * @tparam T Numeric data type of the input matrix.
     * @tparam IDX Integer index type of the input matrix.
     * @tparam Ref Numeric data type of the reference profile.
     * @tparam Out Numeric data type of the output vector.
     *
     * @param mat Matrix containing non-negative expression data, usually counts.
     * Rows should be genes and columns should be cells.
     * @param[in] ref Pointer to an array containing the reference expression profile to normalize against.
     * This should be of length equal to the number of rows in `mat` and should contain non-negative values.
     * @param[out] output Pointer to an array to use to store the output size factors.
     * This should be of length equal to the number of columns in `mat`.
     *
     * @return `output` is filled with the size factors for each cell in `mat`.
     */
    template<typename T, typename IDX, typename Ref, typename Out>
    void run(const tatami::Matrix<T, IDX>* mat, const Ref* ref, Out* output) const {
        size_t NR = mat->nrow(), NC = mat->ncol();
        Factory<T, Ref, Out> fact(NR, NC, ref, output);
        tatami::apply<1>(mat, fact);

        // Mild squeezing towards library size-derived factors.
        if (prior_count && NR && NC) {
            const auto& sums = fact.sums;
            double mean = std::accumulate(sums.begin(), sums.end(), static_cast<T>(0));
            mean /= NC;

            double reftotal = std::accumulate(ref, ref + NR, static_cast<double>(0));

            if (mean && reftotal) {
                double scaling = prior_count / mean;
                for (size_t i = 0; i < NC; ++i) {
                    output[i] += sums[i] * scaling / reftotal;
                    output[i] /= 1.0 + scaling; // adjusting for the addition, as if we were dealing with a library size.
                }
            }
        }

        return;
    }

    /**
     * Compute per-cell size factors against an average pseudo-sample constructed from the row means of the input matrix.
     *
     * @tparam T Numeric data type of the input matrix.
     * @tparam IDX Integer index type of the input matrix.
     * @tparam Out Numeric data type of the output vector.
     *
     * @param mat Matrix containing non-negative expression data, usually counts.
     * Rows should be genes and columns should be cells.
     * @param[out] output Pointer to an array to use to store the output size factors.
     * This should be of length equal to the number of columns in `mat`.
     *
     * @return `output` is filled with the size factors for each cell in `mat`.
     */
    template<typename T, typename IDX, typename Out>
    void run_with_mean(const tatami::Matrix<T, IDX>* mat, Out* output) const {
        auto ref = tatami::row_sums(mat);
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
    template<typename Out>
    struct Result {
        Result(size_t NC) : factors(NC) {}
        std::vector<Out> factors;
    };

    /**
     * Compute per-cell size factors against a user-supplied reference profile.
     *
     * @tparam T Numeric data type of the input matrix.
     * @tparam IDX Integer index type of the input matrix.
     * @tparam Ref Numeric data type of the reference profile.
     *
     * @param mat Matrix containing non-negative expression data, usually counts.
     * Rows should be genes and columns should be cells.
     * @param[in] ref Pointer to an array containing the reference expression profile to normalize against.
     * This should be of length equal to the number of rows in `mat` and should contain non-negative values.
     *
     * @return A `Result` containing the size factors for each cell in `mat`.
     */
    template<typename Out = double, typename T, typename IDX, typename Ref>
    Result<Out> run(const tatami::Matrix<T, IDX>* mat, const Ref* ref) const {
        Result<Out> output(mat->ncol());
        run(mat, ref, output.factors.data());
        return output;
    }

    /**
     * Compute per-cell size factors against an average pseudo-sample constructed from the row means of the input matrix.
     *
     * @tparam T Numeric data type of the input matrix.
     * @tparam IDX Integer index type of the input matrix.
     *
     * @param mat Matrix containing non-negative expression data, usually counts.
     * Rows should be genes and columns should be cells.
     *
     * @return A `Result` containing the size factors for each cell in `mat`.
     */
    template<typename Out = double, typename T, typename IDX>
    Result<Out> run_with_mean(const tatami::Matrix<T, IDX>* mat) const {
        Result<Out> output(mat->ncol());
        run_with_mean(mat, output.factors.data());
        return output;
    }
};

}

#endif
