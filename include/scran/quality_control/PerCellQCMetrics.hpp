#ifndef SCRAN_PER_CELL_QC_METRICS_HPP
#define SCRAN_PER_CELL_QC_METRICS_HPP

#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>

#include "tatami/base/Matrix.hpp"
#include "tatami/stats/apply.hpp"
#include "../utils/vector_to_pointers.hpp"

/**
 * @file PerCellQCMetrics.hpp
 *
 * Compute typical per-cell quality control metrics.
 */

namespace scran {

/**
 * @brief Compute typical per-cell quality control metrics.
 *
 * Given a feature-by-cell count matrix, this class computes several QC metrics:
 * 
 * - The total sum for each cell quantifies the efficiency of library preparation and sequencing.
 * - The number of detected features also quantifies the library preparation efficiency,
 *   but with a greater focus on capturing the transcriptional complexity.
 * - The interpretation of the subset proportions depend on the subsets.
 *   In the most common case of mitochondrial transcripts, higher proportions indicate cell damage.
 *   Spike-in proportions can be interpreted in a similar manner.
 */
class PerCellQCMetrics {
public:
    /**
     * @brief Result store for QC metric calculations.
     */
    struct Results {
        /**
         * @param ncells Number of cells, i.e., the number of columns in the input matrix.
         * @param nsubsets Number of feature subsets of interest.
         */
        Results(size_t ncells, size_t nsubsets) : sums(ncells), detected(ncells), subset_proportions(nsubsets, std::vector<double>(ncells)) {}

        /**
         * Sum of counts for each cell.
         */
        std::vector<double> sums;

        /**
         * Number of detected features in each cell.
         */
        std::vector<int> detected;

        /**
         * Proportion of counts in each feature subset in each cell.
         * Each inner vector corresponds to a feature subset and is of length equal to the number of cells.
         */
        std::vector<std::vector<double> > subset_proportions;
    };

    /**
     * Compute the QC metrics from an input matrix and return the results.
     *
     * @tparam MAT Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam SUB Pointer to an array of values interpretable as booleans.
     *
     * @param mat Pointer to a feature-by-cells **tatami** matrix containing counts.
     * @param[in] subsets Vector of pointers to arrays of length equal to `mat->nrow()`.
     * Each array represents a feature subset and indicating whether each feature in `mat` belongs to that subset.
     * Users can pass `{}` if no subsets are to be used. 
     *
     * @return A `PerCellQCMetrics::Results` object containing the QC metrics.
     * Subset proportions are returned depending on the subsets defined at construction or by `set_subsets()`.
     *
     */
    template<class MAT, typename SUB = const uint8_t*>
    Results run(const MAT* mat, std::vector<SUB> subsets) {
        Results output(mat->ncol(), subsets.size());
        run(mat, std::move(subsets), output.sums.data(), output.detected.data(), vector_to_pointers(output.subset_proportions));
        return output;
    }

private:
    template<typename SUB, typename S, typename D, typename PROP>
    struct Common {
        Common(std::vector<SUB> subs, S* su, D* de, std::vector<PROP> subp) : subsets(std::move(subs)), sums(su), detected(de), subset_proportions(std::move(subp)) {}

        std::vector<SUB> subsets;
        S* sums;
        D* detected;
        std::vector<PROP> subset_proportions;
    };

    template<typename SUB, typename S, typename D, typename PROP>
    struct Factory : public Common<SUB, S, D, PROP> {
        typedef typename std::remove_reference<decltype(*std::declval<PROP>())>::type P;
        
        Factory(size_t nr, size_t nc, std::vector<SUB> subs, S* s, D* d, std::vector<PROP> subp) : NR(nr), NC(nc), Common<SUB, S, D, PROP>(std::move(subs), s, d, std::move(subp)) {
            std::fill(this->sums, this->sums + nc, static_cast<S>(0));
            std::fill(this->detected, this->detected + nc, static_cast<D>(0));
            for (size_t s = 0; s < this->subsets.size(); ++s) {
                std::fill(this->subset_proportions[s], this->subset_proportions[s] + nc, static_cast<P>(0.0));
            }
        }

        size_t NR, NC;

    public:
        struct DenseDirect : public Common<SUB, S, D, PROP> {
            DenseDirect(size_t nr, std::vector<SUB>& subs, S* s, D* d, std::vector<PROP>& subp) : NR(nr), Common<SUB, S, D, PROP>(std::move(subs), s, d, std::move(subp)) {}

            template<typename T>
            void compute(size_t c, const T* ptr, T* buffer) {
                for (size_t r = 0; r < NR; ++r) {
                    this->sums[c] += ptr[r];
                    this->detected[c] += static_cast<D>(ptr[r] > 0);
                }

                for (size_t s = 0; s < this->subsets.size(); ++s) {
                    const auto& sub = this->subsets[s];
                    auto& prop = this->subset_proportions[s][c];
                    for (size_t r = 0; r < NR; ++r) {
                        prop += sub[r] * ptr[r];
                    }

                    if (this->sums[c]) {
                        prop /= this->sums[c];
                    } else {
                        prop = std::numeric_limits<P>::quiet_NaN();
                    }
                }
            }

            size_t NR;
        };

        DenseDirect dense_direct() {
            return DenseDirect(NR, this->subsets, this->sums, this->detected, this->subset_proportions);
        }

    public:
        struct SparseDirect : public Common<SUB, S, D, PROP> {
            SparseDirect(std::vector<SUB> subs, S* s, D* d, std::vector<PROP> subp) : Common<SUB, S, D, PROP>(std::move(subs), s, d, std::move(subp)) {}

            template<typename T, typename IDX>
            void compute(size_t c, const tatami::SparseRange<T, IDX>& range, T* vbuffer, IDX* ibuffer) {
                for (size_t r = 0; r < range.number; ++r) {
                    this->sums[c] += range.value[r];
                    this->detected[c] += static_cast<D>(range.value[r] > 0);
                }

                for (size_t s = 0; s < this->subsets.size(); ++s) {
                    const auto& sub = this->subsets[s];
                    auto& prop = this->subset_proportions[s][c];
                    for (size_t i = 0; i < range.number; ++i) {
                        prop += sub[range.index[i]] * range.value[i];
                    }

                    if (this->sums[c]) {
                        prop /= this->sums[c];
                    } else {
                        prop = std::numeric_limits<P>::quiet_NaN();
                    }
                }
            }
        };
        
        SparseDirect sparse_direct() {
            return SparseDirect(this->subsets, this->sums, this->detected, this->subset_proportions);
        }

    public:
        struct DenseRunning : public Common<SUB, S, D, PROP> {
            DenseRunning(size_t nc, std::vector<SUB> subs, S* s, D* d, std::vector<PROP> subp) : NC(nc), Common<SUB, S, D, PROP>(std::move(subs), s, d, std::move(subp)) {}

            template<class T>
            void add (const T* ptr, T* buffer) {
                for (size_t c = 0; c < NC; ++c) {
                    this->sums[c] += ptr[c];
                    this->detected[c] += static_cast<D>(ptr[c] > 0);
                }

                for (size_t s = 0; s < this->subsets.size(); ++s) {
                    if (this->subsets[s][counter]) {
                        auto& sub = this->subset_proportions[s];
                        for (size_t c = 0; c < NC; ++c) {
                            sub[c] += ptr[c];
                        }
                    }
                }

                ++counter;
            }

            void finish() {
                for (size_t s = 0; s < this->subsets.size(); ++s) {
                    for (size_t c = 0; c < NC; ++c) {
                        auto& prop = this->subset_proportions[s][c];
                        if (this->sums[c]) {
                            prop /= this->sums[c];
                        } else {
                            prop = std::numeric_limits<P>::quiet_NaN();
                        }
                    }
                }
            }

            size_t NC;
            size_t counter = 0;
        };

        DenseRunning dense_running() {
            return DenseRunning(NC, this->subsets, this->sums, this->detected, this->subset_proportions);
        }

        DenseRunning dense_running(size_t start, size_t end) {
            auto subs = this->subset;
            for (auto& s : subs) {
                s += start;
            }

            auto subp = this->subset_proportions;
            for (auto& s : subp) {
                s += start;
            }

            return DenseRunning(end - start, std::move(subs), this->sums, this->detected, std::move(subp));
        }

    public:
        struct SparseRunning : public Common<SUB, S, D, PROP> {
            SparseRunning(size_t nc, std::vector<SUB> subs, S* s, D* d, std::vector<PROP> subp) : NC(nc), Common<SUB, S, D, PROP>(std::move(subs), s, d, std::move(subp)) {}

            template<typename T, typename IDX>
            void add (const tatami::SparseRange<T, IDX> range, T* vbuffer, IDX* ibuffer) {
                for (size_t i = 0; i < range.number; ++i) {
                    this->sums[range.index[i]] += range.value[i];
                    this->detected[range.index[i]] += static_cast<D>(range.value[i] > 0);
                }

                for (size_t s = 0; s < this->subsets.size(); ++s) {
                    if (this->subsets[s][counter]) {
                        for (size_t i = 0; i < range.number; ++i) {
                            this->subset_proportions[s][range.index[i]] += range.value[i];
                        }
                    }
                }

                ++counter;
            }

            void finish() {
                for (size_t s = 0; s < this->subsets.size(); ++s) {
                    for (size_t c = 0; c < NC; ++c) {
                        auto& prop = this->subset_proportions[s][c];
                        if (this->sums[c]) {
                            prop /= this->sums[c];
                        } else {
                            prop = std::numeric_limits<P>::quiet_NaN();
                        }
                    }
                }
            }

            size_t NC;
            size_t counter = 0;
        };
         
        SparseRunning sparse_running() {
            return SparseRunning(NC, this->subsets, this->sums, this->detected, this->subset_proportions);
        }

        SparseRunning sparse_running(size_t start, size_t end) {
            auto subs = this->subset;
            for (auto& s : subs) {
                s += start;
            }

            auto subp = this->subset_proportions;
            for (auto& s : subp) {
                s += start;
            }

            return SparseRunning(end - start, std::move(subs), this->sums, this->detected, std::move(subp));
        }
    };

public:
    /**
     * Compute the QC metrics from an input matrix and return the results.
     *
     * @tparam MAT Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam SUB Pointer to a type interpretable as boolean.
     * @tparam S Floating-point value, to store the sums.
     * @tparam D Integer value, to store the number of detected features.
     * @tparam PROP Floating point value, to store the subset proportions.
     *
     * @param mat Pointer to a feature-by-cells matrix containing counts.
     * @param[in] subsets Vector of pointers to arrays of length equal to `mat->nrow()`.
     * Each array represents a feature subset and indicating whether each feature in `mat` belongs to that subset.
     * Users can pass `{}` if no subsets are to be used. 
     * @param[out] sums Pointer to an array of length equal to the number of columns in `mat`.
     * This is used to store the computed sums for all cells.
     * @param[out] detected Pointer to an array of length equal to the number of columns in `mat`.
     * This is used to store the number of detected features for all cells.
     * @param[out] subset_proportions Vector of pointers to arrays of length equal to the number of columns in `mat`.
     * Each array corresponds to a feature subset and is used to store the proportion of counts in that subset across all cells.
     * The vector should be of length equal to that of `subsets`.
     * Users can pass `{}` if no subsets are used.
     *
     * @return `sums`, `detected`, and each array in `subset_proportions` is filled with the relevant statistics.
     */
    template<class MAT, typename SUB = const uint8_t*, typename S, typename D, typename PROP>
    void run(const MAT* mat, std::vector<SUB> subsets, S* sums, D* detected, std::vector<PROP> subset_proportions) {
        size_t nr = mat->nrow(), nc = mat->ncol();

#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::PerCellQCMetrics", "Computing quality control metrics for each cell");
#endif

        Factory fact(nr, nc, subsets, sums, detected, subset_proportions);
        tatami::apply<1>(mat, fact);
        return;
    }
};

}

#endif
