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
 * @file PerCellRNAQcMetrics.hpp
 *
 * @brief Compute typical per-cell quality control metrics from an RNA count matrix.
 */

namespace scran {

/**
 * @brief Compute typical per-cell quality control metrics from an RNA count matrix.
 *
 * Given a feature-by-cell RNA count matrix, this class computes several QC metrics:
 * 
 * - The total sum for each cell, which represents the efficiency of library preparation and sequencing.
 *   Low totals indicate that the library was not successfully captured.
 * - The number of detected features.
 *   This also quantifies the library preparation efficiency, but with a greater focus on capturing the transcriptional complexity.
 * - The proportion of counts in pre-defined feature subsets.
 *   The exact interpretation depends on the nature of the subset -
 *   most commonly, one subset will contain all genes on the mitochondrial chromosome,
 *   where higher proportions indicate cell damage due to loss of cytoplasmic transcripts.
 *   Spike-in proportions can be interpreted in a similar manner.
 */
class PerCellRnaQcMetrics {
public:
    /**
     * @brief Result store for QC metric calculations.
     * 
     * Meaningful instances of this object should generally be constructed by calling the `PerCellRnaQcMetrics::run()` methods.
     * Empty instances can be default-constructed as placeholders.
     */
    struct Results {
        /**
         * @cond
         */
        Results() {}

        Results(size_t ncells, size_t nsubsets) : sums(ncells), detected(ncells), subset_proportions(nsubsets, std::vector<double>(ncells)) {}
        /**
         * @endcond
         */

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
     * @return A `PerCellRnaQcMetrics::Results` object containing the QC metrics.
     * Subset proportions are returned depending on the `subsets`.
     */
    template<class MAT, typename SUB = const uint8_t*>
    Results run(const MAT* mat, std::vector<SUB> subsets) {
        Results output(mat->ncol(), subsets.size());
        run(mat, std::move(subsets), output.sums.data(), output.detected.data(), vector_to_pointers(output.subset_proportions));
        return output;
    }

public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /**
         * See `set_subset_totals()` for more details.
         */
        static constexpr bool subset_totals = false;

        /**
         * See `set_num_threads()`.
         */
        static constexpr int num_threads = 1;
    };

    /**
     * @param t Whether to compute the total count for each subset rather than the proportion.
     *
     * @return A reference to this `PerCellRnaQcMetrics` object.
     *
     * Note that the total subset count will still be stored in the `subset_proportions` vectors or arrays in `run()`;
     * the only difference is that the division by the total count for each cell is omitted.
     */
    PerCellRnaQcMetrics& set_subset_totals(bool t = Defaults::subset_totals) {
        subset_totals = t;
        return *this;
    }

    /**
     * @param n Number of threads to use. 
     * @return A reference to this `PerCellRnaQcMetrics` object.
     */
    PerCellRnaQcMetrics& set_num_threads(int n = Defaults::num_threads) {
        num_threads = n;
        return *this;
    }

private:
    bool subset_totals = Defaults::subset_totals;
    int num_threads = Defaults::num_threads;

private:
    template<typename SUB, typename S, typename D, typename PROP>
    struct Common {
        Common(const std::vector<SUB>* subs, S* su, D* de, std::vector<PROP> subp, bool t) : 
            subsets_ptr(subs), sums(su), detected(de), subset_proportions(std::move(subp)), subset_totals(t) {}

        const std::vector<SUB>* subsets_ptr;
        S* sums;
        D* detected;
        std::vector<PROP> subset_proportions;
        bool subset_totals;

    public:
        template<typename P>
        void divide_subset_direct(size_t c, double& prop) {
            if (!subset_totals) {
                if (sums[c]) {
                    prop /= sums[c];
                } else {
                    prop = std::numeric_limits<P>::quiet_NaN();
                }
            }
        }

        template<typename P>
        void divide_subset_running(size_t start, size_t end) {
            if (!subset_totals) {
                for (size_t s = 0; s < subsets_ptr->size(); ++s) {
                    for (size_t c = start; c < end; ++c) {
                        auto& prop = subset_proportions[s][c];
                        if (sums[c]) {
                            prop /= sums[c];
                        } else {
                            prop = std::numeric_limits<P>::quiet_NaN();
                        }
                    }
                }
            }
        }
    };

    template<typename SUB, typename S, typename D, typename PROP>
    struct Factory : public Common<SUB, S, D, PROP> {
        typedef typename std::remove_reference<decltype(*std::declval<PROP>())>::type P;

        Factory(size_t nr, size_t nc, const std::vector<SUB>* subs, S* s, D* d, std::vector<PROP> subp, bool t) : NR(nr), NC(nc), Common<SUB, S, D, PROP>(subs, s, d, std::move(subp), t) {
            std::fill(this->sums, this->sums + nc, static_cast<S>(0));
            std::fill(this->detected, this->detected + nc, static_cast<D>(0));
            for (size_t s = 0; s < this->subsets_ptr->size(); ++s) {
                std::fill(this->subset_proportions[s], this->subset_proportions[s] + nc, static_cast<P>(0.0));
            }
        }

        size_t NR, NC;

    public:
        struct DenseDirect : public Common<SUB, S, D, PROP> {
            DenseDirect(size_t nr, const std::vector<SUB>* subs, S* s, D* d, std::vector<PROP> subp, bool t) : NR(nr), Common<SUB, S, D, PROP>(subs, s, d, std::move(subp), t) {}

            template<typename T>
            void compute(size_t c, const T* ptr) {
                for (size_t r = 0; r < NR; ++r) {
                    this->sums[c] += ptr[r];
                    this->detected[c] += static_cast<D>(ptr[r] > 0);
                }

                for (size_t s = 0; s < this->subsets_ptr->size(); ++s) {
                    const auto& sub = (*(this->subsets_ptr))[s];
                    auto& prop = this->subset_proportions[s][c];
                    for (size_t r = 0; r < NR; ++r) {
                        prop += sub[r] * ptr[r];
                    }
                    this->template divide_subset_direct<P>(c, prop); 
                }
            }

            size_t NR;
        };

        DenseDirect dense_direct() {
            return DenseDirect(NR, this->subsets_ptr, this->sums, this->detected, this->subset_proportions, this->subset_totals);
        }

    public:
        struct SparseDirect : public Common<SUB, S, D, PROP> {
            SparseDirect(const std::vector<SUB>* subs, S* s, D* d, std::vector<PROP> subp, bool t) : Common<SUB, S, D, PROP>(subs, s, d, std::move(subp), t) {}

            template<typename T, typename IDX>
            void compute(size_t c, const tatami::SparseRange<T, IDX>& range) {
                for (size_t r = 0; r < range.number; ++r) {
                    this->sums[c] += range.value[r];
                    this->detected[c] += static_cast<D>(range.value[r] > 0);
                }

                for (size_t s = 0; s < this->subsets_ptr->size(); ++s) {
                    const auto& sub = (*(this->subsets_ptr))[s];
                    auto& prop = this->subset_proportions[s][c];
                    for (size_t i = 0; i < range.number; ++i) {
                        prop += sub[range.index[i]] * range.value[i];
                    }
                    this->template divide_subset_direct<P>(c, prop); 
                }
            }
        };
        
        SparseDirect sparse_direct() {
            return SparseDirect(this->subsets_ptr, this->sums, this->detected, this->subset_proportions,  this->subset_totals);
        }

    public:
        struct DenseRunning : public Common<SUB, S, D, PROP> {
            DenseRunning(size_t s, size_t e, const std::vector<SUB>* subs, S* ss, D* d, std::vector<PROP> subp, bool t) : 
                num(e - s), Common<SUB, S, D, PROP>(subs, ss, d, std::move(subp), t) {}

            template<class T>
            void add (const T* ptr) {
                for (size_t c = 0; c < num; ++c) {
                    this->sums[c] += ptr[c];
                    this->detected[c] += static_cast<D>(ptr[c] > 0);
                }

                const auto& subsets = *(this->subsets_ptr);
                for (size_t s = 0; s < subsets.size(); ++s) {
                    if (subsets[s][counter]) {
                        auto& sub = this->subset_proportions[s];
                        for (size_t c = 0; c < num; ++c) {
                            sub[c] += ptr[c];
                        }
                    }
                }

                ++counter;
            }

            void finish() {
                this->template divide_subset_running<P>(0, num);
            }

            size_t counter = 0;
            size_t num;
        };

        DenseRunning dense_running() {
            return DenseRunning(0, NC, this->subsets_ptr, this->sums, this->detected, this->subset_proportions, this->subset_totals);
        }

        DenseRunning dense_running(size_t start, size_t end) {
            auto subp = this->subset_proportions;
            for (auto& s : subp) {
                s += start;
            }
            return DenseRunning(start, end, this->subsets_ptr, this->sums + start, this->detected + start, std::move(subp), this->subset_totals);
        }

    public:
        struct SparseRunning : public Common<SUB, S, D, PROP> {
            SparseRunning(size_t s, size_t e, const std::vector<SUB>* subs, S* ss, D* d, std::vector<PROP> subp, bool t) : 
                start(s), end(e), Common<SUB, S, D, PROP>(subs, ss, d, std::move(subp), t) {}

            template<typename T, typename IDX>
            void add (const tatami::SparseRange<T, IDX> range) {
                for (size_t i = 0; i < range.number; ++i) {
                    this->sums[range.index[i]] += range.value[i];
                    this->detected[range.index[i]] += static_cast<D>(range.value[i] > 0);
                }

                const auto& subsets = *(this->subsets_ptr);
                for (size_t s = 0; s < this->subsets_ptr->size(); ++s) {
                    if (subsets[s][counter]) {
                        for (size_t i = 0; i < range.number; ++i) {
                            this->subset_proportions[s][range.index[i]] += range.value[i];
                        }
                    }
                }

                ++counter;
            }

            void finish() {
                this->template divide_subset_running<P>(start, end);
            }

            const size_t start, end;
            size_t counter = 0;
        };
         
        SparseRunning sparse_running() {
            return SparseRunning(0, NC, this->subsets_ptr, this->sums, this->detected, this->subset_proportions, this->subset_totals);
        }

        SparseRunning sparse_running(size_t start, size_t end) {
            return SparseRunning(start, end, this->subsets_ptr, this->sums, this->detected, this->subset_proportions, this->subset_totals);
        }
    };

public:
    /**
     * Compute the QC metrics from an input matrix.
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
    void run(const MAT* mat, const std::vector<SUB>& subsets, S* sums, D* detected, std::vector<PROP> subset_proportions) {
        size_t nr = mat->nrow(), nc = mat->ncol();

#ifdef SCRAN_LOGGER
        SCRAN_LOGGER("scran::PerCellRnaQcMetrics", "Computing quality control metrics for each cell");
#endif

        Factory fact(nr, nc, &subsets, sums, detected, subset_proportions, subset_totals);
        tatami::apply<1>(mat, fact, num_threads);
        return;
    }
};

/**
 * An alias for `PerCellRnaQcMetrics`, provided for back-compatibility.
 */
typedef PerCellRnaQcMetrics PerCellQCMetrics;

}

#endif
