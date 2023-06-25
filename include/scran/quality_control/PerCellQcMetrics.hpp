#ifndef SCRAN_PER_CELL_QC_METRICS_HPP
#define SCRAN_PER_CELL_QC_METRICS_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>

#include "tatami/tatami.hpp"
#include "../utils/vector_to_pointers.hpp"

/**
 * @file PerCellQcMetrics.hpp
 *
 * @brief Compute a variety of per-cell quality control metrics from a count matrix.
 */

namespace scran {

/**
 * @brief Compute a variety of per-cell quality control metrics from a count matrix.
 *
 * Given a feature-by-cell count matrix, this class computes several QC metrics:
 * 
 * - The total sum for each cell, which represents the efficiency of library preparation and sequencing.
 *   Low totals indicate that the library was not successfully captured.
 * - The number of detected features (i.e., with non-zero counts).
 *   This also quantifies the library preparation efficiency, but with a greater focus on capturing the transcriptional complexity.
 * - The maximum count across all features.
 *   This is useful in situations where only one feature is expected to be present, e.g., CRISPR guides, hash tags.
 * - The row index of the feature with the maximum count.
 *   If multiple features are tied for the maximum count, the earliest feature is reported.
 * - The total count in pre-defined feature subsets.
 *   The exact interpretation depends on the nature of the subset -
 *   most commonly, one subset will contain all genes on the mitochondrial chromosome,
 *   where higher proportions of counts in the mitochondrial subset indicate cell damage due to loss of cytoplasmic transcripts.
 *   Spike-in proportions can be interpreted in a similar manner.
 * - The number of detected features in pre-defined feature subsets.
 *   Analogous to the number of detected features for the entire feature space.
 * 
 * These calculations are done in a single pass, avoiding the need to reload data from the count matrix.
 */
class PerCellQcMetrics {
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

        Results(size_t nsubsets) : subset_total(nsubsets), subset_detected(nsubsets) {}
        /**
         * @endcond
         */

        /**
         * Total count for each cell.
         * Empty if `PerCellQcMetrics::compute_total` is false.
         */
        std::vector<double> total;

        /**
         * Number of detected features in each cell.
         * Empty if `PerCellQcMetrics::compute_detected` is false.
         */
        std::vector<int> detected;

        /**
         * Row index of the most-expressed feature in each cell.
         * On ties, the first feature is arbitrarily chosen.
         * Empty if `PerCellQcMetrics::compute_max_index` is false.
         */
        std::vector<int> max_index;

        /**
         * Maximum count value in each cell.
         * Empty if `PerCellQcMetrics::compute_max_count` is false.
         */
        std::vector<double> max_count;

        /**
         * Total count of each feature subset in each cell.
         * Each inner vector corresponds to a feature subset and is of length equal to the number of cells.
         * Empty if there are no feature subsets or if `PerCellQcMetrics::compute_subset_total` is false.
         */
        std::vector<std::vector<double> > subset_total;

        /**
         * Number of detected features in each feature subset in each cell.
         * Each inner vector corresponds to a feature subset and is of length equal to the number of cells.
         * Empty if there are no feature subsets or if `PerCellQcMetrics::compute_subset_detected` is false.
         */
        std::vector<std::vector<int> > subset_detected;
    };

public:
    /**
     * @brief Buffers for direct storage of the calculated statistics.
     * @tparam Float Floating point type to store the totals.
     * @tparam Integer Integer type to store the counts and indices.
     */
    template<typename Float = double, typename Integer = int>
    struct Buffers {
        /**
         * @cond
         */
        Buffers() {}

        Buffers(size_t nsubsets) : subset_total(nsubsets, NULL), subset_detected(nsubsets, NULL) {}
        /**
         * @endcond
         */

        /**
         * Pointer to an array of length equal to the number of cells, equivalent to `Results::total`.
         * Set to `NULL` to skip this calculation.
         */
        Float* total = NULL;

        /**
         * Pointer to an array of length equal to the number of cells, equivalent to `Results::detected`.
         * Set to `NULL` to skip this calculation.
         */
        Integer* detected = NULL;

        /**
         * Pointer to an array of length equal to the number of cells, equivalent to `Results::max_index`.
         * Set to `NULL` to skip this calculation.
         */
        Integer* max_index = NULL;

        /**
         * Pointer to an array of length equal to the number of cells, equivalent to `Results::max_count`.
         * Set to `NULL` to skip this calculation.
         */
        Float* max_count = NULL;

        /**
         * Vector of pointers of length equal to the number of feature subsets,
         * where each point is to an array of length equal to the number of cells; equivalent to `Results::subset_total`.
         * Set any value to `NULL` to skip the calculation for the corresponding feature subset,
         * or leave empty to skip calculations for all feature subsets.
         */
        std::vector<Float*> subset_total;

        /**
         * Vector of pointers of length equal to the number of feature subsets,
         * where each point is to an array of length equal to the number of cells; equivalent to `Results::subset_detected`.
         * Set any value to `NULL` to skip the calculation for the corresponding feature subset,
         * or leave empty to skip calculations for all feature subsets.
         */
        std::vector<Integer*> subset_detected;

        /**
         * Whether all of the arrays in `Buffers` have already been filled with zeros.
         * (For `Buffers::max_count`, this should be filled with the lowest possible value of `Float`,
         * if expression might be negative.)
         */
        bool already_zeroed = false;
    };

public:
    /**
     * @brief Default parameters.
     */
    struct Defaults {
        /**
         * See `set_compute_total()` for details.
         */
        static constexpr bool compute_total = true;

        /**
         * See `set_compute_detected()` for details.
         */
        static constexpr bool compute_detected = true;

        /**
         * See `set_compute_max_count()` for details.
         */
        static constexpr bool compute_max_count = true;

        /**
         * See `set_compute_max_index()` for details.
         */
        static constexpr bool compute_max_index = true;

        /**
         * See `set_compute_subset_total()` for details.
         */
        static constexpr bool compute_subset_total = true;

        /**
         * See `set_compute_subset_detected()` for details.
         */
        static constexpr bool compute_subset_detected = true;

        /**
         * See `set_num_threads()` for details.
         */
        static constexpr int num_threads = 1;
    };

private:
    bool compute_total = Defaults::compute_total;
    bool compute_detected = Defaults::compute_detected;
    bool compute_max_count = Defaults::compute_max_count;
    bool compute_max_index = Defaults::compute_max_index;
    bool compute_subset_total = Defaults::compute_subset_total;
    bool compute_subset_detected = Defaults::compute_subset_detected;
    int num_threads = Defaults::num_threads;

public:
    /**
     * @param s Whether to compute the total count for each cell.
     * This option only affects the `run()` method that returns a `Results` object.
     *
     * @return Reference to this `PerCellQcMetrics` object.
     */
    PerCellQcMetrics& set_compute_total(bool s = Defaults::compute_total) {
        compute_total = s;
        return *this;        
    }

    /**
     * @param s Whether to compute the number of detected features for each cell.
     * This option only affects the `run()` method that returns a `Results` object.
     *
     * @return Reference to this `PerCellQcMetrics` object.
     */
    PerCellQcMetrics& set_compute_detected(bool s = Defaults::compute_detected) {
        compute_detected = s;
        return *this;
    }

    /**
     * @param s Whether to compute the maximmum count for each cell.
     * This option only affects the `run()` method that returns a `Results` object.
     *
     * @return Reference to this `PerCellQcMetrics` object.
     */
    PerCellQcMetrics& set_compute_max_count(bool s = Defaults::compute_max_count) {
        compute_max_count = s;
        return *this;        
    }

    /**
     * @param s Whether to store the index of the feature with the maximum count for each cell.
     * This option only affects the `run()` method that returns a `Results` object.
     *
     * @return Reference to this `PerCellQcMetrics` object.
     */
    PerCellQcMetrics& set_compute_max_index(bool s = Defaults::compute_max_index) {
        compute_max_index = s;
        return *this;        
    }

    /**
     * @param s Whether to compute the total count in each feature subset.
     * This option only affects the `run()` method that returns a `Results` object.
     *
     * @return Reference to this `PerCellQcMetrics` object.
     */
    PerCellQcMetrics& set_compute_subset_total(bool s = Defaults::compute_subset_total) {
        compute_subset_total = s;
        return *this;
    }

    /**
     * @param s Whether to compute the number of detected features in each feature subset.
     * This option only affects the `run()` method that returns a `Results` object.
     *
     * @return Reference to this `PerCellQcMetrics` object.
     */
    PerCellQcMetrics& set_compute_subset_detected(bool s = Defaults::compute_subset_detected) {
        compute_subset_detected = s;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     *
     * @return Reference to this `PerCellQcMetrics` object.
     */
    PerCellQcMetrics& set_num_threads(int n = Defaults::num_threads) {
        num_threads = n;
        return *this;
    }

public:
    /**
     * Compute the QC metrics from an input matrix and return the results.
     *
     * @tparam Matrix Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam Subset Pointer to an array of values interpretable as booleans.
     *
     * @param mat Pointer to a feature-by-cells **tatami** matrix containing counts.
     * @param[in] subsets Vector of pointers to arrays of length equal to `mat->nrow()`.
     * Each array represents a feature subset and indicating whether each feature in `mat` belongs to that subset.
     * Users can pass `{}` if no subsets are to be used. 
     *
     * @return A `Results` object containing the QC metrics.
     * Subset totals are returned depending on the `subsets`.
     */
    template<class Matrix, typename Subset = const uint8_t*>
    Results run(const Matrix* mat, const std::vector<Subset>& subsets) const {
        Results output;
        Buffers<> buffers;
        auto ncells = mat->ncol();

        if (compute_total) {
            output.total.resize(ncells);
            buffers.total = output.total.data();
        }
        if (compute_detected) {
            output.detected.resize(ncells);
            buffers.detected = output.detected.data();
        }
        if (compute_max_index) {
            output.max_index.resize(ncells);
            buffers.max_index = output.max_index.data();
        }
        if (compute_max_count) {
            output.max_count.resize(ncells, pick_fill_value<double>());
            buffers.max_count = output.max_count.data();
        }

        size_t nsubsets = subsets.size();

        if (compute_subset_total) {
            output.subset_total.resize(nsubsets);
            buffers.subset_total.resize(nsubsets);
            for (size_t s = 0; s < nsubsets; ++s) {
                output.subset_total[s].resize(ncells);
                buffers.subset_total[s] = output.subset_total[s].data();
            }
        }

        if (compute_subset_detected) {
            output.subset_detected.resize(nsubsets);
            buffers.subset_detected.resize(nsubsets);
            for (size_t s = 0; s < nsubsets; ++s) {
                output.subset_detected[s].resize(ncells);
                buffers.subset_detected[s] = output.subset_detected[s].data();
            }
        }

        buffers.already_zeroed = true;
        run(mat, subsets, buffers);
        return output;
    }

private:
    template<typename T>
    static constexpr T pick_fill_value() {
        if constexpr(std::numeric_limits<T>::has_infinity) {
            return -std::numeric_limits<T>::infinity();
        } else {
            return std::numeric_limits<T>::lowest();
        }
    }

private:
    template<typename Data_, typename Index_, typename Subset_, typename Float_, typename Integer_>
    void compute_direct_dense(const tatami::Matrix<Data_, Index_>* mat, const std::vector<Subset_>& subsets, Buffers<Float_, Integer_>& output) const {
        std::vector<std::vector<int> > subset_indices;
        if (!output.subset_total.empty() || !output.subset_detected.empty()) {
            size_t nsubsets = subsets.size();
            subset_indices.resize(nsubsets);
            auto NR = mat->nrow();

            for (size_t s = 0; s < nsubsets; ++s) {
                auto& current = subset_indices[s];
                const auto& source = subsets[s];

                for (int i = 0; i < NR; ++i) {
                    if (source[i]) {
                        current.push_back(i);
                    }
                }
            }
        }

        tatami::parallelize([&](size_t, Index_ start, Index_ length) {
            auto NR = mat->nrow();
            auto ext = tatami::consecutive_extractor<false, false>(mat, start, length);
            std::vector<Data_> vbuffer(NR);
            bool do_max = output.max_index || output.max_count;

            for (Index_ c = start, end = start + length; c < end; ++c) {
                auto ptr = ext->fetch(c, vbuffer.data());

                if (output.total) {
                    output.total[c] = std::accumulate(ptr, ptr + NR, static_cast<Float_>(0));
                }

                if (output.detected) {
                    Integer_ count = 0;
                    for (Index_ r = 0; r < NR; ++r) {
                        count += (ptr[r] != 0);
                    }
                    output.detected[c] = count;
                }

                if (do_max) {
                    auto max_count = PerCellQcMetrics::pick_fill_value<Float_>();
                    Integer_ max_index = 0;
                    for (Index_ r = 0; r < NR; ++r) {
                        if (max_count < ptr[r]) {
                            max_count = ptr[r];
                            max_index = r;
                        }
                    }

                    if (output.max_index) {
                        output.max_index[c] = max_index;
                    }
                    if (output.max_count) {
                        output.max_count[c] = max_count;
                    }
                }

                size_t nsubsets = subset_indices.size();
                for (size_t s = 0; s < nsubsets; ++s) {
                    const auto& sub = subset_indices[s];

                    if (!output.subset_total.empty() && output.subset_total[s]) {
                        Float_ current = 0;
                        for (auto r : sub) {
                            current += ptr[r];
                        }
                        output.subset_total[s][c] = current;
                    }

                    if (!output.subset_detected.empty() && output.subset_detected[s]) {
                        Integer_ current = 0;
                        for (auto r : sub) {
                            current += ptr[r] != 0;
                        }
                        output.subset_detected[s][c] = current;
                    }
                }
            }
        }, mat->ncol(), num_threads);
    }

    template<typename Data_, typename Index_, typename Subset_, typename Float_, typename Integer_>
    void compute_direct_sparse(const tatami::Matrix<Data_, Index_>* mat, const std::vector<Subset_>& subsets, Buffers<Float_, Integer_>& output) const {
        tatami::Options opt;
        opt.sparse_ordered_index = false;

        tatami::parallelize([&](size_t, Index_ start, Index_ length) {
            auto NR = mat->nrow();
            auto ext = tatami::consecutive_extractor<false, true>(mat, start, length, opt);
            std::vector<Data_> vbuffer(NR);
            std::vector<Index_> ibuffer(NR);

            bool do_max = output.max_index || output.max_count;
            std::vector<unsigned char> internal_is_nonzero(output.max_index ? NR : 0);

            for (Index_ c = start, end = start + length; c < end; ++c) {
                auto range = ext->fetch(c, vbuffer.data(), ibuffer.data());

                if (output.total) {
                    output.total[c] = std::accumulate(range.value, range.value + range.number, static_cast<Float_>(0));
                }

                if (output.detected) {
                    Integer_ current = 0;
                    for (Index_ i = 0; i < range.number; ++i) {
                        current += (range.value[i] != 0);
                    }
                    output.detected[c] = current;
                }

                if (do_max) {
                    auto max_count = PerCellQcMetrics::pick_fill_value<Float_>();
                    Integer_ max_index = 0;
                    for (Index_ i = 0; i < range.number; ++i) {
                        if (max_count < range.value[i]) {
                            max_count = range.value[i];
                            max_index = range.index[i];
                        }
                    }

                    if (max_count <= 0 && range.number < NR) {
                        // Zero is the max.
                        max_count = 0;

                        // Finding the index of the first zero by tracking all
                        // indices with non-zero values. This isn't the fastest
                        // approach but it's simple and avoids assuming that
                        // indices are sorted. Hopefully we don't have to hit
                        // this section often.
                        if (output.max_index) {
                            for (Index_ i = 0; i < range.number; ++i) {
                                if (range.value[i]) {
                                    internal_is_nonzero[range.index[i]] = 1;
                                }
                            }
                            for (Index_ r = 0; r < NR; ++r) {
                                if (internal_is_nonzero[r] == 0) {
                                    max_index = r;
                                    break;
                                }
                            }
                            for (Index_ i = 0; i < range.number; ++i) { // setting back to zero.
                                internal_is_nonzero[range.index[i]] = 0;
                            }
                        }
                    }

                    if (output.max_index) {
                        output.max_index[c] = max_index;
                    }
                    if (output.max_count) {
                        output.max_count[c] = max_count;
                    }
                }

                size_t nsubsets = subsets.size();
                for (size_t s = 0; s < nsubsets; ++s) {
                    const auto& sub = subsets[s];

                    if (!output.subset_total.empty() && output.subset_total[s]) {
                        auto& current = output.subset_total[s][c];
                        for (Index_ i = 0; i < range.number; ++i) {
                            current += (sub[range.index[i]] != 0) * range.value[i];
                        }
                    }

                    if (!output.subset_detected.empty() && output.subset_detected[s]) {
                        auto& current = output.subset_detected[s][c];
                        for (Index_ i = 0; i < range.number; ++i) {
                            current += (sub[range.index[i]] != 0) * (range.value[i] != 0);
                        }
                    }
                }
            }
        }, mat->ncol(), num_threads);
    }

    template<typename Data_, typename Index_, typename Subset_, typename Float_, typename Integer_>
    void compute_running_dense(const tatami::Matrix<Data_, Index_>* mat, const std::vector<Subset_>& subsets, Buffers<Float_, Integer_>& output) const {
        tatami::parallelize([&](size_t, Index_ start, Index_ len) {
            auto NR = mat->nrow();
            auto ext = tatami::consecutive_extractor<true, false>(mat, 0, NR, start, len);
            std::vector<Data_> vbuffer(len);
            bool do_max = output.max_index || output.max_count;
            std::vector<Float_> internal_max_count(do_max && !output.max_count ? len : 0);

            for (Index_ r = 0; r < NR; ++r) {
                auto ptr = ext->fetch(r, vbuffer.data());

                if (output.total) {
                    auto outt = output.total + start;
                    for (Index_ i = 0; i < len; ++i) {
                        outt[i] += ptr[i];
                    }
                }

                if (output.detected) {
                    auto outd = output.detected + start;
                    for (Index_ i = 0; i < len; ++i) {
                        outd[i] += (ptr[i] != 0);
                    }
                }

                if (do_max) {
                    auto outmc = (output.max_count ? output.max_count + start : internal_max_count.data());

                    if (r == 0) {
                        std::copy(ptr, ptr + len, outmc);
                        if (output.max_index) {
                            auto outmi = output.max_index + start;
                            std::fill(outmi, outmi + len, 0);
                        }

                    } else {
                        for (Index_ i = 0; i < len; ++i) {
                            auto& curmax = outmc[i];
                            if (curmax < ptr[i]) {
                                curmax = ptr[i];
                                if (output.max_index) {
                                    output.max_index[i + start] = r;
                                }
                            }
                        }
                    }
                }

                size_t nsubsets = subsets.size();
                for (size_t s = 0; s < nsubsets; ++s) {
                    const auto& sub = subsets[s];
                    if (sub[r] == 0) {
                        continue;
                    }

                    if (!output.subset_total.empty() && output.subset_total[s]) {
                        auto current = output.subset_total[s] + start;
                        for (Index_ i = 0; i < len; ++i) {
                            current[i] += ptr[i];
                        }
                    }

                    if (!output.subset_detected.empty() && output.subset_detected[s]) {
                        auto current = output.subset_detected[s] + start;
                        for (Index_ i = 0; i < len; ++i) {
                            current[i] += (ptr[i] != 0);
                        }
                    }
                }
            }
        }, mat->ncol(), num_threads);
    }

    template<typename Data_, typename Index_, typename Subset_, typename Float_, typename Integer_>
    void compute_running_sparse(const tatami::Matrix<Data_, Index_>* mat, const std::vector<Subset_>& subsets, Buffers<Float_, Integer_>& output) const {
        tatami::Options opt;
        opt.sparse_ordered_index = false;

        Index_ NC = mat->ncol();
        bool do_max = output.max_index || output.max_count;
        std::vector<Float_> internal_max_count(do_max && !output.max_count ? NC : 0);
        std::vector<Integer_> internal_last_consecutive_nonzero(do_max ? NC : 0);

        tatami::parallelize([&](size_t t, Index_ start, Index_ len) {
            auto NR = mat->nrow();
            auto ext = tatami::consecutive_extractor<true, true>(mat, 0, NR, start, len, opt);
            std::vector<Data_> vbuffer(len);
            std::vector<Index_> ibuffer(len);

            for (Index_ r = 0; r < NR; ++r) {
                auto range = ext->fetch(r, vbuffer.data(), ibuffer.data());

                if (output.total) {
                    for (Index_ i = 0; i < range.number; ++i) {
                        output.total[range.index[i]] += range.value[i];
                    }
                }

                if (output.detected) {
                    for (Index_ i = 0; i < range.number; ++i) {
                        output.detected[range.index[i]] += (range.value[i] != 0);
                    }
                }

                if (do_max) {
                    auto outmc = (output.max_count ? output.max_count : internal_max_count.data());

                    for (Index_ i = 0; i < range.number; ++i) {
                        auto& curmax = outmc[range.index[i]];
                        if (curmax < range.value[i]) {
                            curmax = range.value[i];
                            if (output.max_index) {
                                output.max_index[range.index[i]] = r;
                            }
                        }

                        // Getting the index of the last consecutive non-zero entry, so that
                        // we can check if zero is the max and gets its first occurrence, if necessary.
                        auto& last = internal_last_consecutive_nonzero[range.index[i]];
                        if (static_cast<Index_>(last) == r) {
                            if (range.value[i] != 0) {
                                ++last;
                            }
                        }
                    }
                }

                size_t nsubsets = subsets.size();
                for (size_t s = 0; s < nsubsets; ++s) {
                    const auto& sub = subsets[s];
                    if (sub[r] == 0) {
                        continue;
                    }

                    if (!output.subset_total.empty() && output.subset_total[s]) {
                        auto& current = output.subset_total[s];
                        for (size_t i = 0; i < range.number; ++i) {
                            current[range.index[i]] += range.value[i];
                        }
                    }

                    if (!output.subset_detected.empty() && output.subset_detected[s]) {
                        auto& current = output.subset_detected[s];
                        for (size_t i = 0; i < range.number; ++i) {
                            current[range.index[i]] += (range.value[i] != 0);
                        }
                    }
                }
            }
        }, NC, num_threads);

        if (do_max) {
            auto outmc = (output.max_count ? output.max_count : internal_max_count.data());
            auto NR = mat->nrow();

            // Checking anything with non-positive maximum, and replacing it with zero
            // if there are any zeros (i.e., consecutive non-zeros is not equal to the number of rows).
            for (Index_ c = 0; c < NC; ++c) {
                auto& current = outmc[c];
                if (current > 0) {
                    continue;
                }

                auto last_nz = internal_last_consecutive_nonzero[c];
                if (last_nz == NR) {
                    continue;
                }

                current = 0;
                if (output.max_index) {
                    output.max_index[c] = last_nz;
                }
            }
        }
    }

public:
    /**
     * Compute the QC metrics from an input matrix.
     *
     * @tparam Matrix Type of matrix, usually a `tatami::NumericMatrix`.
     * @tparam Subset Pointer to a type interpretable as boolean.
     *
     * @param mat Pointer to a feature-by-cells matrix containing counts.
     * @param[in] subsets Vector of pointers to arrays of length equal to `mat->nrow()`.
     * Each array represents a feature subset and indicating whether each feature in `mat` belongs to that subset.
     * Users can pass `{}` if no subsets are to be used. 
     * @param[out] output A `Buffers` object in which the computed statistics are to be stored.
     */
    template<class Matrix, typename Subset = const uint8_t*, typename Float, typename Integer>
    void run(const Matrix* mat, const std::vector<Subset>& subsets, Buffers<Float, Integer>& output) const {
        if (!output.already_zeroed) {
            size_t n = mat->ncol();
            auto check_and_fill = [&](auto* ptr, auto value) -> void {
                if (ptr) {
                    std::fill(ptr, ptr + n, value);
                }
            };

            check_and_fill(output.total, static_cast<Float>(0));
            check_and_fill(output.detected, static_cast<Integer>(0));
            check_and_fill(output.max_count, pick_fill_value<Float>());
            check_and_fill(output.max_index, static_cast<Integer>(0));

            for (size_t s = 0; s < subsets.size(); ++s) {
                if (!output.subset_total.empty()) {
                    check_and_fill(output.subset_total[s], static_cast<Float>(0));
                }
                if (!output.subset_detected.empty()) {
                    check_and_fill(output.subset_detected[s], static_cast<Integer>(0));
                }
            }
        }

        if (mat->sparse()) {
            if (mat->prefer_rows()) {
                compute_running_sparse(mat, subsets, output);
            } else {
                compute_direct_sparse(mat, subsets, output);
            }
        } else {
            if (mat->prefer_rows()) {
                compute_running_dense(mat, subsets, output);
            } else {
                compute_direct_dense(mat, subsets, output);
            }
        }
    }
};

}

#endif
