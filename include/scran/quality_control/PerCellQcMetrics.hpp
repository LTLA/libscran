#ifndef SCRAN_PER_CELL_QC_METRICS_HPP
#define SCRAN_PER_CELL_QC_METRICS_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <algorithm>
#include <limits>
#include <cstdint>

#include "tatami/base/Matrix.hpp"
#include "tatami/stats/apply.hpp"
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
    template<typename Subset, typename Float, typename Integer>
    struct Factory {
        Factory(size_t nr, size_t nc, const std::vector<Subset>& subs, Buffers<Float, Integer>& out) : 
            NR(nr), NC(nc), subsets(subs), output(out) {}

        size_t NR, NC;
        const std::vector<Subset>& subsets;
        Buffers<Float, Integer>& output;

    private:
        std::vector<std::vector<int> > subset_indices;

    public:
        void prepare_dense_direct() {
            if (!output.subset_total.empty() || !output.subset_detected.empty()) {
                size_t nsubsets = subsets.size();
                subset_indices.resize(nsubsets);

                for (size_t s = 0; s < nsubsets; ++s) {
                    auto& current = subset_indices[s];
                    const auto& source = subsets[s];
                    for (int i = 0, end = NR; i < end; ++i) {
                        if (source[i]) {
                            current.push_back(i);
                        }
                    }
                }
            }
            return;
        }

        struct DenseDirect {
            DenseDirect(size_t nr, const std::vector<std::vector<int> >& subs, Buffers<Float, Integer>& out) : 
                NR(nr), subset_indices(subs), output(out) {}

            template<typename T>
            void compute(size_t c, const T* ptr) {
                if (output.total) {
                    auto& current = output.total[c];
                    for (size_t r = 0; r < NR; ++r) {
                        current += ptr[r];
                    }
                }

                if (output.detected) {
                    auto& current = output.detected[c];
                    for (size_t r = 0; r < NR; ++r) {
                        current += (ptr[r] != 0);
                    }
                }

                if (output.max_index || output.max_count) {
                    auto max_count = PerCellQcMetrics::pick_fill_value<Float>();
                    Integer max_index = 0;
                    for (size_t r = 0; r < NR; ++r) {
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
                        auto& current = output.subset_total[s][c];
                        for (auto r : sub) {
                            current += ptr[r];
                        }
                    }

                    if (!output.subset_detected.empty() && output.subset_detected[s]) {
                        auto& current = output.subset_detected[s][c];
                        for (auto r : sub) {
                            current += ptr[r] != 0;
                        }
                    }
                }
            }

            size_t NR;
            const std::vector<std::vector<int> >& subset_indices;
            Buffers<Float, Integer>& output;
        };

        DenseDirect dense_direct() {
            return DenseDirect(NR, subset_indices, output);
        }

    public:
        struct SparseDirect {
            SparseDirect(size_t nr, const std::vector<Subset>& subs, Buffers<Float, Integer>& out) : 
                NR(nr), subsets(subs), output(out), internal_is_nonzero(out.max_index ? nr : 0) {}

            template<typename T, typename IDX>
            void compute(size_t c, const tatami::SparseRange<T, IDX>& range) {
                if (output.total) {
                    auto& current = output.total[c];
                    for (size_t i = 0; i < range.number; ++i) {
                        current += range.value[i];
                    }
                }

                if (output.detected) {
                    auto& current = output.detected[c];
                    for (size_t i = 0; i < range.number; ++i) {
                        current += (range.value[i] != 0);
                    }
                }

                if (output.max_index || output.max_count) {
                    auto max_count = PerCellQcMetrics::pick_fill_value<Float>();
                    Integer max_index = 0;
                    for (size_t i = 0; i < range.number; ++i) {
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
                            for (size_t i = 0; i < range.number; ++i) {
                                if (range.value[i]) {
                                    internal_is_nonzero[range.index[i]] = 1;
                                }
                            }
                            for (size_t r = 0; r < NR; ++r) {
                                if (internal_is_nonzero[r] == 0) {
                                    max_index = r;
                                    break;
                                }
                            }
                            for (size_t i = 0; i < range.number; ++i) { // setting back to zero.
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
                        for (size_t i = 0; i < range.number; ++i) {
                            current += (sub[range.index[i]] != 0) * range.value[i];
                        }
                    }

                    if (!output.subset_detected.empty() && output.subset_detected[s]) {
                        auto& current = output.subset_detected[s][c];
                        for (size_t i = 0; i < range.number; ++i) {
                            current += (sub[range.index[i]] != 0) * (range.value[i] != 0);
                        }
                    }
                }
            }

            size_t NR;
            const std::vector<Subset>& subsets;
            Buffers<Float, Integer>& output;
            std::vector<uint8_t> internal_is_nonzero;
        };
        
        SparseDirect sparse_direct() {
            return SparseDirect(NR, subsets, output);
        }

    public:
        struct DenseRunning {
            DenseRunning(size_t n, size_t nr, const std::vector<Subset>& subs, Buffers<Float, Integer> out) :
                num(n), NR(nr), subsets(subs), output(std::move(out)), 
                internal_max_count(output.max_count ? 0 : n, PerCellQcMetrics::pick_fill_value<Float>()) {}

            template<class T>
            void add(const T* ptr) {
                if (output.total) {
                    for (size_t c = 0; c < num; ++c) {
                        output.total[c] += ptr[c];
                    }
                }

                if (output.detected) {
                    for (size_t c = 0; c < num; ++c) {
                        output.detected[c] += (ptr[c] != 0);
                    }
                }

                if (output.max_count || output.max_index) {
                    auto* tracker = (output.max_count ? output.max_count : internal_max_count.data());
                    for (size_t c = 0; c < num; ++c) {
                        if (tracker[c] < ptr[c]) {
                            tracker[c] = ptr[c];
                            if (output.max_index) {
                                output.max_index[c] = counter;
                            }
                        }
                    }
                }

                size_t nsubsets = subsets.size();
                for (size_t s = 0; s < nsubsets; ++s) {
                    const auto& sub = subsets[s];
                    if (sub[counter] == 0) {
                        continue;
                    }

                    if (!output.subset_total.empty() && output.subset_total[s]) {
                        auto& current = output.subset_total[s];
                        for (size_t c = 0; c < num; ++c) {
                            current[c] += ptr[c];
                        }
                    }

                    if (!output.subset_detected.empty() && output.subset_detected[s]) {
                        auto& current = output.subset_detected[s];
                        for (size_t c = 0; c < num; ++c) {
                            current[c] += (ptr[c] != 0);
                        }
                    }
                }

                ++counter;
            }

            size_t counter = 0;
            size_t num;
            size_t NR;
            const std::vector<Subset>& subsets;
            Buffers<Float, Integer> output;
            std::vector<Float> internal_max_count;
        };

        DenseRunning dense_running() {
            return DenseRunning(NC, NR, subsets, output);
        }

        DenseRunning dense_running(size_t start, size_t end) {
            auto advance = [&](auto& ptr) -> void {
                if (ptr) {
                    ptr += start;
                }
            };

            auto copy = output;
            advance(copy.total);
            advance(copy.detected);
            advance(copy.max_count);
            advance(copy.max_index);
            for (auto& s : copy.subset_total) {
                advance(s);
            }
            for (auto& s : copy.subset_detected) {
                advance(s);
            }

            return DenseRunning(end - start, NR, subsets, copy);
        }

    public:
        struct SparseRunning {
            SparseRunning(size_t s, size_t e, size_t nr, const std::vector<Subset>& subs, Buffers<Float, Integer>& out) :
                start(s), end(e), NR(nr), subsets(subs), output(out),
                internal_max_count(output.max_count ? 0 : e - s, PerCellQcMetrics::pick_fill_value<Float>()),
                internal_last_consecutive_nonzero(output.max_index || output.max_count ?  e - s : 0)
                {}

            template<typename T, typename IDX>
            void add (const tatami::SparseRange<T, IDX> range) {
                if (output.total) {
                    for (size_t i = 0; i < range.number; ++i) {
                        output.total[range.index[i]] += range.value[i];
                    }
                }

                if (output.detected) {
                    for (size_t i = 0; i < range.number; ++i) {
                        output.detected[range.index[i]] += (range.value[i] != 0);
                    }
                }

                if (output.max_count || output.max_index) {
                    auto* tracker = (output.max_count ? output.max_count + start : internal_max_count.data());
                    for (size_t i = 0; i < range.number; ++i) {
                        auto offset = range.index[i] - start;

                        auto& curmax = tracker[offset];
                        if (curmax < range.value[i]) {
                            curmax = range.value[i];
                            if (output.max_index) {
                                output.max_index[range.index[i]] = counter;
                            }
                        }

                        auto& last = internal_last_consecutive_nonzero[offset];
                        if (static_cast<size_t>(last) == counter) {
                            if (range.value[i] != 0) {
                                ++last;
                            }
                        }
                    }
                }

                size_t nsubsets = subsets.size();
                for (size_t s = 0; s < nsubsets; ++s) {
                    const auto& sub = subsets[s];
                    if (sub[counter] == 0) {
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

                ++counter;
            }

            void finish() {
                if (output.max_count || output.max_index) {
                    auto* tracker = (output.max_count ? output.max_count + start : internal_max_count.data());

                    // Checking anything with non-positive max count.
                    for (size_t s = start; s < end; ++s) {
                        auto& current = tracker[s - start];
                        if (current > 0) {
                            continue;
                        }

                        auto last_nz = internal_last_consecutive_nonzero[s - start];
                        if (last_nz == NR) {
                            continue;
                        }

                        current = 0;
                        if (output.max_index) {
                            output.max_index[s] = last_nz;
                        }
                    }
                }
            }

            const size_t start, end;
            size_t counter = 0;
            size_t NR;
            const std::vector<Subset>& subsets;
            Buffers<Float, Integer>& output;
            std::vector<Float> internal_max_count;
            std::vector<Integer> internal_last_consecutive_nonzero;
        };
         
        SparseRunning sparse_running() {
            return SparseRunning(0, NC, NR, subsets, output);
        }

        SparseRunning sparse_running(size_t start, size_t end) {
            return SparseRunning(start, end, NR, subsets, output);
        }
    };

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

        size_t nr = mat->nrow(), nc = mat->ncol();
        Factory fact(nr, nc, subsets, output);
        tatami::apply<1>(mat, fact, num_threads);
        return;
    }
};

}

#endif
