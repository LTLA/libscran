#ifndef SCRAN_COMPUTE_MEDIAN_MAD_H
#define SCRAN_COMPUTE_MEDIAN_MAD_H

#include "../utils/macros.hpp"

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <cstdint>

#include "utils.hpp"

/**
 * @file ComputeMedianMad.hpp
 *
 * @brief Compute the median and MAD from an array of values.
 */

namespace scran {

/**
 * @brief Compute the median and MAD from an array of values.
 *
 * Pretty much as it says on the can; computes the median of an array of values first,
 * and uses the median to then compute the median absolute deviation (MAD) from that array.
 * We support calculation of medians and MADs for separate "blocks" of observations in the same array.
 * These statistics can be used in `FilterOutliers` to obtain filter thresholds for removing outliers. 
 */
class ComputeMedianMad {
public:
    /**
     * @brief Default parameters.
     */
    struct Defaults {
        /**
         * See `set_log()` for details.
         */
        static constexpr bool log = false;

        /**
         * See `set_median_only()` for details.
         */
        static constexpr bool median_only = false;
    };

private:
    bool log = Defaults::log;

    bool median_only = Defaults::median_only;

public:
    /**
     * @param l Whether to compute the median and MAD after log-transformation of the values.
     * This is useful for defining thresholds based on fold changes from the center.
     * If `true`, all values are assumed to be non-negative.
     *
     * @return A reference to this `ComputeMedianMad` object.
     */
    ComputeMedianMad& set_log(bool l = Defaults::log) {
        log = l;
        return *this;
    }

    /**
     * @param m Whether to only compute the median.
     * If true, `Results::mads` will be filled with NaNs.
     *
     * @return A reference to this `ComputeMedianMad` object.
     */
    ComputeMedianMad& set_median_only(bool m = Defaults::median_only) {
        median_only = m;
        return *this;
    }

public:
    /**
     * @brief Medians and MADs, possibly for multiple blocks.
     *
     * Meaningful instances of this object should generally be constructed by calling the `ComputeMedianMad::run()` methods.
     * Empty instances can be default-constructed as placeholders.
     */
    struct Results {
        /**
         * @cond
         */
        Results(int nblocks=0) : medians(nblocks), mads(nblocks) {}
        /**
         * @endcond
         */

        /**
         * Vector of length equal to the number of blocks.
         * Each value contains the median for that block.
         *
         * Note that this may be NaN if there are no non-NaN observations,
         * or infinite if observations contain non-finite values.
         */
        std::vector<double> medians;

        /**
         * Vector of length equal to the number of blocks.
         * Each value contains the MAD for that block.
         *
         * Note that this may be NaN if there are no non-NaN observations.
         */
        std::vector<double> mads;

        /**
         * Whether the medians and MADs were computed on the (natural) log-transformed values.
         */
        bool log = false;
    };

public:
    /**
     * @tparam Block Integer type, containing the block IDs.
     * 
     * @param n Number of observations.
     * @param[in] block Pointer to an array of block identifiers.
     * If provided, the array should be of length equal to `n`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * If a null pointer is supplied, all observations are assumed to belong to the same block.
     * 
     * @return Vector of indices specifying the start position for each block on the workspace buffer,
     * if observations from successive blocks were arranged contiguously onto the buffer.
     */
    template<typename Block>
    static std::vector<int> compute_block_starts(size_t n, const Block* block) {
        std::vector<int> starts;

        for (size_t i = 0; i < n; ++i) {
            size_t candidate = block[i] + 1;
            if (candidate > starts.size()) {
                starts.resize(candidate);
            }
            ++starts[block[i]];
        }
        
        int sofar = 0;
        for (auto& s : starts) {
            int last = sofar;
            sofar += s;
            s = last;
        }

        return starts;
    }

public:
    /**
     * @tparam Input Numeric type for the input.
     * @tparam Buffer Floating-point type for the buffer.
     *
     * @param n Number of observations.
     * @param[in] metrics Pointer to an array of observations of length `n`.
     * NaNs are ignored.
     * @param buffer Pointer to an array of length `n`, to be used as a workspace.
     *
     * @return Median and MAD for the `metrics`.
     */
    template<typename Input, typename Buffer> 
    Results run(size_t n, const Input* metrics, Buffer* buffer) const {
        Results output(1);
        output.log = log;

        auto copy = buffer;
        for (size_t i = 0; i < n; ++i) {
            if (!std::isnan(metrics[i])) {
                *copy = metrics[i];
                ++copy;
            }
        }

        compute(copy - buffer, buffer, output.medians[0], output.mads[0]);
        return output;
    }

    /**
     * @overload
     * @tparam Input Numeric type for the input.
     *
     * @param n Number of observations.
     * @param[in] metrics Pointer to an array of observations of length `n`.
     * NaNs are ignored.
     *
     * @return Median and the MAD for `metrics`.
     */
    template<typename Input>
    Results run(size_t n, const Input* metrics) const {
        std::vector<double> buffer(n);
        return run(n, metrics, buffer.data());
    } 

public:
    /**
     * @tparam Block Integer type, containing the block IDs.
     * @tparam Input Numeric type for the input.
     * @tparam Buffer Floating-point type for the buffer.
     *
     * @param n Number of observations.
     * @param[in] block Optional pointer to an array of block identifiers.
     * If provided, the array should be of length equal to `ncells`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * If a null pointer is supplied, all observations are assumed to belong to the same block.
     * @param starts Vector of start positions for the observations from each block,
     * typically generated by calling `compute_block_starts()` on `n` and `block`.
     * @param[in] metrics Pointer to an array of observations of length `n`.
     * @param buffer Pointer to an array of length `n`, to be used as a workspace.
     *
     * @return Medians and MADs for each block in `metrics`.
     */
    template<typename Input, typename Block, typename Buffer>
    Results run_blocked(size_t n, const Block* block, std::vector<int> starts, const Input* metrics, Buffer* buffer) const {
        if (block) {
            // Unscrambling into the buffer.
            auto ends = starts;
            for (size_t i = 0; i < n; ++i) {
                if (!std::isnan(metrics[i])) {
                    auto& pos = ends[block[i]];
                    buffer[pos] = metrics[i];
                    ++pos;
                }
            }

            // Using the ranges on the buffer.
            size_t nblocks = starts.size();
            Results output(nblocks);
            output.log = log;
            for (size_t g = 0; g < nblocks; ++g) {
                compute(ends[g] - starts[g], buffer + starts[g], output.medians[g], output.mads[g]);
            }

            return output;
        } else {
            return run(n, metrics, buffer);
        }
    }

    /**
     * @overload
     * @tparam Block Integer type, containing the block IDs.
     * @tparam Input Type of observation, numeric.
     *
     * @param n Number of observations.
     * @param[in] block Optional pointer to an array of block identifiers.
     * If provided, the array should be of length equal to `ncells`.
     * Values should be integer IDs in \f$[0, N)\f$ where \f$N\f$ is the number of blocks.
     * If a null pointer is supplied, all observations are assumed to belong to the same block.
     * @param[in] metrics Pointer to an array of observations of length `n`.
     *
     * @return Median and the MAD for each block in `metrics`.
     */
    template<typename Block, typename Input>
    Results run_blocked(size_t n, const Block* block, const Input* metrics) const {
        std::vector<double> buffer(n);
        return run_blocked(n, block, compute_block_starts(n, block), metrics, buffer.data());
    }

private:
    template<typename Buffer>
    void compute(size_t n, Buffer* ptr, double& median, double& mad) const {
        if (!n) {
            median = std::numeric_limits<double>::quiet_NaN();
            mad = std::numeric_limits<double>::quiet_NaN();
            return;
        }

        if (log) {
            auto copy = ptr;
            for (size_t i = 0; i < n; ++i, ++copy) {
                if (*copy > 0) {
                    *copy = std::log(*copy);
                } else if (*copy == 0) {
                    *copy = -std::numeric_limits<double>::infinity();
                } else {
                    throw std::runtime_error("cannot log-transform negative values");
                }
            }
        }

        // First, getting the median.
        size_t halfway = n / 2;
        median = compute_median(ptr, halfway, n);

        if (median_only || std::isnan(median)) {
            // Giving up.
            mad = std::numeric_limits<double>::quiet_NaN();
            return;
        } else if (std::isinf(median)) {
            // MADs should be no-ops when added/subtracted from infinity. Any
            // finite value will do here, so might as well keep it simple.
            mad = 0;
            return;
        }

        // Now getting the MAD. 
        auto copy = ptr;
        for (size_t i = 0; i < n; ++i, ++copy) {
            *copy = std::abs(*copy - median);
        }
        mad = compute_median(ptr, halfway, n);
        mad *= 1.4826; // for equivalence with the standard deviation under normality.
        return;
    }

    static double compute_median (double* ptr, size_t halfway, size_t n) {
        std::nth_element(ptr, ptr + halfway, ptr + n);
        double medtmp = *(ptr + halfway);

        if (n % 2 == 0) {
            std::nth_element(ptr, ptr + halfway - 1, ptr + n);
            double left = *(ptr + halfway - 1);

            if (std::isinf(left) && std::isinf(medtmp)) {
                if ((left > 0) != (medtmp > 0)) {
                    return std::numeric_limits<double>::quiet_NaN();
                } else {
                    return medtmp;
                }
            } 

            medtmp = (medtmp + left)/2;
        }

        return medtmp;
    }
};

}

#endif
