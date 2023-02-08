#ifndef SCRAN_HYPERGEOMETRIC_TAIL_HPP
#define SCRAN_HYPERGEOMETRIC_TAIL_HPP

#include "../utils/macros.hpp"

#include <cmath>

/**
 * @file HypergeometricTail.hpp
 *
 * @brief Compute hypergeometric tail probabilities.
 */

namespace scran {

/**
 * @brief Compute hypergeometric tail probabilities.
 *
 * This computes the tail probabilities for the hypergeometric distribution.
 * It is intended for use in quantifying feature set enrichment in marker lists.
 * The white balls are the features in the set, the black balls are all other features, and the drawing process involves picking the top N markers;
 * our aim is to compute the p-value for enrichment of features in the set among the top markers.
 */
class HypergeometricTail {
public:
    /**
     * @brief Default parameters.
     */
    struct Defaults {
        /**
         * See `set_log()` for more details.
         */
        static constexpr bool log = false;

        /**
         * See `set_upper_tail()` for more details.
         */
        static constexpr bool upper_tail = true;
    };

private:
    bool log = Defaults::log;
    bool upper_tail = Defaults::upper_tail;

public:
    /**
     * @param l Whether to report log-probabilities, which avoids underflow for very small values.
     * @return A reference to this `HypergeometricTail` instance.
     */
    HypergeometricTail& set_log(bool l = Defaults::log) {
        log = l;
        return *this;
    }

    /**
     * @param u Whether to report the upper tail.
     * If `false`, the lower tail is returned.
     * @return A reference to this `HypergeometricTail` instance.
     */
    HypergeometricTail& set_upper_tail(bool u = Defaults::upper_tail) {
        upper_tail = u;
        return *this;
    }

public:
    /**
     * @brief Cached statistics for re-use across `run()` calls.
     */
    struct Cache {
        /**
         * @cond
         */
        int drawn_white = 0;
        int num_white = 0;
        int num_black = 0;
        int num_drawn = 0;

        std::vector<long double> cumulative;
        long double probability = 0;
        double scale = 0;
        /**
         * @endcond
         */
    };

    /**
     * Create a new cache for use in `run()`.
     *
     * @return A new `Cache` instance.
     */
    Cache new_cache() {
        return Cache();
    }

private:
    // Using Stirling's approximation compute choose(num_black, num_drawn) / choose(num_black + num_white, num_drawn).
    static double stirling(double x) {
        return x * std::log(x) - x + 0.5 * std::log(2 * 3.14159265358979323846 * x);
    }

private:
    /*
     * Computing the cumulative sum after factorizing out choose(num_white + num_black, num_drawn), obviously.
     *
     * If num_drawn < num_black, we also factorize out choose(num_black, num_drawn), so that we can compute the series by simply increasing 'k'.
     * This avoids the need to do two passes where the first pass computes choose(num_black, num_drawn - drawn_white) and the second pass handles the summation.
     *
     * Otherwise, we factorize out choose(num_white, num_drawn - num_black), which is the first non-zero probability mass in the series.
     * This reduces the size of the summed terms to avoid overflow, and is reflected in 'start > 1'.
     * 
     * We can check the accuracy of our calculations with:
     * sum(choose(num_white, 0:drawn_white) * choose(num_black, num_drawn - 0:drawn_white)) / max(choose(num_white, num_drawn - num_black), choose(num_black, num_drawn)) - 1
     */
    static long double compute_cumulative(int drawn_white, int num_white, int num_black, int num_drawn) {
        // We use long double for some extra precision here.
        long double probability = 1;

        // We need to add 1 for the starting probability, but we'll do this in compute_tail_probability() to make use of the more precise log1p.
        long double cumulative = 0; 

        int start = num_drawn - num_black + 1;
        if (start <= 0) {
            start = 1;
        }

        for (int k = start; k <= drawn_white; ++k) {
            probability *= static_cast<double>(num_white - k + 1) * static_cast<double>(num_drawn - k + 1) / static_cast<double>(k) / static_cast<double>(num_black - num_drawn + k);
            cumulative += probability;
        }

        return cumulative;
    }

private:
    static void compute_cumulative(int drawn_white, int num_white, int num_black, int num_drawn, Cache& cache) {
        auto& probability = cache.probability;
        auto& cumulative = cache.cumulative;
        auto& k = cache.drawn_white;

        if (num_white == cache.num_white && num_black == cache.num_black && num_drawn == cache.num_drawn) {
            if (k >= drawn_white) {
                return;
            }
        } else {
            cumulative.clear();
            cumulative.push_back(0);
            probability = 1;
            k = 0;
        }

        cumulative.reserve(drawn_white + 1);

        // Same logic as above, but we now store the probability at each step.
        while (k < drawn_white) {
            ++k;
            if (num_drawn - k < num_black) {
                probability *= static_cast<double>(num_white - k + 1) * static_cast<double>(num_drawn - k + 1) / static_cast<double>(k) / static_cast<double>(num_black - num_drawn + k);
                cumulative.push_back(cumulative.back() + probability);
            } else {
                cumulative.push_back(0);
            }
        }
    }

    static double compute_log_scale(int num_white, int num_black, int num_drawn) {
        int num_total = num_white + num_black;
        if (num_black > num_drawn) {
            // Approximates lchoose(num_black, num_drawn) - lchoose(num_total, num_drawn).
            return stirling(num_black) - stirling(num_black - num_drawn)
                - (stirling(num_total) - stirling(num_total - num_drawn));
        } else if (num_drawn > num_black) {
            // Approximates lchoose(num_white, num_drawn - num_black) - lchoose(num_total, num_drawn).
            return stirling(num_white) - stirling(num_drawn - num_black)
                - (stirling(num_total) - stirling(num_drawn));
        } else {
            // Approximates -lchoose(num_total, num_drawn).
            return -(stirling(num_total) - stirling(num_total - num_drawn) - stirling(num_drawn));
        }
    }

private:
    static double invert_tail_log(double val) {
        // Logic from https://github.com/SurajGupta/r-source/blob/master/src/nmath/dpq.h;
        // if 'lp' is close to zero, exp(lp) will be close to 1, and thus the precision of
        // expm1 is more important. If 'lp' is large and negative, exp(lp) will be close to
        // zero, and thus the precision of log1p is more important.
        if (val > -std::log(2)) {
            auto p = -std::expm1(val);
            return (p > 0 ? std::log(p) : -std::numeric_limits<double>::infinity());
        } else {
            auto p = -std::exp(val);
            return (p > -1 ? std::log1p(p) : -std::numeric_limits<double>::infinity());
        }
    }

    double compute_tail_probability(long double cumulative, double scale, bool do_lower) const {
        if (log) {
            double lp = std::log1p(cumulative) + scale;
            if (do_lower) {
                return lp;
            } else {
                return invert_tail_log(lp);
            }
        } else {
            auto p = (1 + cumulative) * std::exp(scale);
            return (do_lower ? p : 1 - p);
        }
    }

    double edge_handler(bool zero) const {
        if (zero) {
            return (log ? -std::numeric_limits<double>::infinity() : 0);
        } else {
            return (log ? 0 : 1);
        }
    }

private:
    double core(int drawn_white, int num_white, int num_black, int num_drawn, Cache* cache) const {
        if (drawn_white <= 0 || drawn_white < num_drawn - num_black) {
            return edge_handler(!upper_tail);
        } else if (drawn_white >= num_drawn || drawn_white >= num_white) {
            return edge_handler(upper_tail);
        }

        // Flipping the tails to avoid having to calculate large summations.
        // While more efficient, the real reason is that it avoids having to
        // subtract large sums from 1 when computing upper tails, which could
        // result in catastrophic cancellation. 
        bool do_lower = !upper_tail;
        if (drawn_white * (num_white + num_black) > num_drawn * num_white) { 
            auto tmp = num_white;
            num_white = num_black;
            num_black = tmp;
            drawn_white = num_drawn - drawn_white - 1;
            do_lower = !do_lower;
        }

        if (cache == NULL) {
            double logscale = compute_log_scale(num_white, num_black, num_drawn);
            auto cum = compute_cumulative(drawn_white, num_white, num_black, num_drawn);
            return compute_tail_probability(cum, logscale, do_lower);
        }

        // Picking up from cached values if we can.
        compute_cumulative(drawn_white, num_white, num_black, num_drawn, *cache);

        double& logscale = cache->scale;
        if (!(num_white == cache->num_white && num_black == cache->num_black && num_drawn == cache->num_drawn)) {
            logscale = compute_log_scale(num_white, num_black, num_drawn);
            cache->num_white = num_white;
            cache->num_black = num_black;
            cache->num_drawn = num_drawn;
        }

        return compute_tail_probability(cache->cumulative[drawn_white], cache->scale, do_lower);
    }

public:
    /**
     * @param drawn_white Number of white balls that were drawn.
     * @param num_white Number of white balls in the pot.
     * @param num_black Number of black balls in the plot.
     * @param num_drawn Number of balls that were drawn from the pot.
     *
     * @return Probability of drawing a number of white balls greater than than or equal to `drawn_white`, if `set_upper_tail()` is set to true.
     * Otherwise, the probability of drawing a number less than or equal to `drawn_white` is returned.
     */
    double run(int drawn_white, int num_white, int num_black, int num_drawn) const {
        return core(drawn_white, num_white, num_black, num_drawn, NULL);
    }

    /**
     * This re-uses cached intermediate values for the cumulative probability if all inputs other than `drawn_white` are the same.
     * For greatest efficiency, it is best to group calls to `run()` such that `drawn_white` is the fastest-changing value and other inputs are constant, allowing the cache to be effectively re-used.
     *
     * @param drawn_white Number of white balls that were drawn.
     * @param num_white Number of white balls in the pot.
     * @param num_black Number of black balls in the plot.
     * @param num_drawn Number of balls that were drawn from the pot.
     * @param cache A `Cache` object, typically constructed with `new_cache()`.
     *
     * @return Probability of drawing a number of white balls greater than than or equal to `drawn_white`, if `set_upper_tail()` is set to true.
     * Otherwise, the probability of drawing a number less than or equal to `drawn_white` is returned.
     */
    double run(int drawn_white, int num_white, int num_black, int num_drawn, Cache& cache) const {
        return core(drawn_white, num_white, num_black, num_drawn, &cache);
    }
};

}

#endif
