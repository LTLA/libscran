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
 * The "successes" are the features in the set, the "failures" are all other features, and the drawing process typically involves picking the top N markers;
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
     * @param u Whether to report the upper tail, including the probability mass of the observed number of drawn white balls.
     * This allows the tail probability to be directly used as the p-value for testing enrichment.
     * If `false`, the lower tail is returned, again including the probability mass of `drawn_inside`.
     *
     * @return A reference to this `HypergeometricTail` instance.
     */
    HypergeometricTail& set_upper_tail(bool u = Defaults::upper_tail) {
        upper_tail = u;
        return *this;
    }

public: // only exported for testing.
    /**
     * @cond
     */
    static double lfactorial(int x) {
        // Computing it exactly for small numbers, to avoid unnecessarily
        // large relative inaccuracy from the approximation. Threshold of
        // 12 is chosen more-or-less arbitrarily... but 12! is the largest
        // value that can be represented by a 32-bit int, if that helps.
        switch(x) {
            case 0: case 1: return 0;
            case 2: return std::log(2.0); 
            case 3: return std::log(6.0); 
            case 4: return std::log(24.0); 
            case 5: return std::log(120.0); 
            case 6: return std::log(720.0); 
            case 7: return std::log(5040.0); 
            case 8: return std::log(40320.0); 
            case 9: return std::log(362880.0); 
            case 10: return std::log(3628800.0); 
            case 11: return std::log(39916800.0); 
            case 12: return std::log(479001600.0); 
        }

        // For large numbers, using Ramanujan's approximation rather than R's complicated thing. 
        // Check out https://www.johndcook.com/blog/2012/09/25/ramanujans-factorial-approximation/.
        double y = x;
        return 1.0/6.0 * std::log(y * (1 + 4 * y * (1 + 2 * y)) + 1.0/30.0) + y * std::log(y) - y + 0.5 * std::log(3.14159265358979323846);
    }
    /**
     * @endcond
     */

private:
    /*
     * Computing the cumulative sum after factorizing out the probability mass at drawn_inside.
     * This allows us to do one pass from k to 0 to compute the probability.
     * 
     * We can check the accuracy of our calculations with:
     * sum(choose(num_inside, 0:drawn_inside) * choose(num_outside, num_drawn - 0:drawn_inside)) / max(choose(num_inside, num_drawn - num_outside), choose(num_outside, num_drawn)) - 1
     */
    static long double compute_cumulative(int drawn_inside, int num_inside, int num_outside, int num_drawn) {
        // Improved precision for this step, possibly involving small probabilities.
        long double probability = 1;

        // We need to add 1 for the probability mass at drawn_inside,
        // but we'll do this in compute_tail_probability() to make use of the more precise log1p.
        long double cumulative = 0; 

        for (int k = drawn_inside; k > 0 && probability > 0; --k) {
            probability *= static_cast<double>(k) * static_cast<double>(num_outside - num_drawn + k) / static_cast<double>(num_inside - k + 1) / static_cast<double>(num_drawn - k + 1);
            cumulative += probability;
        }

        return cumulative;
    }

    static double compute_probability_mass(int drawn_inside, int num_inside, int num_outside, int num_drawn) {
        int num_total = num_inside + num_outside;
        return lfactorial(num_inside) - lfactorial(drawn_inside) - lfactorial(num_inside - drawn_inside) // lchoose(num_inside, drawn_inside)
            + lfactorial(num_outside) - lfactorial(num_drawn - drawn_inside) - lfactorial(num_outside - num_drawn + drawn_inside) // lchoose(num_outside, num_drawn - drawn_inside)
            - lfactorial(num_total) + lfactorial(num_drawn) + lfactorial(num_total - num_drawn); // -lchoose(num_total, num_drawn)
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
    double core(int drawn_inside, int num_inside, int num_outside, int num_drawn) const {
        // Subtracting 1 to include the PMF of 'drawn_inside' in the upper tail calculations.
        if (upper_tail) {
            --drawn_inside;
        }

        if (drawn_inside <= 0 || drawn_inside < num_drawn - num_outside) {
            return edge_handler(!upper_tail);
        } else if (drawn_inside >= num_drawn || drawn_inside >= num_inside) {
            return edge_handler(upper_tail);
        }

        // Flipping the tails to avoid having to calculate large summations.
        // This it avoids having to subtract large sums from 1 when computing
        // upper tails, which could result in catastrophic cancellation. It can
        // also sometimes improve efficiency by summing over the smaller tail. 
        bool do_lower = !upper_tail;
        if (drawn_inside * (num_inside + num_outside) > num_drawn * num_inside) { 
            auto tmp = num_inside;
            num_inside = num_outside;
            num_outside = tmp;
            drawn_inside = num_drawn - drawn_inside - 1;
            do_lower = !do_lower;
        }

        double logscale = compute_probability_mass(drawn_inside, num_inside, num_outside, num_drawn);
        auto cum = compute_cumulative(drawn_inside, num_inside, num_outside, num_drawn);
        return compute_tail_probability(cum, logscale, do_lower);
    }

public:
    /**
     * @param drawn_inside Number of genes inside the set that were drawn.
     * @param num_inside Total number of genes in the set.
     * @param num_outside Total number of genes outside the set.
     * @param num_drawn Number of genes that were drawn.
     *
     * @return Probability of randomly drawing at least `drawn_inside` genes from the set, if `set_upper_tail()` is set to true.
     * Otherwise, the probability of randomly drawing no more than `drawn_inside` genes from the set is returned.
     */
    double run(int drawn_inside, int num_inside, int num_outside, int num_drawn) const {
        return core(drawn_inside, num_inside, num_outside, num_drawn);
    }
};

}

#endif
