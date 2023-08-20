#ifndef SCRAN_NORMALIZATION_SANITIZE_SIZE_FACTORS_HPP
#define SCRAN_NORMALIZATION_SANITIZE_SIZE_FACTORS_HPP

#include <limits>
#include <cmath>

namespace scran {

/**
 * @brief Validity of size factors.
 */
struct SizeFactorValidity {
    /**
     * Whether negative factors were detected.
     */
    bool has_negative = false;

    /**
     * Whether size factors of zero were detected.
     */
    bool has_zero = false;

    /**
     * Whether NaN size factors were detected.
     */
    bool has_nan = false;

    /**
     * Whether size factors of positive infinity were detected.
     */
    bool has_infinite = false;
};

/**
 * Check whether there are any invalid size factors.
 * Size factors are only technically valid if they are finite and positive.
 *
 * @tparam T Floating-point type for the size factors.
 *
 * @param n Number of size factors.
 * @param[in] size_factors Pointer to an array of size factors of length `n`.
 *
 * @return Validation results, indicating whether any zero or non-finite size factors exist.
 */
template<typename T>
SizeFactorValidity validate_size_factors(size_t n, const T* size_factors) {
    SizeFactorValidity output;

    for (size_t i = 0; i < n; ++i) {
        auto sf = size_factors[i];
        if (sf < 0) {
            output.has_negative = true;
        } else if (sf == 0) {
            output.has_zero = true;
        } else if (std::isnan(sf)) {
            output.has_nan = true;
        } else if (std::isinf(sf)) {
            output.has_infinite = true;
        }
    }

    return output;
}

/**
 * @brief Sanitize invalid size factors.
 *
 * Replace zero, missing or infinite values in the size factor array so that it can be used to compute well-defined normalized values.
 * Such size factors can occasionally arise if, e.g., insufficient quality control was performed upstream.
 * Check out the documentation in `set_handle_zero()`, `set_handle_negative()`, and `set_handle_nan()` and `set_handle_infinite()` for more details.
 *
 * missing size factors with unity,
 * and infinite size factors with the largest finite size factor.
 */
class SanitizeSizeFactors {
public:
    /**
     * How invalid size factors should be handled:
     *
     * - `IGNORE`: ignore invalid size factors with no error or change.
     * - `ERROR`: throw an error.
     * - `SANITIZE`: fix each invalid size factor.
     */
    enum class HandlerAction : char { IGNORE, ERROR, SANITIZE };

    /**
     * @brief Default parameters.
     */
    struct Defaults {
        /**
         * Set `set_handle_zero()` for more details.
         */
        static constexpr HandlerAction handle_zero = HandlerAction::ERROR;

        /**
         * Set `set_handle_negative()` for more details.
         */
        static constexpr HandlerAction handle_negative = HandlerAction::ERROR;

        /**
         * Set `set_handle_nan()` for more details.
         */
        static constexpr HandlerAction handle_nan = HandlerAction::ERROR;

        /**
         * Set `set_handle_infinite()` for more details.
         */
        static constexpr HandlerAction handle_infinite = HandlerAction::ERROR;
    };

private:
    HandlerAction handle_zero = Defaults::handle_zero;
    HandlerAction handle_negative = Defaults::handle_negative;
    HandlerAction handle_nan = Defaults::handle_nan;
    HandlerAction handle_infinite = Defaults::handle_infinite;

public:
    /**
     * How should we handle zero size factors?
     * If `SANITIZE`, they will be automatically set to the smallest valid size factor (or 1, if all size factors are invalid).
     *
     * This approach is motivated by the observation that size factors of zero are typically generated from all-zero cells.
     * By replacing the size factor with a finite value, we ensure that any all-zero cells are represented by all-zero columns in the normalized matrix,
     * which is a reasonable outcome if those cells cannot be filtered out during upstream quality control.
     *
     * We also need to handle cases where a zero size factor may be generated from a cell with non-zero rows, e.g., with `MedianSizeFactors`.
     * By using a "relatively small" replacement value, we ensure that the normalized values reflect the extremity of the scaling.
     *
     * @param h How to handle a size factor of zero.
     *
     * @return A reference to this `SanitizeSizeFactors` object.
     */
    SanitizeSizeFactors& set_handle_zero(HandlerAction h = Defaults::handle_zero) {
        handle_zero = h;
        return *this;
    }

    /**
     * How should we handle negative size factors?
     * If `SANITIZE`, they will be automatically set to the smallest valid size factor (or 1, if all size factors are invalid).
     * This approach follows the same logic as `set_handle_zero()`, though negative size factors are quite unusual.
     *
     * @param h How to handle a negative size factor.
     *
     * @return A reference to this `SanitizeSizeFactors` object.
     */
    SanitizeSizeFactors& set_handle_negative(HandlerAction h = Defaults::handle_negative) {
        handle_negative = h;
        return *this;
    }

    /**
     * How should we handle NaN size factors?
     * If `SANITIZE, NaN size factors will be automatically set to 1, meaning that scaling is a no-op.
     *
     * @param h How to handle NaN size factors.
     *
     * @return A reference to this `SanitizeSizeFactors` object.
     */
    SanitizeSizeFactors& set_handle_nan(HandlerAction h = Defaults::handle_nan) {
        handle_nan = h;
        return *this;
    }

    /**
     * How shuld be handle infinite size factors.
     * If `SANITIZE`, infinite size factors will be automatically set to the largest valid size factor (or 1, if all size factors are invalid).
     * This ensures that any normalized values will be, at least, finite; the choice of a relatively large replacement value reflects the extremity of the scaling.
     *
     * @param h How to handle infinite size factors.
     *
     * @return A reference to this `SanitizeSizeFactors` object.
     */
    SanitizeSizeFactors& set_handle_infinite(HandlerAction h = Defaults::handle_infinite) {
        handle_infinite = h;
        return *this;
    }

public:
    /**
     * Wrapper to both `set_handle_zero()` and `set_handle_negative()` in one call.
     *
     * @param h How to handle non-positive size factors.
     *
     * @return A reference to this `SanitizeSizeFactors` object.
     */
    SanitizeSizeFactors& set_handle_non_positive(HandlerAction h) {
        set_handle_negative(h);
        set_handle_zero(h);
        return *this;
    }

    /**
     * Wrapper to both `set_handle_infinte()` and `set_handle_nan()` in one call.
     *
     * @param h How to handle non-finite size factors.
     *
     * @return A reference to this `SanitizeSizeFactors` object.
     */
    SanitizeSizeFactors& set_handle_non_finite(HandlerAction h) {
        set_handle_infinite(h);
        set_handle_nan(h);
        return *this;
    }

private:
    template<typename T>
    static double find_smallest_valid_factor(size_t n, T* size_factors) {
        auto smallest = std::numeric_limits<double>::infinity();
        for (size_t i = 0; i < n; ++i) {
            auto s = size_factors[i];
            if (smallest > s && s > 0) { // NaN returns false here, and Inf won't be less than the starting 'smallest', so no need to handle these separately.
                smallest = s;
            }
        }

        if (std::isinf(smallest)) {
            smallest = 1;
        }
        return smallest;
    }

    template<typename T>
    static double find_largest_valid_factor(size_t n, T* size_factors) {
        // Replacing them with the largest non-zero size factor, or 1.
        double largest = 0;
        for (size_t i = 0; i < n; ++i) {
            auto s = size_factors[i];
            if (std::isfinite(s) && largest < s) {
                largest = s;
            }
        }

        if (largest == 0) {
            largest = 1;
        }
        return largest;
    }

public:
    /**
     * @tparam T Floating-point type for the size factors.
     *
     * @param n Number of size factors.
     * @param[in,out] size_factors Pointer to an array of positive size factors of length `n`.
     * On output, invalid size factors are replaced.
     * @param status A pre-computed object indicating whether invalid size factors are present in `size_factors`.
     * This can be useful if this information is already provided by, e.g., `CenterSizeFactors::run()`.
     */
    template<typename T>
    void run(size_t n, T* size_factors, const SizeFactorValidity& status) const {
        T smallest = -1;

        if (status.has_negative) {
            if (handle_negative == HandlerAction::ERROR) {
                throw std::runtime_error("detected negative size factor");
            } else if (handle_negative == HandlerAction::SANITIZE) {
                smallest = find_smallest_valid_factor(n, size_factors);
                for (size_t i = 0; i < n; ++i) {
                    auto& s = size_factors[i];
                    if (s < 0) {
                        s = smallest;
                    }
                }
            }
        }

        if (status.has_zero) {
            if (handle_zero == HandlerAction::ERROR) {
                throw std::runtime_error("detected size factor of zero");
            } else if (handle_zero == HandlerAction::SANITIZE) {
                if (smallest < 0) {
                    smallest = find_smallest_valid_factor(n, size_factors);
                }
                for (size_t i = 0; i < n; ++i) {
                    auto& s = size_factors[i];
                    if (s == 0) {
                        s = smallest;
                    }
                }
            }
        }

        if (status.has_nan) {
            if (handle_nan == HandlerAction::ERROR) {
                throw std::runtime_error("detected NaN size factor");
            } else if (handle_nan == HandlerAction::SANITIZE) {
                for (size_t i = 0; i < n; ++i) {
                    auto& s = size_factors[i];
                    if (std::isnan(s)) {
                        s = 1;
                    }
                }
            }
        }

        if (status.has_infinite) {
            if (handle_infinite == HandlerAction::ERROR) {
                throw std::runtime_error("detected infinite size factor");
            } else if (handle_infinite == HandlerAction::SANITIZE) {
                auto largest = find_largest_valid_factor(n, size_factors);
                for (size_t i = 0; i < n; ++i) {
                    auto& s = size_factors[i];
                    if (std::isinf(s)) {
                        s = largest;
                    }
                }
            }
        }
    }

public:
    /**
     * @tparam T Floating-point type for the size factors.
     *
     * @param n Number of size factors.
     * @param[in,out] size_factors Pointer to an array of positive size factors of length `n`.
     * On output, invalid size factors are replaced.
     */
    template<typename T>
    void run(size_t n, T* size_factors) const {
        auto status = validate_size_factors(n, size_factors);
        run(n, size_factors, status);
    }
};

}

#endif
