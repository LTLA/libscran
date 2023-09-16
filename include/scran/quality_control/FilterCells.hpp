#ifndef SCRAN_FILTER_CELLS_H
#define SCRAN_FILTER_CELLS_H

#include "../utils/macros.hpp"

#include "tatami/tatami.hpp"
#include <vector>
#include <numeric>
#include <algorithm>

/**
 * @file FilterCells.hpp
 *
 * @brief Filter out low-quality cells from the count matrix.
 */

namespace scran {

/**
 * @brief Filter out low-quality cells.
 *
 * This class removes low-quality cells from the input matrix by subsetting on the columns,
 * typically using the filters defined in `PerCellRnaQcFilters` and/or `PerCellAdtQcFilters`.
 * It is effectively a wrapper around the `tatami::make_DelayedSubset` function.
 */
class FilterCells {
public:
    /**
     * @brief Default choices for all parameters.
     */
    struct Defaults {
        /**
         * See `set_discard()` for more details.
         */
        static constexpr bool discard = true;

        /**
         * See `set_intersect()` for more details.
         */
        static constexpr bool intersect = false;
    };

private:
    bool discard = Defaults::discard;
    bool intersect = Defaults::intersect;

public:
    /**
     * Indicate that the filtering vector specifies the cells to retain.
     *
     * @return A reference to this `FilterCells` object.
     */
    FilterCells& set_retain() {
        discard = false;
        return *this;
    }

    /**
     * Indicate that the filtering vector specifies the cells to discard.
     *
     * @return A reference to this `FilterCells` object.
     */
    FilterCells& set_discard(bool d = Defaults::discard) {
        discard = d;
        return *this;
    }

    /**
     * @param i Whether to take the intersection or union of the filtering vectors,
     * if multiple filtering vectors are supplied.
     *
     * @return A reference to this `FilterCells` object.
     */
    FilterCells& set_intersect(bool i = Defaults::intersect) {
        intersect = i;
        return *this;
    }

public:
    /**
     * Filter out low-quality cells by subsetting on the columns on the input matrix.
     * This is done in a delayed manner to avoid creating a copy of the matrix -
     * see the `DelayedSubset` class in the [**tatami**](https://github.com/LTLA/tatami) library for details.
     *
     * @tparam MAT A matrix class, typically a `tatami::Matrix`.
     * @tparam IDX Integral type to use for the subsetting indices.
     * @tparam X Boolean type for the filtering vector.
     *
     * @param mat Pointer to the input matrix to be subsetted.
     * @param filter Pointer to a filtering vector of length equal to the number of columns in `mat`.
     *
     * @return A pointer to the submatrix.
     */
    template<class MAT, typename IDX = int, typename X = uint8_t>
    std::shared_ptr<MAT> run(std::shared_ptr<MAT> mat, const X* filter) {
        size_t NC = mat->ncol();

        // Don't use accumulate: we want to defend against non-0/1 values in 'filter'.
        IDX num = 0;
        for (size_t i = 0; i < NC; ++i) {
            num += (filter[i] != 0);
        }

        std::vector<IDX> retained;
        retained.reserve(discard ? (NC - num) : num);

        for (size_t i = 0; i < NC; ++i) {
            if (discard == (filter[i] == 0)) {
                retained.push_back(i);
            }
        }

        return tatami::make_DelayedSubset<1>(mat, retained);
    }

public:
    /**
     * Filter out low-quality cells by subsetting on the columns on the input matrix,
     * taking the intersection or union of multiple filtering vectors according to `set_intersect()`.
     * The intersection is defined by setting the filtering value to 1 for a cell if the corresponding values across all supplied vectors are also 1, and zero otherwise;
     * the union is defined by setting the filtering value to 1 for a cell if any corresponding value for any supplied vector is 1, and zero otherwise.
     * The actual effect of filtering is determined by `set_discard()`.
     *
     * @tparam MAT A matrix class, typically a `tatami::Matrix`.
     * @tparam IDX Integral type to use for the subsetting indices.
     * @tparam X Boolean type for the filtering vector.
     *
     * @param mat Pointer to the input matrix to be subsetted.
     * @param filters Pointers to filtering vectors.
     * Each vector should be of length equal to the number of columns in `mat`.
     *
     * @return A pointer to the submatrix.
     */
    template<class MAT, typename IDX = int, typename X = uint8_t>
    std::shared_ptr<MAT> run(std::shared_ptr<MAT> mat, const std::vector<X*>& filters) {
        size_t NC = mat->ncol();

        std::vector<uint8_t> finalvec(NC, intersect);
        if (intersect) {
            for (auto filter : filters) {
                for (size_t i = 0; i < NC; ++i) {
                    finalvec[i] &= static_cast<uint8_t>(filter[i] != 0);
                }
            }
        } else {
            for (auto filter : filters) {
                for (size_t i = 0; i < NC; ++i) {
                    finalvec[i] |= static_cast<uint8_t>(filter[i] != 0);
                }
            }
        }

        IDX num = std::accumulate(finalvec.begin(), finalvec.end(), static_cast<IDX>(0));
        std::vector<IDX> retained;
        retained.reserve(discard ? (NC - num) : num);

        for (size_t i = 0; i < NC; ++i) {
            if (discard == (finalvec[i] == 0)) {
                retained.push_back(i);
            }
        }

        return tatami::make_DelayedSubset<1>(mat, retained);
    }
};

}

#endif
