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
     * Indicate that the filtering vector specifies the cells to retain.
     *
     * @return A reference to this `FilterCells` object.
     */
    FilterCells& set_retain() {
        retain = true;
        return *this;
    }

    /**
     * Indicate that the filtering vector specifies the cells to discard.
     *
     * @return A reference to this `FilterCells` object.
     */
    FilterCells& set_discard() {
        retain = false;
        return *this;
    }

public:
    /**
     * Filter out low-quality cells by subsetting on the columns on the input matrix.
     * This is done in a delayed manner to avoid creating a copy of the matrix -
     * see the `DelayedSubset` class in the [**tatami**](https://github.com/LTLA/tatami) library for details.
     *
     * @tparam MAT A matrix class, typically a `tatami::typed_matrix`.
     * @tparam IDX Integral type to use for the subsetting indices.
     * @tparam X Boolean type for the filtering vector.
     *
     * @param mat Pointer to the input matrix to be subsetted.
     * @param filter The filtering vector.
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

        std::vector<IDX> retained(retain ? num : (NC - num));
        auto rIt = retained.begin();

        for (size_t i = 0; i < NC; ++i) {
            if (retain == (filter[i] != 0)) {
                *rIt = i;
                ++rIt;
            }
        }
    
        return tatami::make_DelayedSubset<1>(mat, retained);
    }

private:  
    bool retain = false;
};

}

#endif
