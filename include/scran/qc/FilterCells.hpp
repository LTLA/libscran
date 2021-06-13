#ifndef SCRAN_FILTER_CELLS_H
#define SCRAN_FILTER_CELLS_H

#include "tatami/base/DelayedSubset.hpp"
#include <vector>
#include <numeric>
#include <algorithm>

namespace {

template<typename X = uint8_t>
class FilterCells {
public:
    template<typename SIT>
    FilterCells& add_filter_retain(size_t n, SIT ptr) {
        keep.resize(n, true);
        for (size_t i = 0; i < n; ++i, ++ptr) {
            keep[i] &= *ptr;
        }
        return *this;
    }

    template<typename SIT>
    FilterCells& add_filter_discard(size_t n, SIT ptr) {
        keep.resize(n, true);
        for (size_t i = 0; i < n; ++i, ++ptr) {
            keep[i] &= !(*ptr);
        }
        return *this;
    }

    FilterCells& clear_filter() {
        keep.clear();
        return *this;
    }

    template<class MAT, typename IDX = int>
    std::shared_ptr<MAT> run(std::shared_ptr<MAT> mat) {
        if (mat->ncol() != keep.size()) {
            throw std::runtime_error("number of columns and length of filtering vector is not the same");
        }

        auto num = std::accumulate(keep.begin(), keep.end(), static_cast<IDX>(0));
        std::vector<IDX> retained(num);
        auto rIt = retained.begin();

        for (IDX i = 0; i < static_cast<IDX>(keep.size()); ++i) {
            if (keep[i]) {
                *rIt = i;
                ++rIt;
            }
        }
    
        return tatami::make_DelayedSubset<1>(mat, retained);
    }

private:  
    std::vector<X> keep;
};

}

#endif
