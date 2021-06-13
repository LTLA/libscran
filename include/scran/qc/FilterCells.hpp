#ifndef SCRAN_FILTER_CELLS_H
#define SCRAN_FILTER_CELLS_H

#include "tatami/base/DelayedSubset.hpp"
#include <vector>
#include <numeric>
#include <algorithm>

namespace {

template<typename X=uint8_t>
class FilterCells {
public:
    FilterCells(size_t n) : keep(n, true) {}

    template<typename T>
    FilterCells& add_filter_retain(const T* ptr) {
        for (size_t i = 0; i < keep.size(); ++i) {
            keep[i] &= ptr[i];
        }
        return *this;
    }

    template<typename T>
    FilterCells& add_filter_discard(const T* ptr) {
        for (size_t i = 0; i < keep.size(); ++i) {
            keep[i] &= !ptr[i];
        }
        return *this;
    }

    FilterCells& clear_filter() {
        std::fill(keep.begin(), keep.end(), true);
        return *this;
    }

    template<class MAT, typename IDX = int>
    std::shared_ptr<MAT> run(std::shared_ptr<MAT> mat) {
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
