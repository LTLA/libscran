#ifndef SCRAN_FILTER_CELLS_H
#define SCRAN_FILTER_CELLS_H

#include "tatami/base/DelayedSubset.hpp"
#include <vector>
#include <numeric>
#include <algorithm>

namespace scran {

class FilterCells {
public:
    FilterCells& set_retain() {
        retain = true;
        return *this;
    }

    FilterCells& set_discard() {
        retain = false;
        return *this;
    }

public:
    template<class MAT, typename IDX = int, typename X = uint8_t>
    std::shared_ptr<MAT> run(std::shared_ptr<MAT> mat, const X* filter) {
        size_t NC = mat->ncol();
        auto num = std::accumulate(filter, filter + NC, static_cast<IDX>(0));
        std::vector<IDX> retained(retain ? num : (NC - num));
        auto rIt = retained.begin();

        for (IDX i = 0; i < NC; ++i) {
            if (retain == static_cast<bool>(filter[i])) {
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
