#ifndef SCRAN_LOGNORMCOUNTS_H
#define SCRAN_LOGNORMCOUNTS_H

#include <algorithm>
#include <vector>
#include <numeric>

#include "tatami/base/DelayedIsometricOp.hpp"

#include "../utils/block_indices.hpp"

namespace scran {

class LogNormCounts {
public:
    LogNormCounts() {}

    LogNormCounts& set_pseudo_count (double p = 1) {
        pseudo_count = p;
        return *this;
    }

    LogNormCounts& set_centered (bool c = false) {
        centered = c;
        return *this;
    }

public:
    template<typename SIT>
    LogNormCounts& set_blocks(size_t n, SIT b) {
        block_indices(n, b, by_group);
        group_ncells = n;
        return *this;
    }

    LogNormCounts& set_blocks() {
        by_group.clear();
        group_ncells = 0;
        return *this;
    }

public:
    template<class MAT, class V>
    std::shared_ptr<MAT> run(std::shared_ptr<MAT> mat, V size_factors) {
        if (size_factors.size() != mat->ncol()) {
            throw std::runtime_error("number of size factors and columns are not equal");
        }

        if (!centered) {
            if (by_group.size()) {
                if (group_ncells != mat->ncol()) {
                    throw std::runtime_error("length of grouping vector and number of columns are not equal");
                }
                for (const auto& g : by_group) {
                    if (g.size()) {
                        double mean = 0;
                        for (auto i : g) {
                            mean += size_factors[i];
                        }
                        mean /= g.size();

                        if (mean > 0) {
                            for (auto i : g) {
                                size_factors[i] /= mean;
                            }
                        }
                    }
                }
            } else if (size_factors.size()) {
                double mean = std::accumulate(size_factors.begin(), size_factors.end(), static_cast<double>(0)) / size_factors.size();
                if (mean) {
                    for (auto& x : size_factors) {
                        x /= mean;
                    }
                }
            }
        }

        for (auto x : size_factors) {
            if (x <= 0) {
                throw std::runtime_error("non-positive size factors detected");
            }
        }

        auto div = tatami::make_DelayedIsometricOp(mat, tatami::make_DelayedDivideVectorHelper<true, 1>(std::move(size_factors)));
        if (pseudo_count == 1) {
            return tatami::make_DelayedIsometricOp(div, tatami::DelayedLog1pHelper(2.0));
        } else {
            auto add = tatami::make_DelayedIsometricOp(div, tatami::DelayedAddScalarHelper<double>(pseudo_count));
            return tatami::make_DelayedIsometricOp(add, tatami::DelayedLogHelper(2.0));
        }
    }

private:
    double pseudo_count = 1;
    bool centered = false;

    std::vector<std::vector<size_t> > by_group;
    size_t group_ncells = 0;
};

};

#endif
