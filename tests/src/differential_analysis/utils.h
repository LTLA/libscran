#ifndef SCRANTEST_DIFFERENTIAL_ANALYSIS_UTILS_HPP
#define SCRANTEST_DIFFERENTIAL_ANALYSIS_UTILS_HPP

#include "tatami/base/Matrix.hpp"
#include "tatami/utils/convert_to_dense.hpp"
#include "tatami/utils/convert_to_sparse.hpp"

#include "../data/Simulator.hpp"

#include <memory>
#include <algorithm>

class DifferentialAnalysisTestCore {
protected:
    std::shared_ptr<tatami::NumericMatrix> dense_row, dense_column, sparse_row, sparse_column;

    void assemble(size_t nr = 100, size_t nc = 50) {
        auto mat = Simulator().matrix(nr, nc);
        dense_row.reset(new decltype(mat)(std::move(mat)));
        dense_column = tatami::convert_to_dense(dense_row.get(), 1);
        sparse_row = tatami::convert_to_sparse(dense_row.get(), 0);
        sparse_column = tatami::convert_to_sparse(dense_row.get(), 1);
    }

    static std::vector<int> create_groupings(size_t n, int ngroups) {
        std::vector<int> groupings(n);
        for (size_t g = 0; g < groupings.size(); ++g) {
            groupings[g] = g % ngroups;
        }
        return groupings;
    }

    static std::vector<int> create_blocks(size_t n, int nblocks) {
        size_t per_block = std::ceil(static_cast<double>(n) / nblocks);
        std::vector<int> blocks;
        blocks.reserve(n);
        for (int b = 0; b < nblocks; ++b) {
            size_t extend_to = std::min(per_block + blocks.size(), n);
            blocks.resize(extend_to, b);
        }
        return blocks;
    }

protected:
    struct EffectsOverlord {
        EffectsOverlord(bool a, size_t nrows, int ngroups) : do_auc(a), store(nrows * ngroups * ngroups) {}

        bool needs_auc() const {
            return do_auc;
        }

        double* prepare_auc_buffer(size_t i, int ngroups) {
            return store.data() + i * ngroups * ngroups;
        }

        bool do_auc;
        std::vector<double> store;
    };
};

#endif
