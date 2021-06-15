#ifndef MODEL_GENE_VAR_H
#define MODEL_GENE_VAR_H

#include "tatami/base/typed_matrix.hpp"
#include "tatami/stats/sums.hpp"
#include "tatami/stats/variances.hpp"

#include <algorithm>
#include <vector>

namespace scran {

class ModelGeneVar {
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
    struct Results {
        Results(size_t ncells, int nblocks) : means(nblocks, std::vector<double>(ncells)),
                                              variances(nblocks, std::vector<double>(ncells)),
                                              fitted(nblocks, std::vector<double>(ncells)),
                                              residual(nblocks, std::vector<double>(ncells)) {}
        std::vector<std::vector<double> > means, variances, trend, residual;
    };

public:
    template<typename T, typename IDX>
    void run(const tatami::typed_matrix<T, IDX>* mat, 
             std::vector<double*> means,
             std::vector<double*> variances,
             std::vector<double*> fitted,
             std::vector<double*> residual)
    {
        // Estimating the raw values.
        size_t NR = mat->nrow(), NC = mat->ncol();

        if (mat->prefer_rows()) {
            std::vector<T> obuffer(dim);
            auto wrk = p->new_workspace(true);

            if (p->sparse()) {
                std::vector<IDX> ibuffer(dim);
                for (size_t i = 0; i < otherdim; ++i) {
                    auto range = p->sparse_row(i, obuffer.data(), ibuffer.data(), wrk.get());
                }
            } else {
                for (size_t i = 0; i < otherdim; ++i) {
                    auto ptr = p->row(i, obuffer.data(), wrk.get());
                }
            }
        } else {
            std::vector<T> obuffer(otherdim);
            auto wrk = p->new_workspace(false);

            if (p->sparse()) {
                std::vector<IDX> ibuffer(otherdim);
                for (size_t i = 0; i < dim; ++i) {
                    auto range = p->sparse_column(i, obuffer.data(), ibuffer.data(), wrk.get());
                }
            } else {
                for (size_t i = 0; i < dim; ++i) {
                    auto ptr = p->column(i, obuffer.data(), wrk.get());
                }
            }
        }

        // Applying the trend fit to each thing.

        return;
    }
    
private:
    double span = 0;
    std::vector<std::vector<size_t> > by_group;
    size_t group_ncells = 0;

private:
};

}

#endif
