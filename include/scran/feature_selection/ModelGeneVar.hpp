#ifndef MODEL_GENE_VAR_H
#define MODEL_GENE_VAR_H

#include "tatami/base/typed_matrix.hpp"

#include <algorithm>
#include <vector>

namespace scran {

template<typename SIT = uint8_t>
class ModelGeneVar {
public:
    ModelGeneVar& set_blocks(size_t n, SIT b) {
        blocks = b;
        group_ncells = n;
        return *this;
    }

    ModelGeneVar& set_blocks() {
        blocks = NULL;
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
        size_t nblocks = 1
        if (blocks!=NULL) {
            if (group_ncells != NC) {
                throw std::runtime_error("length of grouping vector is not equal to the number of columns");
            }
            nblocks = *std::max_element(b, b + group_ncells) + 1;
        }

        std::vector<int> block_size(nblocks);
        {
            auto copy = blocks;
            for (size_t j = 0; j < group_ncells; ++j, ++copy) {
                ++block_size[*copy];
            }
        }

        if (mat->prefer_rows()) {
            std::vector<T> obuffer(NC);
            auto wrk = p->new_workspace(true);

            // We use temporary buffers to improve memory locality for frequent
            // write operations, before transferring the result to the actual stores.
            std::vector<double> tmp_means(nblocks), tmp_vars(nblocks);

            if (p->sparse()) {
                std::vector<int> tmp_nzero(nblocks);
                std::vector<IDX> ibuffer(NC);

                for (size_t i = 0; i < NR; ++i) {
                    auto range = p->sparse_row(i, obuffer.data(), ibuffer.data(), wrk.get());
                    std::fill(tmp_means.begin(), tmp_means.end(), 0);
                    std::fill(tmp_vars.begin(), tmp_vars.end(), 0);

                    for (size_t j = 0; j < range.number; ++j) {
                        auto b = blocks[range.index[j]];
                        tmp_means[b] += range.value[j];
                        ++tmp_nzero[b];
                    }

                    for (size_t b = 0; b < nblocks; ++b) {
                        if (block_size[b]) {
                            tmp_means[b] /= block_size[b];
                        }
                    }

                    for (size_t j = 0; j < range.number; ++j) {
                        auto b = blocks[range.index[j]];
                        tmp_vars[b] += (range.value[j] - tmp_means[b]) * (range.value[j] - tmp_means[b]);
                    }

                    for (size_t b = 0; b < nblocks; ++b) {
                        means[b][i] = tmp_means[b];
                        vars[b][i] = tmp_vars[b] + tmp_means[b] * tmp_means[b] * (block_size[b] - tmp_nzero[b]);
                    }
                }
            } else {
                for (size_t i = 0; i < NR; ++i) {
                    auto ptr = p->row(i, obuffer.data(), wrk.get());
                    std::fill(tmp_means.begin(), tmp_means.end(), 0);
                    std::fill(tmp_vars.begin(), tmp_vars.end(), 0);

                    for (size_t j = 0; j < NC; ++j) {
                        auto b = blocks[j];
                        tmp_means[b] += ptr[j];
                    }

                    for (size_t b = 0; b < nblocks; ++b) {
                        if (block_size[b]) {
                            tmp_means[b] /= block_size[b];
                        }
                    }

                    for (size_t j = 0; j < range.number; ++j) {
                        auto b = blocks[j];
                        tmp_vars[b] += (ptr[j] - tmp_means[b]) * (ptr[j] - tmp_means[b]);
                    }

                    for (size_t b = 0; b < nblocks; ++b) {
                        means[b][i] = tmp_means[b];
                        vars[b][i] = tmp_vars[b];
                    }
                }
            }
        } else {
            std::vector<T> obuffer(NR);
            auto wrk = p->new_workspace(false);

            for (size_t b = 0; b < nblocks; ++b) {
                std::fill(means[b], means[b] + NR, 0);
                std::fill(variances[b], variances[b] + NR, 0);
                std::fill(fitted[b], fitted[b] + NR, 0);
                std::fill(residuals[b], residuals[b] + NR, 0);
            }

            if (p->sparse()) {
                std::vector<IDX> ibuffer(NR);
                auto& my_nzero = fitted; // re-using the memory here.

                for (size_t i = 0; i < NC; ++i) {
                    auto range = p->sparse_column(i, obuffer.data(), ibuffer.data(), wrk.get());

                    auto b = blocks[j];
                    double* mean = means[b];
                    double* var = vars[b];
                    double* nzero = my_nzero[b];

                    for (size_t j = 0; j < range.number; ++j, ++range.value, ++range.index) {
                        double& curmean = mean[*range.index];
                        double& curvar = var[*range.index];
                        int& curnzero = nzero[*range.index];
                        ++curnzero;

                        const double delta = *range.value - curmean;
                        curmean += delta/curnzero;
                        curvar += delta*(*range.value - curmean);
                    }
                }

                for (size_t b = 0; b < nblocks; ++b) {
                    if (block_size[i]) {
                        for (size_t i = 0 ; i < NR; ++i) {
                            const double curNZ = my_nzero[b][i];
                            const double ratio = curNZ / block_size[i];
                            auto& curM = means[b][i];
                            variances[b][i] += curM * curM * ratio * (block_size[i] - curNZ);
                            curM *= ratio;
                        }
                    }
                }

            } else {
                for (size_t i = 0; i < NC; ++i) {
                    auto ptr = p->column(i, obuffer.data(), wrk.get());

                    auto b = blocks[j];
                    double* mean = means[b];
                    double* var = vars[b];

                    for (size_t j = 0; j < NR; ++j, ++ptr, ++mean, ++var) {
                        const double delta = *ptr - *mean;
                        *mean += delta/(i+1);
                        *var += delta*(*ptr - *mean);
                    }
                }
            }
        }

        // Applying the trend fit to each thing.

        return;
    }
    
private:
    SIT blocks = NULL;
    size_t group_ncells = 0;
private:
};

}

#endif
