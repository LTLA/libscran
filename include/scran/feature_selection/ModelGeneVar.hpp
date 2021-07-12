#ifndef SCRAN_MODEL_GENE_VAR_H
#define SCRAN_MODEL_GENE_VAR_H

#include "tatami/base/Matrix.hpp"
#include "../utils/vector_to_pointers.hpp"
#include "FitTrendVar.hpp"

#include <algorithm>
#include <vector>
#include <limits>

namespace scran {

template<typename BLOCK = int>
class ModelGeneVar {
public:
    ModelGeneVar& set_blocks(const std::vector<BLOCK>& b) {
        set_blocks(b.size(), b.data());
        return *this;
    }

    ModelGeneVar& set_blocks(size_t n, const BLOCK* b) {
        blocks = b;
        group_ncells = n;
        if (n) {
            nblocks = *std::max_element(b, b + n) + 1;
        }
        return *this;
    }

    ModelGeneVar& set_blocks() {
        blocks = NULL;
        group_ncells = 0;
        nblocks = 1;
        return *this;
    }

    ModelGeneVar& set_span() {
        fit.set_span();
        return *this;
    }

    ModelGeneVar& set_span(double s) {
        fit.set_span(s);
        return *this;
    }

public:
    struct Results {
        Results(size_t ngenes, int nblocks) : means(nblocks, std::vector<double>(ngenes)),
                                              variances(nblocks, std::vector<double>(ngenes)),
                                              fitted(nblocks, std::vector<double>(ngenes)),
                                              residuals(nblocks, std::vector<double>(ngenes)) {}
        std::vector<std::vector<double> > means, variances, fitted, residuals;
    };

public:
    template<typename T, typename IDX>
    Results run(const tatami::Matrix<T, IDX>* mat) {
        Results output(mat->nrow(), nblocks);
        run(mat, 
            vector_to_pointers(output.means),
            vector_to_pointers(output.variances),
            vector_to_pointers(output.fitted),
            vector_to_pointers(output.residuals));
        return output;
    }

    template<typename T, typename IDX>
    void run(const tatami::Matrix<T, IDX>* p, 
             std::vector<double*> means,
             std::vector<double*> variances,
             std::vector<double*> fitted,
             std::vector<double*> residuals)
    {
        // Estimating the raw values.
        size_t NR = p->nrow(), NC = p->ncol();
        std::vector<int> block_size(nblocks);

        if (blocks!=NULL) {
            if (group_ncells != NC) {
                throw std::runtime_error("length of grouping vector is not equal to the number of columns");
            }
            auto copy = blocks;
            for (size_t j = 0; j < NC; ++j, ++copy) {
                ++block_size[*copy];
            }
        } else {
            block_size[0] = NC;
        }

        if (p->prefer_rows()) {
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
                    std::fill(tmp_nzero.begin(), tmp_nzero.end(), 0);

                    for (size_t j = 0; j < range.number; ++j) {
                        auto b = (blocks == NULL ? 0 : blocks[range.index[j]]);
                        tmp_means[b] += range.value[j];
                        ++tmp_nzero[b];
                    }

                    for (size_t b = 0; b < nblocks; ++b) {
                        if (block_size[b]) {
                            tmp_means[b] /= block_size[b];
                        }
                    }

                    for (size_t j = 0; j < range.number; ++j) {
                        auto b = (blocks == NULL ? 0 : blocks[range.index[j]]);
                        tmp_vars[b] += (range.value[j] - tmp_means[b]) * (range.value[j] - tmp_means[b]);
                    }

                    for (size_t b = 0; b < nblocks; ++b) {
                        means[b][i] = tmp_means[b];
                        variances[b][i] = tmp_vars[b] + tmp_means[b] * tmp_means[b] * (block_size[b] - tmp_nzero[b]);
                    }
                }
            } else {
                for (size_t i = 0; i < NR; ++i) {
                    auto ptr = p->row(i, obuffer.data(), wrk.get());
                    std::fill(tmp_means.begin(), tmp_means.end(), 0);
                    std::fill(tmp_vars.begin(), tmp_vars.end(), 0);

                    for (size_t j = 0; j < NC; ++j) {
                        auto b = (blocks == NULL ? 0 : blocks[j]);
                        tmp_means[b] += ptr[j];
                    }

                    for (size_t b = 0; b < nblocks; ++b) {
                        if (block_size[b]) {
                            tmp_means[b] /= block_size[b];
                        }
                    }

                    for (size_t j = 0; j < NC; ++j) {
                        auto b = (blocks == NULL ? 0 : blocks[j]);
                        tmp_vars[b] += (ptr[j] - tmp_means[b]) * (ptr[j] - tmp_means[b]);
                    }

                    for (size_t b = 0; b < nblocks; ++b) {
                        means[b][i] = tmp_means[b];
                        variances[b][i] = tmp_vars[b];
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
            }

            if (p->sparse()) {
                std::vector<IDX> ibuffer(NR);
                auto& my_nzero = fitted; // re-using the memory here.

                for (size_t i = 0; i < NC; ++i) {
                    auto range = p->sparse_column(i, obuffer.data(), ibuffer.data(), wrk.get());

                    auto b = (blocks == NULL ? 0 : blocks[i]);
                    double* mean = means[b];
                    double* var = variances[b];
                    double* nzero = my_nzero[b];

                    for (size_t j = 0; j < range.number; ++j, ++range.value, ++range.index) {
                        double& curmean = mean[*range.index];
                        double& curvar = var[*range.index];
                        double& curnzero = nzero[*range.index];
                        ++curnzero;

                        const double delta = *range.value - curmean;
                        curmean += delta/curnzero;
                        curvar += delta*(*range.value - curmean);
                    }
                }

                for (size_t b = 0; b < nblocks; ++b) {
                    if (block_size[b]) {
                        for (size_t i = 0 ; i < NR; ++i) {
                            const double curNZ = my_nzero[b][i];
                            const double ratio = curNZ / block_size[b];
                            auto& curM = means[b][i];
                            variances[b][i] += curM * curM * ratio * (block_size[b] - curNZ);
                            curM *= ratio;
                        }
                    }
                }

            } else {
                std::vector<int> used_sofar(nblocks);

                for (size_t i = 0; i < NC; ++i) {
                    auto ptr = p->column(i, obuffer.data(), wrk.get());

                    auto b = (blocks == NULL ? 0 : blocks[i]);
                    double* mean = means[b];
                    double* var = variances[b];
                    int& used = used_sofar[b];
                    ++used;

                    for (size_t j = 0; j < NR; ++j, ++ptr, ++mean, ++var) {
                        const double delta = *ptr - *mean;
                        *mean += delta/used;
                        *var += delta*(*ptr - *mean);
                    }
                }
            }
        }

        for (size_t b = 0; b < nblocks; ++b) {
            if (block_size[b] < 2) {
                if (block_size[b] < 1) {
                    std::fill(means[b], means[b] + NR, std::numeric_limits<double>::quiet_NaN());
                }
                std::fill(variances[b], variances[b] + NR, std::numeric_limits<double>::quiet_NaN());
                std::fill(fitted[b], fitted[b] + NR, std::numeric_limits<double>::quiet_NaN());
                std::fill(residuals[b], residuals[b] + NR, std::numeric_limits<double>::quiet_NaN());
                continue;
            }

            for (size_t i = 0; i < NR; ++i) {
                variances[b][i] /= block_size[b] - 1;
            }

            // Applying the trend fit to each block.
            fit.run(NR, means[b], variances[b], fitted[b], residuals[b]);
        }

        return;
    }
    
private:
    const BLOCK* blocks = NULL;
    size_t group_ncells = 0;
    int nblocks = 1;
    FitTrendVar fit;
};

}

#endif
