#ifndef SCRAN_BLOCK_SUMMARIES_HPP
#define SCRAN_BLOCK_SUMMARIES_HPP

#include <vector>
#include <algorithm>
#include <limits>

#include "tatami/stats/variances.hpp"

namespace scran {

namespace feature_selection {

template<bool blocked, typename B>
B get_block(size_t j, const B* block) {
    if constexpr(blocked) {
        return block[j];
    } else {
        return 0;
    }
}

template<class Bs, class Tmp>
void finish_means(Bs& block_size, Tmp& tmp_means) {
    for (size_t b = 0; b < tmp_means.size(); ++b) {
        if (block_size[b]) {
            tmp_means[b] /= block_size[b];
        } else {
            tmp_means[b] = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

template<class Bs, class Tmp>
void finish_variances(Bs& block_size, Tmp& tmp_vars) {
    for (size_t b = 0; b < tmp_vars.size(); ++b) {
        if (block_size[b] > 1) {
            tmp_vars[b] /= block_size[b] - 1;
        } else {
            tmp_vars[b] = std::numeric_limits<double>::quiet_NaN();
        }
    }
}

template<bool blocked, typename T, typename B, class Bs, class Tmp>
void blocked_variance_with_mean(const T* ptr, size_t NC, const B* block, Bs& block_size, Tmp& tmp_means, Tmp& tmp_vars) {
    std::fill(tmp_means.begin(), tmp_means.end(), 0);
    std::fill(tmp_vars.begin(), tmp_vars.end(), 0);

    for (size_t j = 0; j < NC; ++j) {
        auto b = get_block<blocked>(j, block);
        tmp_means[b] += ptr[j];
    }
    finish_means(block_size, tmp_means);

    for (size_t j = 0; j < NC; ++j) {
        auto b = get_block<blocked>(j, block);
        tmp_vars[b] += (ptr[j] - tmp_means[b]) * (ptr[j] - tmp_means[b]);
    }
    finish_variances(block_size, tmp_vars);
}

template<bool blocked, class SparseRange, typename B, class Bs, class Tmpd, class Tmpi> 
void blocked_variance_with_mean(SparseRange&& range, const B* block, Bs& block_size, Tmpd& tmp_means, Tmpd& tmp_vars, Tmpi& tmp_nzero) {
    std::fill(tmp_means.begin(), tmp_means.end(), 0);
    std::fill(tmp_vars.begin(), tmp_vars.end(), 0);
    std::fill(tmp_nzero.begin(), tmp_nzero.end(), 0);

    for (size_t j = 0; j < range.number; ++j) {
        auto b = get_block<blocked>(j, block);
        tmp_means[b] += range.value[j];
        ++tmp_nzero[b];
    }
    finish_means(block_size, tmp_means);

    for (size_t j = 0; j < range.number; ++j) {
        auto b = get_block<blocked>(j, block);
        tmp_vars[b] += (range.value[j] - tmp_means[b]) * (range.value[j] - tmp_means[b]);
    }
    for (size_t b = 0; b < block_size.size(); ++b) {
        tmp_vars[b] += tmp_means[b] * tmp_means[b] * (block_size[b] - tmp_nzero[b]);
    }
    finish_variances(block_size, tmp_vars);
}

template<bool blocked, class MAT, typename B, class V>
std::vector<int> block_summaries(const MAT* p, const B* block, V&& means, V&& variances) {
    // Estimating the raw values.
    size_t NR = p->nrow(), NC = p->ncol();
    size_t nblocks = means.size();
    std::vector<int> block_size(nblocks);

    if constexpr(blocked) {
        auto copy = block;
        for (size_t j = 0; j < NC; ++j, ++copy) {
            ++block_size[*copy];
        }
    } else {
        block_size[0] = NC;
    }

    if (p->prefer_rows()) {
        std::vector<typename MAT::value> obuffer(NC);
        auto wrk = p->new_workspace(true);

        // We use temporary buffers to improve memory locality for frequent
        // write operations, before transferring the result to the actual stores.
        std::vector<double> tmp_means(nblocks), tmp_vars(nblocks);

        if (p->sparse()) {
            std::vector<int> tmp_nzero(nblocks);
            std::vector<typename MAT::index> ibuffer(NC);
            for (size_t i = 0; i < NR; ++i) {
                auto range = p->sparse_row(i, obuffer.data(), ibuffer.data(), wrk.get());
                blocked_variance_with_mean<blocked>(range, block, block_size, tmp_means, tmp_vars, tmp_nzero);

                for (size_t b = 0; b < nblocks; ++b) {
                    means[b][i] = tmp_means[b];
                    variances[b][i] = tmp_vars[b];
                }
            }
        } else {
            for (size_t i = 0; i < NR; ++i) {
                auto ptr = p->row(i, obuffer.data(), wrk.get());
                blocked_variance_with_mean<blocked>(ptr, NC, block, block_size, tmp_means, tmp_vars);

                for (size_t b = 0; b < nblocks; ++b) {
                    means[b][i] = tmp_means[b];
                    variances[b][i] = tmp_vars[b];
                }
            }
        }

    } else {
        std::vector<typename MAT::value> obuffer(NR);
        auto wrk = p->new_workspace(false);

        for (size_t b = 0; b < nblocks; ++b) {
            std::fill(means[b], means[b] + NR, 0);
            std::fill(variances[b], variances[b] + NR, 0);
        }

        if (p->sparse()) {
            std::vector<typename MAT::index> ibuffer(NR);
            std::vector<tatami::stats::VarianceHelper::Sparse> running(nblocks, tatami::stats::VarianceHelper::Sparse(NR));

            for (size_t i = 0; i < NC; ++i) {
                auto range = p->sparse_column(i, obuffer.data(), ibuffer.data(), wrk.get());
                auto b = get_block<blocked>(i, block);
                running[b].add(range);
            }

            for (size_t b = 0; b < nblocks; ++b) {
                running[b].finish();
                const auto& running_vars = running[b].statistics();
                std::copy(running_vars.begin(), running_vars.end(), variances[b]);
                const auto& running_means = running[b].means();
                std::copy(running_means.begin(), running_means.end(), means[b]);
            }

        } else {
            std::vector<tatami::stats::VarianceHelper::Dense> running(nblocks, tatami::stats::VarianceHelper::Dense(NR));

            for (size_t i = 0; i < NC; ++i) {
                auto ptr = p->column(i, obuffer.data(), wrk.get());
                auto b = get_block<blocked>(i, block);
                running[b].add(ptr);
            }

            for (size_t b = 0; b < nblocks; ++b) {
                running[b].finish();
                const auto& running_vars = running[b].statistics();
                std::copy(running_vars.begin(), running_vars.end(), variances[b]);
                const auto& running_means = running[b].means();
                std::copy(running_means.begin(), running_means.end(), means[b]);
            }
        }
    }

    return block_size;
}

}

}

#endif
