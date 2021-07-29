#ifndef SCRAN_BLOCK_SUMMARIES_HPP
#define SCRAN_BLOCK_SUMMARIES_HPP

#include <vector>
#include <algorithm>
#include <limits>

#include "tatami/stats/variances.hpp"

namespace scran {

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

        auto sum2mean = [&]() -> void {
            for (size_t b = 0; b < nblocks; ++b) {
                if (block_size[b]) {
                    tmp_means[b] /= block_size[b];
                } else {
                    tmp_means[b] = std::numeric_limits<double>::quiet_NaN();
                }
            }
        };

        if (p->sparse()) {
            std::vector<int> tmp_nzero(nblocks);
            std::vector<typename MAT::index> ibuffer(NC);

            for (size_t i = 0; i < NR; ++i) {
                auto range = p->sparse_row(i, obuffer.data(), ibuffer.data(), wrk.get());
                std::fill(tmp_means.begin(), tmp_means.end(), 0);
                std::fill(tmp_vars.begin(), tmp_vars.end(), 0);
                std::fill(tmp_nzero.begin(), tmp_nzero.end(), 0);

                auto get_block = [&](size_t j) -> int {
                    if constexpr(blocked) {
                        return block[range.index[j]];
                    } else {
                        return 0;
                    }
                };

                for (size_t j = 0; j < range.number; ++j) {
                    int b = get_block(j);
                    tmp_means[b] += range.value[j];
                    ++tmp_nzero[b];
                }

                sum2mean();

                for (size_t j = 0; j < range.number; ++j) {
                    auto b = get_block(j);
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

                auto get_block = [&](size_t j) -> int {
                    if constexpr(blocked) {
                        return block[j];
                    } else {
                        return 0;
                    }
                };

                for (size_t j = 0; j < NC; ++j) {
                    auto b = get_block(j);
                    tmp_means[b] += ptr[j];
                }

                sum2mean();

                for (size_t j = 0; j < NC; ++j) {
                    auto b = get_block(j);
                    tmp_vars[b] += (ptr[j] - tmp_means[b]) * (ptr[j] - tmp_means[b]);
                }

                for (size_t b = 0; b < nblocks; ++b) {
                    means[b][i] = tmp_means[b];
                    variances[b][i] = tmp_vars[b];
                }
            }
        }

        // Dividing by the relevant denominators.
        for (size_t b = 0; b < nblocks; ++b) {
            if (block_size[b] < 2) {
                std::fill(variances[b], variances[b] + NR, std::numeric_limits<double>::quiet_NaN());
            } else {
                double denominator = block_size[b] - 1;
                for (size_t i = 0; i < NR; ++i) {
                    variances[b][i] /= denominator;
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

        auto get_block = [&](size_t i) -> int {
            if constexpr(blocked) {
                return block[i];
            } else {
                return 0;
            }
        };

        if (p->sparse()) {
            std::vector<typename MAT::index> ibuffer(NR);
            std::vector<tatami::stats::VarianceHelper::Sparse> running(nblocks, tatami::stats::VarianceHelper::Sparse(NR));

            for (size_t i = 0; i < NC; ++i) {
                auto range = p->sparse_column(i, obuffer.data(), ibuffer.data(), wrk.get());
                auto b = get_block(i);
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
                auto b = get_block(i);
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

#endif
