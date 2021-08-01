#ifndef SCRAN_BLOCKED_VARIANCES_HPP
#define SCRAN_BLOCKED_VARIANCES_HPP

#include <vector>
#include <algorithm>
#include <limits>

#include "tatami/stats/variances.hpp"
#include "tatami/stats/apply.hpp"

namespace scran {

namespace feature_selection {

template<bool blocked, class B>
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
        auto b = get_block<blocked>(range.index[j], block);
        tmp_means[b] += range.value[j];
        ++tmp_nzero[b];
    }
    finish_means(block_size, tmp_means);

    for (size_t j = 0; j < range.number; ++j) {
        auto b = get_block<blocked>(range.index[j], block);
        tmp_vars[b] += (range.value[j] - tmp_means[b]) * (range.value[j] - tmp_means[b]);
    }
    for (size_t b = 0; b < block_size.size(); ++b) {
        tmp_vars[b] += tmp_means[b] * tmp_means[b] * (block_size[b] - tmp_nzero[b]);
    }
    finish_variances(block_size, tmp_vars);
}

template<class V, typename B, class Bs>
struct Common {
    Common(V& m, V& v, const B* b, const Bs& bs) : means(m), variances(v), block(b), block_size(bs) {}
    V& means;
    V& variances;
    const B* block;
    const Bs& block_size;
};

template<bool blocked, class V, typename B, class Bs>
struct BlockedVarianceFactory : public Common<V, B, Bs> {
public:
    BlockedVarianceFactory(size_t nr, size_t nc, V& m, V& v, const B* b, const Bs& bs) : NR(nr), NC(nc), Common<V, B, Bs>(m, v, b, bs) {}

    static constexpr bool supports_running = true;
    static constexpr bool supports_sparse = true;
private:
    size_t NR, NC;

public:
    struct DenseByRow : public Common<V, B, Bs> {
        DenseByRow(size_t nc, V& m, V& v, const B* b, const Bs& bs) : NC(nc), Common<V, B, Bs>(m, v, b, bs), tmp_means(bs.size()), tmp_vars(bs.size()) {}

        template<typename T>
        void compute(size_t i, const T* ptr, T* buffer) {
            blocked_variance_with_mean<blocked>(ptr, NC, this->block, this->block_size, tmp_means, tmp_vars);
            for (size_t b = 0; b < tmp_means.size(); ++b) {
                this->means[b][i] = tmp_means[b];
                this->variances[b][i] = tmp_vars[b];
            }
        }
    private:
        size_t NC;
        std::vector<double> tmp_means, tmp_vars;
    };

    DenseByRow dense_direct() {
        return DenseByRow(NC, this->means, this->variances, this->block, this->block_size);
    }

public:
    struct SparseByRow : public Common<V, B, Bs> {
        SparseByRow(V& m, V& v, const B* b, const Bs& bs) : Common<V, B, Bs>(m, v, b, bs), tmp_means(bs.size()), tmp_vars(bs.size()), tmp_nzero(bs.size()) {}

        template<class SparseRange, typename T, typename IDX>
        void compute(size_t i, SparseRange&& range, T* xbuffer, IDX* ibuffer) {
            blocked_variance_with_mean<blocked>(range, this->block, this->block_size, tmp_means, tmp_vars, tmp_nzero);
            for (size_t b = 0; b < tmp_means.size(); ++b) {
                this->means[b][i] = tmp_means[b];
                this->variances[b][i] = tmp_vars[b];
            }
        }
    private:
        std::vector<double> tmp_means, tmp_vars;
        std::vector<int> tmp_nzero;
    };

    SparseByRow sparse_direct() {
        return SparseByRow(this->means, this->variances, this->block, this->block_size);
    }

public:
    struct DenseByCol : public Common<V, B, Bs> {
        DenseByCol(size_t nr, V& m, V& v, const B* b, const Bs& bs) : NR(nr), counts(bs.size()), Common<V, B, Bs>(m, v, b, bs) {}

        template<typename T>
        void add(const T* ptr, T* buffer) {
            auto b = get_block<blocked>(counter, this->block);
            tatami::stats::variances::compute_running(ptr, NR, this->means[b], this->variances[b], counts[b]);
            ++counter;
        }

        void finish() {
            for (size_t b = 0; b < this->means.size(); ++b) {
                tatami::stats::variances::finish_running(NR, this->means[b], this->variances[b], counts[b]);
            }
        }
    private:
        size_t NR;
        std::vector<int> counts;
        int counter = 0;
    };

    DenseByCol dense_running() {
        return DenseByCol(NR, this->means, this->variances, this->block, this->block_size);
    }

public:
    struct SparseByCol : public Common<V, B, Bs> {
        SparseByCol(size_t nr, V& m, V& v, const B* b, const Bs& bs) : NR(nr), nonzeros(bs.size(), std::vector<int>(nr)), counts(bs.size()), Common<V, B, Bs>(m, v, b, bs) {}

        template<class SparseRange, typename T, typename IDX>
        void add(SparseRange&& range, T* xbuffer, IDX* ibuffer) {
            auto b = get_block<blocked>(counter, this->block);
            tatami::stats::variances::compute_running(range, this->means[b], this->variances[b], nonzeros[b].data(), counts[b]);
            ++counter;
        }

        void finish() {
            for (size_t b = 0; b < this->means.size(); ++b) {
                tatami::stats::variances::finish_running(NR, this->means[b], this->variances[b], nonzeros[b].data(), counts[b]);
            }
        }

    private:
        size_t NR;
        std::vector<std::vector<int> > nonzeros;
        std::vector<int> counts;
        int counter = 0;
    };

    SparseByCol sparse_running() {
        return SparseByCol(NR, this->means, this->variances, this->block, this->block_size);
    }
};

}

}

#endif
