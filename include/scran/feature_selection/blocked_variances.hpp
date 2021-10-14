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

template<typename S, typename B, class Bs>
struct Common {
    Common(std::vector<S*> m, std::vector<S*> v, const B* b, const Bs* bs) : means(std::move(m)), variances(std::move(v)), block(b), block_size_ptr(bs) {}
    std::vector<S*> means;
    std::vector<S*> variances;
    const B* block;
    const Bs* block_size_ptr;
};

template<bool blocked, typename S, typename B, class Bs>
struct BlockedVarianceFactory : public Common<S, B, Bs> {
public:
    BlockedVarianceFactory(size_t nr, size_t nc, std::vector<S*> m, std::vector<S*> v, const B* b, const Bs* bs) : NR(nr), NC(nc), Common<S, B, Bs>(std::move(m), std::move(v), b, bs) {}

private:
    size_t NR, NC;

private:
    struct ByRow : Common<S, B, Bs> {
        ByRow(std::vector<S*> m, std::vector<S*> v, const B* b, const Bs* bs) : Common<S, B, Bs>(std::move(m), std::move(v), b, bs), tmp_means(bs->size()), tmp_vars(bs->size()) {}
    protected:
        std::vector<double> tmp_means, tmp_vars;
    };

public:
    struct DenseByRow : public ByRow {
        DenseByRow(size_t nc, std::vector<S*> m, std::vector<S*> v, const B* b, const Bs* bs) : NC(nc), ByRow(std::move(m), std::move(v), b, bs) {}

        template<typename T>
        void compute(size_t i, const T* ptr) {
            blocked_variance_with_mean<blocked>(ptr, NC, this->block, *(this->block_size_ptr), this->tmp_means, this->tmp_vars);
            for (size_t b = 0; b < this->tmp_means.size(); ++b) {
                this->means[b][i] = this->tmp_means[b];
                this->variances[b][i] = this->tmp_vars[b];
            }
        }
    private:
        size_t NC;
    };

    DenseByRow dense_direct() {
        return DenseByRow(NC, this->means, this->variances, this->block, this->block_size_ptr);
    }

public:
    struct SparseByRow : public ByRow {
        SparseByRow(std::vector<S*> m, std::vector<S*> v, const B* b, const Bs* bs) : ByRow(std::move(m), std::move(v), b, bs), tmp_nzero(bs->size()) {}

        template<class SparseRange, typename T, typename IDX>
        void compute(size_t i, SparseRange&& range) {
            blocked_variance_with_mean<blocked>(range, this->block, *(this->block_size_ptr), this->tmp_means, this->tmp_vars, tmp_nzero);
            for (size_t b = 0; b < this->tmp_means.size(); ++b) {
                this->means[b][i] = this->tmp_means[b];
                this->variances[b][i] = this->tmp_vars[b];
            }
        }
    private:
        std::vector<int> tmp_nzero;
    };

    SparseByRow sparse_direct() {
        return SparseByRow(this->means, this->variances, this->block, this->block_size_ptr);
    }

private:
    struct ByCol : public Common<S, B, Bs> {
        ByCol(size_t nr, std::vector<S*> m, std::vector<S*> v, const B* b, const Bs* bs) : NR(nr), counts(bs->size()), Common<S, B, Bs>(std::move(m), std::move(v), b, bs) {
            for (auto& mptr : this->means) {
                std::fill(mptr, mptr + nr, 0);
            }
            for (auto& vptr : this->variances) {
                std::fill(vptr, vptr + nr, 0);
            }
        }
    protected:
        size_t NR;
        std::vector<int> counts;
        size_t counter = 0;
    };

public:
    struct DenseByCol : public ByCol {
        DenseByCol(size_t nr, std::vector<S*> m, std::vector<S*> v, const B* b, const Bs* bs) : ByCol(nr, std::move(m), std::move(v), b, bs) {}

        template<typename T>
        void add(const T* ptr) {
            auto b = get_block<blocked>(this->counter, this->block);
            tatami::stats::variances::compute_running(ptr, this->NR, this->means[b], this->variances[b], this->counts[b]);
            ++(this->counter);
        }

        void finish() {
            for (size_t b = 0; b < this->means.size(); ++b) {
                tatami::stats::variances::finish_running(this->NR, this->means[b], this->variances[b], this->counts[b]);
            }
        }
    };

    DenseByCol dense_running() {
        return DenseByCol(this->NR, this->means, this->variances, this->block, this->block_size_ptr);
    }

    DenseByCol dense_running(size_t start, size_t end) {
        auto mymean = this->means;
        for (auto& m : mymean) {
            m += start;
        }

        auto myvar = this->variances;
        for (auto& m : myvar) {
            m += start;
        }

        return DenseByCol(end - start, std::move(mymean), std::move(myvar), this->block, this->block_size_ptr);
    }

public:
    struct SparseByCol : public ByCol {
        SparseByCol(size_t nr, size_t s, size_t e, std::vector<S*> m, std::vector<S*> v, const B* b, const Bs* bs) : 
            start(s), end(e), nonzeros(bs->size(), std::vector<int>(nr)), ByCol(nr, std::move(m), std::move(v), b, bs) 
        {
            counter = start;
            return;
        }

        template<class SparseRange, typename T, typename IDX>
        void add(SparseRange&& range) {
            auto b = get_block<blocked>(this->counter, this->block);
            tatami::stats::variances::compute_running(range, this->means[b], this->variances[b], nonzeros[b].data(), this->counts[b]);
            ++(this->counter);
        }

        void finish() {
            for (size_t b = 0; b < this->means.size(); ++b) {
                tatami::stats::variances::finish_running(end - start, this->means[b] + start, this->variances[b] + start, nonzeros[b].data() + start, this->counts[b]);
            }
        }

    private:
        std::vector<std::vector<int> > nonzeros;
        size_t start, end;
    };

    SparseByCol sparse_running() {
        return SparseByCol(NR, 0, NR, this->means, this->variances, this->block, this->block_size_ptr);
    }

    SparseByCol sparse_running(size_t start, size_t end) {
        return SparseByCol(NR, start, end, this->means, this->variances, this->block, this->block_size_ptr);
    }
};

}

}

#endif
