#ifndef SCRAN_AGGREGATION_FACTORY_HPP
#define SCRAN_AGGREGATION_FACTORY_HPP

#include <vector>
#include <type_traits>
#include <algorithm>

namespace scran {

namespace aggregate_across_cells {

// This basically uses the same code as differential_analysis::BidimensionalFactory.
// Unfortunately it is difficult to use _exactly_ the same code because there are 
// just too many differences in behavior; we'll just copy the parts that we need. 
template<typename Factor, typename Sum, typename Detected> 
struct BidimensionalFactory {
public:
    BidimensionalFactory(size_t nr, size_t nc, const Factor* f, std::vector<Sum*> s, std::vector<Detected*> d) :
        NR(nr), 
        NC(nc), 
        factor(f),
        sums(std::move(s)), 
        detected(std::move(d))
    {}

protected:
    size_t NR, NC;
    const Factor* factor;
    std::vector<Sum*> sums;
    std::vector<Detected*> detected;

public:
    struct ByRow {
        ByRow(size_t nc, const Factor* f, std::vector<Sum*> s, std::vector<Detected*> d) : 
            NC(nc),
            factor(f),
            sums(std::move(s)),
            detected(std::move(d)),
            tmp_sums(sums.size()),
            tmp_detected(detected.size())
        {}

        template<typename Row>
        void compute(size_t i, const Row& row) {
            constexpr bool is_sparse = !std::is_pointer<Row>::value;

            if (this->sums.size()) {
                std::fill(tmp_sums.begin(), tmp_sums.end(), 0);

                if constexpr(is_sparse) {
                    for (size_t i = 0; i < row.number; ++i) {
                        tmp_sums[factor[row.index[i]]] += row.value[i];
                    }
                } else {
                    for (size_t j = 0; j < NC; ++j) {
                        tmp_sums[factor[j]] += row[j];
                    }
                }

                // Computing before transferring for more cache-friendliness.
                for (size_t l = 0; l < tmp_sums.size(); ++l) {
                    sums[l][i] = tmp_sums[l];
                }
            }

            if (this->detected.size()) {
                std::fill(tmp_detected.begin(), tmp_detected.end(), 0);

                if constexpr(is_sparse) {
                    for (size_t i = 0; i < row.number; ++i) {
                        tmp_detected[factor[row.index[i]]] += (row.value[i] > 0);
                    }
                } else {
                    for (size_t j = 0; j < NC; ++j) {
                        tmp_detected[factor[j]] += (row[j] > 0);
                    }
                }

                for (size_t l = 0; l < tmp_detected.size(); ++l) {
                    detected[l][i] = tmp_detected[l];
                }
            }
        }

    public:
        size_t NC;
        const Factor* factor;
        std::vector<Sum*> sums;
        std::vector<Detected*> detected; 
        std::vector<Sum> tmp_sums;
        std::vector<Detected> tmp_detected;
    };

    ByRow dense_direct() {
        return ByRow(NC, factor, sums, detected);
    }

    ByRow sparse_direct() {
        return ByRow(NC, factor, sums, detected);
    }

public:
    struct ByCol {
        ByCol(size_t start_, size_t end_, const Factor* f, std::vector<Sum*> s, std::vector<Detected*> d) :
            start(start_),
            end(end_),
            factor(f),
            sums(std::move(s)),
            detected(std::move(d))
        {}

        template<typename Column>
        void add(const Column& col) {
            constexpr bool is_sparse = !std::is_pointer<Column>::value;
            auto current = factor[counter];

            if (sums.size()) {
                auto& cursum = sums[current];
                if constexpr(is_sparse) {
                    for (size_t i = 0; i < col.number; ++i) {
                        cursum[col.index[i]] += col.value[i];
                    }
                } else {
                    for (size_t i = start; i < end; ++i) {
                        cursum[i] += col[i - start];
                    }
                }
            }

            if (detected.size()) {
                auto& curdetected = detected[current];
                if constexpr(is_sparse) {
                    for (size_t i = 0; i < col.number; ++i) {
                        curdetected[col.index[i]] += (col.value[i] > 0);
                    }
                } else {
                    for (size_t i = start; i < end; ++i) {
                        curdetected[i] += (col[i - start] > 0);
                    }
                }
            }

            ++counter;
            return;
        }

        void finish() {}

    private:
        size_t start, end;
        const Factor* factor;
        std::vector<Sum*> sums;
        std::vector<Detected*> detected; 
        size_t counter = 0;
    };

    ByCol dense_running() {
        return ByCol(0, this->NR, this->factor, this->sums, this->detected);
    }

    ByCol dense_running(size_t start, size_t end) {
        return ByCol(start, end, this->factor, this->sums, this->detected);
    }

    ByCol sparse_running() {
        return ByCol(0, this->NR, this->factor, this->sums, this->detected);
    }

    ByCol sparse_running(size_t start, size_t end) {
        return ByCol(start, end, this->factor, this->sums, this->detected);
    }
};

}

}

#endif
