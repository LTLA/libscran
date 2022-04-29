#ifndef SCRAN_CLUSTERED_SIZE_FACTORS_HPP
#define SCRAN_CLUSTERED_SIZE_FACTORS_HPP

#include "MedianSizeFactors.hpp"
#include "../aggregation/AggregateAcrossCells.hpp"

#include "tatami/base/DenseMatrix.hpp"
#include "tatami/ext/ArrayView.hpp"
#include "tatami/stats/sums.hpp"

#include <memory>
#include <vector>
#include <algorithm>

namespace scran {

class GroupedSizeFactors {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults {
        /** 
         * See `set_center()` for more details.
         */
        static constexpr bool center = true;
    };

    /**
     * @param c Whether to center the size factors to have a mean of unity.
     * This is usually desirable for interpretation of relative values.
     * 
     * @return A reference to this `MedianSizeFactors` object.
     *
     * For more control over centering, this can be set to `false` and the resulting size factors can be passed to `CenterSizeFactors`.
     */
    GroupedSizeFactors& set_center(bool c = Defaults::center) {
        center = c;
        return *this;
    }

private:
    bool center = Defaults::center;

public:
    template<typename T, typename IDX, typename Group, typename Out>
    void run(const tatami::Matrix<T, IDX>* mat, const Group* group, Out* output) {
        size_t NR = mat->nrow(), NC = mat->ncol();
        if (!NC) {
            return;
        }

        // Aggregating each group to get a pseudo-bulk sample.
        auto ngroups = *std::max_element(group, group + NC) + 1;
        std::vector<double> combined(ngroups * NR);
        {
            std::vector<double*> sums(ngroups);
            for (size_t i = 0; i < ngroups; ++i) {
                sums[i] = combined.data() + i * NR;
            }
            AggregateAcrossCells aggregator;
            aggregator.run(mat, group, std::move(sums), std::vector<int*>());
        }

        // Choosing one of them to be the reference. Here, we borrow some logic
        // from edgeR and use the one with the largest sum of square roots,
        // which provides a compromise between transcriptome coverage and
        // complexity. The root ensures that we don't pick a sample that just
        // has very high expression in a small subset of genes, while still
        // remaining responsive to the overall coverage level.
        size_t ref = 0;
        double best = 0;

        for (size_t i = 0; i < ngroups; ++i) {
            auto start = combined.data() + i * NR;
            double current = 0;
            for (size_t j = 0; j < NR; ++j) {
                current += std::sqrt(start[j]);
            }
            if (current > best) {
                ref = i;
            }
        }

        // Computing median-based size factors.
        tatami::ArrayView view(combined.data(), combined.size());
        tatami::DenseColumnMatrix<T, IDX, decltype(view)> aggmat(NR, ngroups, std::move(view));
        MedianSizeFactors med;
        auto mres = med.run(&aggmat, combined.data() + ref * NR);

        // Propagating to each cell via library size-based normalization.
        auto aggcolsums = tatami::column_sums(&aggmat);
        auto colsums = tatami::column_sums(mat);
        for (size_t i = 0; i < NC; ++i) {
            auto curgroup = group[i];
            auto scale = static_cast<double>(colsums[i])/static_cast<double>(aggcolsums[curgroup]);
            output[i] = scale * mres.factors[curgroup];
        }

        // Throwing in some centering.
        if (center) {
            CenterSizeFactors centerer;
            centerer.run(NC, output);
        }
    }

public:
    template<typename Out>
    struct Result {
        Result(size_t NC) : factors(NC) {}
        std::vector<Out> factors;
    };

    template<typename Out = double, typename T, typename IDX, typename Group>
    Result<Out> run(const tatami::Matrix<T, IDX>* mat, const Group* group) {
        Result<Out> output(mat->ncol());
        run(mat, group, output.factors.data());
        return output;
    }
};

}

#endif
