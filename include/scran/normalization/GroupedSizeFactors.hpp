#ifndef SCRAN_CLUSTERED_SIZE_FACTORS_HPP
#define SCRAN_CLUSTERED_SIZE_FACTORS_HPP

#include "MedianSizeFactors.hpp"
#include "LogNormCounts.hpp"
#include "../aggregation/AggregateAcrossCells.hpp"
#include "../dimensionality_reduction/RunPCA.hpp"

#include "tatami/base/DenseMatrix.hpp"
#include "tatami/ext/ArrayView.hpp"
#include "tatami/stats/sums.hpp"

#include <memory>
#include <vector>
#include <algorithm>

namespace scran {

class GroupedSizeFactors {
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

        tatami::DenseColumnMatrix<T, IDX> aggmat(NR, ngroups, tatami::ArrayView(combined.data(), combined.size()));
        auto aggcolsums = tatami::column_sums(&aggmat);

        // Choosing one of them to be the reference. Here, we pick the group
        // closest to the middle of the first PC, on the understanding that
        // this is the most "intermediate" and has the fewest DE compared to
        // other groups. This reduces the estimation error on the factors. 
        auto copy = combined;
        size_t closest = 0;

        {
            LogNormCounts logger;
            auto logged = logger.run(std::shared_ptr<const tatami::Matrix<T, IDX> >(&mat, [](const tatami::Matrix<T,IDX>*){}), aggcolsums);

            RunPCA pca;
            auto pres = pca.set_rank(1).run(logged);

            auto middle = std::accumulate(pres.pcs.begin(), pres.pcs.end(), 0) / NC;
            double closest_diff;

            for (size_t i = 0; i < NC; ++i) {
                double absdiff = std::abs(pres.pcs[i] - middle);
                if (i == 0 || absdiff < closest_diff) {
                    closest_diff = absdiff;
                    closest = i;
                }
            }
        }

        // Computing median-based size factors.
        MedianSizeFactors med;
        auto mres = med.run(&aggmat, combined.data() + closest * NR);

        // Propagating to each cell via library size-based normalization.
        auto colsums = tatami::column_sums(mat);
        for (size_t i = 0; i < NC; ++i) {
            auto curgroup = group[i];
            auto scale = static_cast<double>(colsums[i])/static_cast<double>(aggcolsums[curgroup]);
            output[i] = scale * mres.factors[curgroup];
        }
    }

public:
    template<typename Output>
    struct Result {
        Result(size_t NC) : factors(NC) {}
        std::vector<Output> factors;
    };

    template<typename Out = double, typename T, typename IDX, typename Group>
    void run(const tatami::Matrix<T, IDX>* mat, const Group* group) {
        Result<Out> output(mat->ncol());
        run(mat, group, output.factors.data());
        return output;
    }
};

}

#endif
