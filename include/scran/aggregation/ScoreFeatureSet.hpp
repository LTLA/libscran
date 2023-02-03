#ifndef SCRAN_SCORE_FEATURE_SET_HPP
#define SCRAN_SCORE_FEATURE_SET_HPP

#include "../utils/macros.hpp"

#include <algorithm>
#include <vector>
#include "tatami/tatami.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Dense"

#include "../dimensionality_reduction/pca_utils.hpp"
#include "../dimensionality_reduction/CustomSparseMatrix.hpp"

namespace scran {

/**
 * @brief Compute per-cell scores for a given feature set.
 *
 * Per-cell scores are defined as the column sums of the rank-1 approximation of the input matrix for the subset of features in the set.
 * The central idea here is that the primary activity of the feature set can be quantified by the largest component of variance amongst its features.
 * (If this was not the case, one could argue that this feature set is not well-suited to capture the biology attributed to it.)
 * In effect, the rotation vector define weights for all features in the set, focusing on genes that contribute to the primary activity.
 * This is based on the [**GSDecon**](https://github.com/JasonHackney/GSDecon) package from Jason Hackney. 
 *
 * For multi-block analyses, we extend this approach by computing the rotation vector separately within each block.
 * We then aggregate the rotation vector across blocks - 
 * either by computing the weighted average where the weights are defined as the proportion of variance explained by the first PC in each block;
 * or by taking the rotation vector with the largest proportion of variance explained.
 * Each cell is projected onto this rotation vector, and the feature set score for each cell is then defined from the ensuing rank-1 approximation.
 * This approach avoids comparing cells between blocks and favors blocks where the feature set has greater (relative) activity.
 */
class ScoreFeatureSet {
public:
    /**
     * Policy to use to combine rotation vectors across blocks.
     * 
     * - `AVERAGE` will average rotation vectors, using the proportion of variance explained by the first PC within each block as the weight.
     * - `MAXIMUM` will take the rotation vector from the block with the largest proportion of variance explained by the first PC.
     */
    enum class BlockPolicy { AVERAGE, MAXIMUM };

    /**
     * @brief Default parameters.
     */
    struct Defaults {
        /**
         * See `set_block_policy()` for more details.
         */
        static constexpr BlockPolicy block_policy = BlockPolicy::AVERAGE;

        /**
         * See `set_num_threads()` for more details.
         */
        static constexpr int num_threads = 1;

        /**
         * See `set_scale()` for more details.
         */
        static constexpr bool scale = false;
    };

private:
    BlockPolicy block_policy = Defaults::block_policy;
    int nthreads = Defaults::num_threads;
    bool scale = Defaults::scale;

public:
    ScoreFeatureSet& set_block_policy(BlockPolicy b = Defaults::block_policy) {
        block_policy = b;
        return *this;
    }

    ScoreFeatureSet& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

    ScoreFeatureSet& set_scale(bool s = Defaults::scale) {
        scale = s;
        return *this;
    }

protected:
    std::vector<double> combine_rotation_vectors(const std::vector<Eigen::MatrixXd>& rotation, const std::vector<double>& variance_explained, const std::vector<double>& total_variance) const {
        std::vector<double> output;
        size_t nblocks = rotation.size();
        auto compute_var_prop = [&](int b) -> double {
            if (total_variance[b] == 0) {
                return 0; // avoid UB upon an unfortunate div by zero.
            } else {
                return variance_explained[b] / total_variance[b];
            }
        };

        if (block_policy == BlockPolicy::MAXIMUM) {
            double best_var_prop = compute_var_prop(0);
            size_t best_index = 0;

            for (size_t b = 1; b < nblocks; ++b) {
                double var_prop = compute_var_prop(b);
                if (var_prop > best_var_prop) {
                    best_index = b;
                    best_var_prop = var_prop;
                }
            }

            const auto& chosen = rotation[best_index];
            output.insert(output.end(), chosen.data(), chosen.data() + chosen.size());

        } else {
            double total_var_prop = 0;
            size_t nfeatures = rotation[0].size();
            output.resize(nfeatures);

            for (size_t b = 0; b < nblocks; ++b) {
                const auto& current_rotation = rotation[b];
                double var_prop = compute_var_prop(b);
                total_var_prop += var_prop;

                // Deciding whether we need to flip the rotation vector or not,
                // as the vectors are not defined w.r.t. their sign.
                double proj = 1;
                if (b) {
                    if (std::inner_product(output.begin(), output.end(), current_rotation.data(), 0.0) < 0.0) {
                        proj = -1;
                    }
                }

                double mult = proj * var_prop;
                for (size_t f = 0; f < nfeatures; ++f) {
                    output[f] += mult * current_rotation(f, 0);
                }
            }

            if (total_var_prop) {
                double l2 = 0;
                for (auto& x : output) {
                    x /= total_var_prop;
                    l2 += x * x;
                }

                if (l2 > 0) { // convert to unit vector
                    l2 = std::sqrt(l2);
                    for (auto& x : output) {
                        x /= l2;
                    }
                }
            }
        }

        return output;
    }

    struct BlockwiseOutputs {
        std::vector<double> rotation;
        std::vector<std::vector<double> > block_scores;
    };

    template<class Function>
    BlockwiseOutputs compute_blockwise_scores(
        const std::vector<Eigen::MatrixXd>& rotation, 
        const std::vector<double>& variance_explained, 
        const std::vector<double>& total_variance, 
        const std::vector<size_t>& block_size,
        Function mult
    ) const {
        // This involves some mild copying of vectors... oh well.
        BlockwiseOutputs output;
        output.rotation = combine_rotation_vectors(rotation, variance_explained, total_variance);

        size_t nblocks = rotation.size();
        output.block_scores.reserve(nblocks);
        Eigen::VectorXd rotation_as_vector(output.rotation.size());
        std::copy(output.rotation.begin(), output.rotation.end(), rotation_as_vector.data()); 

        // We short-circuit the computation of the low-rank representation by
        // realizing that the column sum just involves taking the sum of the
        // rotation vector and multiplying it by the components.
        double pre_colsum = std::accumulate(output.rotation.begin(), output.rotation.end(), 0.0);

        for (size_t b = 0; b < nblocks; ++b) {
            Eigen::VectorXd scores(block_size[b]);
            mult(b, rotation_as_vector, scores);

            auto& out = output.block_scores[b];
            out.reserve(scores.size());
            for (auto x : scores) {
                out.push_back(x * pre_colsum);
            }
        }

        return output;
    }

private:
    // Re-using the same two-pass philosophy from RunPCA, to save memory.
    struct BlockwiseSparseComponents {
        BlockwiseSparseComponents(size_t nblocks) : ptrs(nblocks), values(nblocks), indices(nblocks) {}
        std::vector<std::vector<size_t> > ptrs;
        std::vector<std::vector<double> > values;
        std::vector<std::vector<int> > indices;
    };

    template<typename T, typename IDX, typename B>
    BlockwiseSparseComponents core_sparse_row(const tatami::Matrix<T, IDX>* mat, const B* block, size_t nblocks, const std::vector<size_t>& reverse_block_map) const {
        size_t NC = mat->ncol();
        size_t NR = mat->nrow();

        BlockwiseSparseComponents output(nblocks);
        auto& ptrs = output.ptrs;
        for (size_t b = 0; b < nblocks; ++b) {
            ptrs.resize(NR + 1);
        }

        /*** First round, to fetch the number of zeros in each row. ***/
        {
#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp parallel num_threads(nthreads)
            {
#else
            SCRAN_CUSTOM_PARALLEL(NR, [&](size_t start, size_t end) -> void {
#endif            

                std::vector<double> xbuffer(NC);
                std::vector<int> ibuffer(NC);
                auto wrk = mat->new_workspace(true);

#ifndef SCRAN_CUSTOM_PARALLEL
                #pragma omp for
                for (size_t r = 0; r < NR; ++r) {
#else
                for (size_t r = start; r < end; ++r) {
#endif

                    auto range = mat->sparse_row(r, xbuffer.data(), ibuffer.data(), wrk.get());
                    for (size_t i = 0; i < range.number; ++i) {
                        ++(ptrs[block[range.index[i]]][r + 1]);
                    }

#ifndef SCRAN_CUSTOM_PARALLEL
                }
            }
#else
                }
            }, nthreads);
#endif
        }

        auto& values = output.values;
        auto& indices = output.indices;

        /*** Second round, to populate the vectors. ***/
        {
            for (size_t b = 0; b < nblocks; ++b) {
                auto& curptrs = ptrs[b];
                for (size_t r = 0; r < NR; ++r) {
                    curptrs[r + 1] += curptrs[r];
                }
                values[b].resize(curptrs.back());
                indices[b].resize(curptrs.back());
            }
            auto ptr_copy = ptrs;

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp parallel num_threads(nthreads)
            {
#else
            SCRAN_CUSTOM_PARALLEL(NR, [&](size_t start, size_t end) -> void {
#endif            

                auto wrk = mat->new_workspace(true);
                std::vector<double> xbuffer(NC);
                std::vector<int> ibuffer(NC);

#ifndef SCRAN_CUSTOM_PARALLEL
                #pragma omp for
                for (size_t r = 0; r < NR; ++r) {
#else
                for (size_t r = start; r < end; ++r) {
#endif

                    auto range = mat->sparse_row(r, xbuffer.data(), ibuffer.data(), wrk.get());
                    for (size_t i = 0; i < range.number; ++i) {
                        auto c = range.index[i];
                        auto b = block[c];
                        auto& offset = ptr_copy[b][r];
                        values[b][offset] = range.value[i];
                        indices[b][offset] = reverse_block_map[c];
                        ++offset;
                    }

#ifndef SCRAN_CUSTOM_PARALLEL
                }
            }
#else
                }
            }, nthreads);
#endif
        }

        return output;
    }

    template<typename T, typename IDX, typename B>
    BlockwiseSparseComponents core_sparse_column(const tatami::Matrix<T, IDX>* mat, const B* block, size_t nblocks, const std::vector<size_t>& reverse_block_map) const {
        auto NR = mat->nrow();
        auto NC = mat->ncol();

        /*** First round, to fetch the number of zeros in each row. ***/
        std::vector<std::vector<size_t> > nonzeros_per_row;
        {
            size_t cols_per_thread = std::ceil(static_cast<double>(NC) / nthreads);
            std::vector<std::vector<std::vector<size_t> > > threaded_nonzeros_per_row(nthreads);

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp parallel for num_threads(nthreads)
            for (int t = 0; t < nthreads; ++t) {
#else
            SCRAN_CUSTOM_PARALLEL(nthreads, [&](int start, int end) -> void { // Trivial allocation of one job per thread.
            for (int t = start; t < end; ++t) {
#endif

                size_t startcol = cols_per_thread * t, endcol = std::min(startcol + cols_per_thread, NC);
                if (startcol < endcol) {
                    std::vector<std::vector<size_t> > nonzeros_per_row(nblocks);
                    for (size_t b = 0; b < nblocks; ++b) {
                        nonzeros_per_row[b].resize(NR);
                    }

                    std::vector<double> xbuffer(NR);
                    std::vector<int> ibuffer(NR);
                    auto wrk = mat->new_workspace(false);

                    for (size_t c = startcol; c < endcol; ++c) {
                        auto range = mat->sparse_column(c, xbuffer.data(), ibuffer.data(), wrk.get());
                        auto& current = nonzeros_per_row[block[c]];
                        for (size_t i = 0; i < range.number; ++i) {
                            ++(current[range.index[i]]);
                        }
                    }

                    threaded_nonzeros_per_row[t] = std::move(nonzeros_per_row);
                }

#ifndef SCRAN_CUSTOM_PARALLEL
            }
#else
            }
            }, nthreads);
#endif

            // There had better be at least one thread!
            nonzeros_per_row = std::move(threaded_nonzeros_per_row[0]);
            for (int t = 1; t < nthreads; ++t) {
                const auto& threaded = threaded_nonzeros_per_row[t];
                for (size_t b = 0; b < nblocks; ++b) {
                    auto it = nonzeros_per_row[b].begin();
                    for (auto x : threaded[b]) {
                        *it += x;
                        ++it;
                    }
                }
            }
        }

        /*** Second round, to populate the vectors. ***/
        BlockwiseSparseComponents output(nblocks);
        {
            auto& ptrs = output.ptrs;
            auto& values = output.values;
            auto& indices = output.indices;

            for (size_t b = 0; b < nblocks; ++b) {
                ptrs.resize(NR + 1);
                size_t total_nzeros = 0;
                const auto& nonzeros_per_block = nonzeros_per_row[b];
                auto& block_ptrs = ptrs[b];
                for (size_t r = 0; r < NR; ++r) {
                    total_nzeros += nonzeros_per_block[r];
                    block_ptrs[r + 1] = total_nzeros;
                }
                values[b].resize(total_nzeros);
                indices[b].resize(total_nzeros);
            }

            // Splitting by row this time, because columnar extraction can't be done safely.
            size_t rows_per_thread = std::ceil(static_cast<double>(NR) / nthreads);
            auto ptr_copy = ptrs;

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp parallel for num_threads(nthreads)
            for (int t = 0; t < nthreads; ++t) {
#else
            SCRAN_CUSTOM_PARALLEL(nthreads, [&](int start, int end) -> void { // Trivial allocation of one job per thread.
            for (int t = start; t < end; ++t) {
#endif

                size_t startrow = rows_per_thread * t, endrow = std::min(startrow + rows_per_thread, NR);
                if (startrow < endrow) {
                    auto wrk = mat->new_workspace(false);
                    std::vector<double> xbuffer(endrow - startrow);
                    std::vector<int> ibuffer(endrow - startrow);

                    for (size_t c = 0; c < NC; ++c) {
                        auto range = mat->sparse_column(c, xbuffer.data(), ibuffer.data(), startrow, endrow, wrk.get());
                        size_t b = block[c];
                        auto& block_values = values[b];
                        auto& block_indices = indices[b];
                        auto& block_ptr_copy = ptr_copy[b];
                        auto blocked_c = reverse_block_map[c];

                        for (size_t i = 0; i < range.number; ++i) {
                            auto r = range.index[i];
                            auto& offset = block_ptr_copy[r];
                            block_values[offset] = range.value[i];
                            block_indices[offset] = blocked_c;
                            ++offset;
                        }
                    }
                }

#ifndef SCRAN_CUSTOM_PARALLEL
            }
#else
            }
            }, nthreads);
#endif
        }

        return output;
    }

    BlockwiseOutputs sparse_core(size_t num_features, const std::vector<size_t>& block_size, BlockwiseSparseComponents components) const {
        size_t nblocks = block_size.size();
        std::vector<Eigen::MatrixXd> rotation(nblocks);
        std::vector<double> variance_explained(nblocks);
        std::vector<double> total_variance(nblocks);

        irlba::Irlba irb;
        irb.set_number(1);

        // Running through and computing the rotation vectors.
        std::vector<pca_utils::CustomSparseMatrix> all_matrices;
        all_matrices.reserve(nblocks);
        std::vector<Eigen::VectorXd> centers;
        centers.reserve(nblocks);
        std::vector<Eigen::VectorXd> scales;
        scales.reserve(nblocks);

        for (size_t b = 0; b < nblocks; ++b) {
            auto& values = components.values[b];
            auto& indices = components.indices[b];
            auto& ptrs = components.ptrs[b];

            centers.emplace_back(num_features);
            auto& center_v = centers.back();
            scales.emplace_back(num_features);
            auto& scale_v = scales.back();
            pca_utils::compute_mean_and_variance_from_sparse_components(num_features, block_size[b], values, indices, ptrs, center_v, scale_v, nthreads);

            double& total_var = total_variance[b];
            pca_utils::set_scale(scale, scale_v, total_var);

            all_matrices.emplace_back(block_size[b], num_features, nthreads); // transposed; we want genes in the columns.
            auto& A = all_matrices.back();
            A.fill_direct(std::move(values), std::move(indices), std::move(ptrs));

            auto& current_rotation = rotation[b];
            Eigen::MatrixXd pcs;
            Eigen::VectorXd d;

            irlba::EigenThreadScope t(nthreads);
            irlba::Centered<std::remove_reference<decltype(A)>::type> centered(&A, &center_v);
            if (scale) {
                irlba::Scaled<decltype(centered)> scaled(&centered, &scale_v);
                irb.run(scaled, pcs, current_rotation, d);
            } else {
                irb.run(centered, pcs, current_rotation, d);
            }

            variance_explained[b] = d[0] / static_cast<double>(A.rows() - 1);
        }

        return compute_blockwise_scores(
            rotation, 
            variance_explained, 
            total_variance, 
            block_size,
            [&](size_t b, Eigen::VectorXd& rotation_as_vector, Eigen::VectorXd& scores) -> void {
                irlba::EigenThreadScope t(nthreads);
                irlba::Centered<std::remove_reference<decltype(all_matrices[b])>::type> centered(&all_matrices[b], &(centers[b]));
                if (scale) {
                    irlba::Scaled<decltype(centered)> scaled(&centered, &(scales[b]));
                    scaled.multiply(rotation_as_vector, scores);
                } else {
                    centered.multiply(rotation_as_vector, scores);
                }
            }
        );
    }

private:
    template<typename T, typename IDX, typename B>
    std::vector<Eigen::MatrixXd> core_dense_row(
        const tatami::Matrix<T, IDX>* mat, 
        const B* block, 
        const std::vector<size_t>& block_size,
        const std::vector<size_t>& reverse_block_map 
    ) const {

        size_t nblocks = block_size.size();
        size_t NC = mat->ncol();
        size_t NR = mat->nrow();

        std::vector<Eigen::MatrixXd> outputs;
        outputs.reserve(nblocks);
        for (size_t b = 0; b < nblocks; ++b) {
            outputs.emplace_back(block_size[b], NR);
        }

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel num_threads(nthreads)
        {
            std::vector<double> work(NC);
            auto work = mat->new_workspace(true);

            #pragma omp for
            for (int r = 0; r < NR; ++r) {
#else
        SCRAN_CUSTOM_PARALLEL(NR, [&](int start, int end) -> void {
            std::vector<double> buffer(NC);
            auto wrk = mat->new_workspace(true);
            for (int r = start; r < end; ++r) {
#endif

                auto ptr = mat->row(r, buffer.data(), wrk.get());
                for (size_t c = 0; c < NC; ++c) {
                    outputs[block[c]](reverse_block_map[c], r) = ptr[c];
                }

#ifndef SCRAN_CUSTOM_PARALLEL
            }
        }
#else
            }
        }, nthreads);
#endif

        return outputs;
    }

    template<typename T, typename IDX, typename B>
    std::vector<Eigen::MatrixXd> core_dense_column(
        const tatami::Matrix<T, IDX>* mat, 
        const B* block, 
        const std::vector<size_t>& block_size,
        const std::vector<size_t>& reverse_block_map 
    ) const {

        size_t nblocks = block_size.size();
        size_t NC = mat->ncol();
        size_t NR = mat->nrow();

        std::vector<Eigen::MatrixXd> outputs;
        outputs.reserve(nblocks);
        for (size_t b = 0; b < nblocks; ++b) {
            outputs.emplace_back(block_size[b], NR);
        }

        // Splitting by row this time, to avoid false sharing across threads.
        size_t rows_per_thread = std::ceil(static_cast<double>(NR) / nthreads);

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel for num_threads(nthreads)
        for (int t = 0; t < nthreads; ++t) {
#else
        SCRAN_CUSTOM_PARALLEL(nthreads, [&](int start, int end) -> void { // Trivial allocation of one job per thread.
        for (int t = start; t < end; ++t) {
#endif

            size_t startrow = rows_per_thread * t, endrow = std::min(startrow + rows_per_thread, NR);
            if (startrow < endrow) {
                auto wrk = mat->new_workspace(false);
                std::vector<double> buffer(endrow - startrow);

                for (size_t c = 0; c < NC; ++c) {
                    auto ptr = mat->column(c, buffer.data(), startrow, endrow, wrk.get());
                    auto b = block[c];
                    auto c2 = reverse_block_map[c];
                    auto& current = outputs[b];

                    for (size_t r = startrow; r < endrow; ++r) {
                        current(c2, r) = ptr[r - startrow];
                    }
                }
            }

#ifndef SCRAN_CUSTOM_PARALLEL
        }
#else
        }
        }, nthreads);
#endif

        return outputs;
    }

    BlockwiseOutputs dense_core(size_t num_features, const std::vector<size_t>& block_size, std::vector<Eigen::MatrixXd> all_matrices) const {
        size_t nblocks = block_size.size();
        std::vector<Eigen::MatrixXd> rotation(nblocks);
        std::vector<double> variance_explained(nblocks);
        std::vector<double> total_variance(nblocks);

        irlba::Irlba irb;
        irb.set_number(1);

        for (size_t b = 0; b < nblocks; ++b) {
            auto& emat = all_matrices[b];
            total_variance[b] = pca_utils::center_and_scale_by_dense_column(emat, scale, nthreads);

            Eigen::MatrixXd pcs;
            Eigen::VectorXd d;
            irlba::EigenThreadScope t(nthreads);
            irb.run(emat, pcs, rotation[b], d);

            variance_explained[b] = d[0] / static_cast<double>(emat.rows() - 1);
        }

        return compute_blockwise_scores(
            rotation, 
            variance_explained, 
            total_variance, 
            block_size,
            [&](size_t b, Eigen::VectorXd& rotation_as_vector, Eigen::VectorXd& scores) -> void {
                irlba::EigenThreadScope t(nthreads);
                scores = all_matrices[b] * rotation_as_vector;
            }
        );
    }

public:
    /**
     * Feature set scoring results.
     */
    struct Results {
        /**
         * Vector of per-cell scores for this feature set.
         * This has length equal to the number of scores in the dataset.
         */
        std::vector<double> scores;

        /**
         * Vector of weights of length equal to the number of features.
         * Each entry contains the weight of each successive feature in the feature set.
         * Weights may be negative.
         */
        std::vector<double> weights;
    };

    template<typename T, typename IDX, typename X, typename B>
    Results run(const tatami::Matrix<T, IDX>* mat, const X* features, const B* block) const {
        std::shared_ptr<const tatami::Matrix<T, IDX> > subsetted;
        {
            size_t NR = mat->nrow();
            std::vector<size_t> which_features;
            for (size_t r = 0; r < NR; ++r) {
                if (features[r]) {
                    which_features.push_back(r);
                }
            }
            subsetted = tatami::make_DelayedSubset<0>(tatami::wrap_shared_ptr(mat), std::move(which_features));
        }

        auto NR = subsetted->nrow();
        auto NC = subsetted->nrow();

        std::vector<size_t> reverse_block_map(NC);
        std::vector<size_t> block_size;
        {
            for (size_t c = 0; c < NC; ++c) {
                size_t b = block[c];
                if (b >= block_size.size()) {
                    block_size.resize(b + 1);
                }

                auto& curoffset = block_size[b];
                reverse_block_map[c] = curoffset;
                ++curoffset;
            }
        }

        BlockwiseOutputs temp;
        if (subsetted->sparse()) {
            if (subsetted->prefer_rows()) {
                auto components = core_sparse_row(subsetted, block, block_size.size(), reverse_block_map);
                temp = sparse_core(NR, block_size, std::move(components));
            } else {
                auto components = core_sparse_column(subsetted, block, block_size.size(), reverse_block_map);
                temp = sparse_core(NR, block_size, std::move(components));
            }
        } else {
            if (subsetted->prefer_rows()) {
                auto subsettedrices = core_dense_row(subsetted, block, block_size, reverse_block_map);
                temp = dense_core(NR, block_size, std::move(subsettedrices));
            } else {
                auto subsettedrices = core_dense_column(subsetted, block, block_size, reverse_block_map);
                temp = dense_core(NR, block_size, std::move(subsettedrices));
            }
        }

        Results output;
        output.weights = std::move(temp.rotation);
        size_t nblocks = temp.block_scores.size();

        if (nblocks > 1) {
            output.scores.resize(NC);
            std::vector<int> positions(block_size.size());
            for (size_t c = 0; c < NC; ++c) {
                auto b = block[c];
                auto& pos = positions[b];
                output.scores[c] = temp.block_scores[b][pos];
                ++pos;
            }
        } else {
            output.scores = std::move(temp.block_scores[0]);
        }

        return output;
    }

    template<typename T, typename IDX, typename X>
    Results run(const tatami::Matrix<T, IDX>* mat, const X* features) const {
        std::vector<uint8_t> dummy_block(mat->ncol());
        return run(mat, features, dummy_block.data());
    }
};

}

#endif
