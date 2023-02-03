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

protected:
    std::vector<double> combine_rotation_vectors(const std::vector<Eigen::MatrixXd>& rotation, std::vector<double>& variance_explained, std::vector<double>& total_variance) const {
        std::vector<double> output;
        auto compute_var_prop = [&](int b) -> double {
            if (total_variance[b] == 0) {
                return 0; // avoid UB upon an unfortunate div by zero.
            } else {
                return variance_explained[b][0] / total_variance[b];
            }
        };

        if (batch_policy == BatchPolicy::MAXIMUM) {
            double best_var_prop = compute_var_prop(0);
            size_t best_index = 0;

            for (size_t b = 1; b < nblocks; ++b) {
                double var_prop = compute_var_prop(b);
                if (var_prop > overall_var_prop) {
                    best_index = b;
                    best_var_prop = var_prop;
                }
            }

            const auto& rotation = rotation[best_index];
            output.insert(output.end(), rotation.begin(), rotation.end());

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
                    if (std::inner_product(rotation[0].begin(), rotation[0].end(), current_rotation.begin()) < 0) {
                        proj = -1;
                    }
                }

                double mult = proj * var_proj;
                for (size_t f = 0; f < nfeatures; ++f) {
                    output[f] += mult * current_rotation[f];
                }
            }

            if (total_var_prop) {
                for (size_t f = 0; f < nfeatures; ++f) {
                    output[f] /= total_var_prop;
                }
            }
        }

        return output;
    }

    struct BlockwiseOutputs {
        std::vector<double> rotation;
        std::vector<std::vector<double> > block_scores;
    };

    template<class Inputs, class Function>
    BlockwiseOutputs compute_blockwise_scores(const std::vector<Eigen::MatrixXd>& rotation, const std::vector<double>& variance_explained, const std::vector<double>& total_variance, Function mult) const {
        // This involves some mild copying of vectors... oh well.
        BlockwiseOutputs output;
        output.rotation = combine_rotation_vectors(rotation, variance_explained, total_variance);

        size_t nblocks = rotation.size();
        output.block_scores.reserve(nblocks);
        Eigen::VectorXd rotation_as_vector(output.rotation.begin(), output.rotation.end()); 

        for (size_t b = 0; b < nblocks; ++b) {
            Eigen::VectorXd scores(block_size[b]);
            mult(b, rotation_as_vector, scores);
            output.block_scores[b].insert(output.block_scores[b].end(), scores.begin(), scores.end());
        }

        return output;

private:
    // Re-using the same two-pass philosophy from RunPCA, to save memory.
    struct BlockwiseSparseComponents {
        BlockedSparseComponents(size_t nblocks) : ptrs(nblocks), values(nblocks), indices(nblocks) {}
        std::vector<std::vector<size_t> > ptrs;
        std::vector<std::vector<double> > values;
        std::vector<std::vector<int> > indices;
    };

    BlockwiseSparseComponents core_sparse_row(
        const tatami::Matrix<T, IDX>* mat, 
        const X* features, 
        const std::vector<size_t>& which_features,
        const std::vector<size_t>& reverse_feature_map, 
        const B* block, 
        size_t nblocks,
        const std::vector<size_t>& reverse_block_map
    ) const {

        size_t num_features = which_features.size();
        BlockedSparseComponents output(nblocks);
        auto& ptrs = output.ptrs;
        for (size_t b = 0; b < nblocks; ++b) {
            ptrs.resize(num_features + 1);
        }

        /*** First round, to fetch the number of zeros in each row. ***/
        {
#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp parallel num_threads(nthreads)
            {
#else
            SCRAN_CUSTOM_PARALLEL(num_features, [&](size_t start, size_t end) -> void {
#endif            

                std::vector<double> xbuffer(NC);
                std::vector<int> ibuffer(NC);
                auto wrk = mat->new_workspace(true);

#ifndef SCRAN_CUSTOM_PARALLEL
                #pragma omp for
                for (size_t f = 0; f < num_features; ++r) {
#else
                for (size_t f = start; f < end; ++r) {
#endif

                    size_t r = which_features[f];
                    size_t index = reverse_feature_map[r] + 1;
                    auto range = mat->sparse_row(r, xbuffer.data(), ibuffer.data(), wrk.get());
                    for (size_t i = 0; i < range.number; ++i) {
                        ++(ptrs[block[range.index[i]]][index]);
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
                values[b].resize(ptrs.back());
                indices[b].resize(ptrs.back());
            }
            auto ptr_copy = ptr;

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp parallel num_threads(nthreads)
            {
#else
            SCRAN_CUSTOM_PARALLEL(NR, [&](size_t start, size_t end) -> void {
#endif            

                auto wrk = mat->new_workspace(true);

#ifndef SCRAN_CUSTOM_PARALLEL
                #pragma omp for
                for (size_t r = 0; r < NR; ++r) {
#else
                for (size_t r = start; r < end; ++r) {
#endif

                    size_t r2 = reverse_feature_map[r];
                    auto range = mat->sparse_row(which_features[r], xbuffer.data(), ibuffer.data(), wrk.get());
                    for (size_t i = 0; i < range.number; ++i) {
                        auto b = block[range.index[i]];
                        auto& offset = ptr_copy[b][r2];
                        values[b][offset] = range.value[i];
                        indices[b][offset] = reverse_block_map[range.index[i]]];
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
    }

    BlockwiseSparseComponents core_sparse_column(
        const tatami::Matrix<T, IDX>* mat, 
        const X* features, 
        const std::vector<size_t>& which_features,
        const std::vector<size_t>& reverse_feature_map, 
        const B* block, 
        size_t nblocks,
        const std::vector<size_t>& reverse_block_map 
    ) const {

        auto NC = mat->ncol();
        size_t num_features = which_features.size();
        size_t first_feature = (num_features ? 0 : which_features.front());
        size_t last_feature = (num_features ? 0 : which_features.back() + 1);
        size_t gap_size = last_feature - first_feature;

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
                        nonzeroes_per_row[b].resize(num_features);
                    }

                    std::vector<double> xbuffer(gap_size);
                    std::vector<int> ibuffer(gap_size);
                    auto wrk = mat->new_workspace(false);

                    for (size_t c = startcol; c < endcol; ++c) {
                        auto range = mat->sparse_column(c, xbuffer.data(), ibuffer.data(), first_feature, last_feature, wrk.get());
                        auto& current = nonzeros_per_row[block[c]];
                        for (size_t i = 0; i < range.number; ++i) {
                            if (features[i]) {
                                ++(current[reverse_feature_map[range.index[i]]]);
                            }
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
        BlockedSparseComponents output(nblocks);
        {
            auto& ptrs = output.ptrs;
            auto& values = output.values;
            auto& indices = output.indices;

            for (size_t b = 0; b < nblocks; ++b) {
                ptrs.resize(num_features + 1);
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
            size_t rows_per_thread = std::ceil(static_cast<double>(gap_size) / nthreads);
            auto ptr_copy = ptrs;

#ifndef SCRAN_CUSTOM_PARALLEL
            #pragma omp parallel for num_threads(nthreads)
            for (int t = 0; t < nthreads; ++t) {
#else
            SCRAN_CUSTOM_PARALLEL(nthreads, [&](int start, int end) -> void { // Trivial allocation of one job per thread.
            for (int t = start; t < end; ++t) {
#endif

                size_t startrow = first_feature + rows_per_thread * t, endrow = std::min(startrow + rows_per_thread, last_feature);
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
                            if (features[r]) {
                                auto r2 = reverse_feature_map[r];
                                auto& offset = block_ptr_copy[r2];
                                block_values[offset] = range.value[i];
                                block_indices[offset] = blocked_c;
                                ++offset;
                            }
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

    BlockwiseOutputs sparse_core_internal(size_t num_features, const std::vector<size_t>& block_size, BlockwiseSparseComponents& components) const {
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

            auto& rotation = rotation[b];
            Eigen::MatrixXd pcs;
            Eigen::VectorXd d;

            irlba::EigenThreadScope t(nthreads);
            irlba::Centered<decltype(emat)> centered(&A, &center_v);
            if (scale) {
                irlba::Scaled<decltype(centered)> scaled(&centered, &scale_v);
                irb.run(scaled, pcs, rotation, d);
            } else {
                irb.run(centered, pcs, rotation, d);
            }

            variance_explained[b] = d[0] / static_cast<double>(emat.rows() - 1);
        }

        return compute_blockwise_scores(rotation, variance_explained, total_variance, 
            [&](size_t b, Eigen::VectorXd& rotation_as_vector, Eigen::VectorXd& scores) -> void {
                irlba::EigenThreadScope t(nthreads);
                irlba::Centered<decltype(emat)> centered(all_matrices[b], &(center_v[b]));
                if (scale) {
                    irlba::Scaled<decltype(centered)> scaled(&centered, &(scale_v[b]));
                    scaled.multiply(rotation_as_vector, scores);
                } else {
                    centered.multiply(rotation_as_vector, scores);
                }
            }
        );
    }

private:
    std::vector<Eigen::MatrixXd> core_dense_row(
        const tatami::Matrix<T, IDX>* mat, 
        const std::vector<size_t>& which_features,
        const std::vector<size_t>& reverse_feature_map, 
        const B* block, 
        const std::vector<size_t>& block_size,
        const std::vector<size_t>& reverse_block_map 
    ) const {

        size_t nfeatures = which_features.size();
        size_t nblocks = block_size.size();
        size_t NC = mat->ncol();

        std::vector<Eigen::MatrixXd> outputs;
        outputs.reserve(nblocks);
        for (size_t b = 0; b < nblocks; ++b) {
            outputs.emplace_back(block_size[b], nfeatures);
        }

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel num_threads(nthreads)
        {
            std::vector<double> work(NC);
            auto work = mat->new_workspace(true);

            #pragma omp for
            for (int f = 0; f < nfeatures; ++f) {
#else
        SCRAN_CUSTOM_PARALLEL(nfeatures, [&](int start, int end) -> void {
            std::vector<double> work(NC);
            auto work = mat->new_workspace(true);
            for (int f = start; f < end; ++f) {
#endif

                size_t r = which_features[f];
                size_t r2 = reverse_feature_map[r];
                auto ptr = mat->row(r, work.data());    
                for (size_t c = 0; c < NC; ++c) {
                    auto b = block[c];
                    outputs[b](reverse_block_map[c], r2) = ptr[c];
                }

#ifndef SCRAN_CUSTOM_PARALLEL
            }
        }
#else
            }
        }, nthreads);
#endif

        return output;
    }

    std::vector<Eigen::MatrixXd> core_dense_row(
        const tatami::Matrix<T, IDX>* mat, 
        const std::vector<size_t>& which_features,
        const std::vector<size_t>& reverse_feature_map, 
        const B* block, 
        const std::vector<size_t>& block_size,
        const std::vector<size_t>& reverse_block_map 
    ) const {

        size_t nfeatures = which_features.size();
        size_t nblocks = block_size.size();
        size_t NC = mat->ncol();

        std::vector<Eigen::MatrixXd> outputs;
        outputs.reserve(nblocks);
        for (size_t b = 0; b < nblocks; ++b) {
            outputs.emplace_back(nfeatures, block_size[b]);
        }

        size_t first_feature = (num_features ? 0 : which_features.front());
        size_t last_feature = (num_features ? 0 : which_features.back() + 1);
        size_t gap_size = last_feature - first_feature;

#ifndef SCRAN_CUSTOM_PARALLEL
        #pragma omp parallel num_threads(nthreads)
        {
            std::vector<double> work(gap_size);
            auto work = mat->new_workspace(false);

            #pragma omp for
            for (int c = 0; c < NC; ++c) {
#else
        SCRAN_CUSTOM_PARALLEL(NC, [&](int start, int end) -> void {
            std::vector<double> work(gap_size);
            auto work = mat->new_workspace(false);
            for (int c = start; c < end; ++c) {
#endif

                auto ptr = mat->column(c, first_feature, last_feature, work.data());    
                auto& current_mat = outputs[block[c]];
                auto current_col = reverse_block_map[c];
                for (auto r : which_features) {
                    current(reverse_feature_map[r], current_col) = ptr[r - first_feature];
                }

#ifndef SCRAN_CUSTOM_PARALLEL
            }
        }
#else
            }
        }, nthreads);
#endif

        for (size_t b = 0; b < nblocks; ++b) {
            outputs[b].adjointInPlace();
        }

        return output;
    }

    BlockwiseOutputs dense_core_internal(size_t num_features, const std::vector<size_t>& block_size, std::vector<Eigen::MatrixXd>& all_matrices) const {
        size_t nblocks = block_size.size();
        std::vector<Eigen::MatrixXd> rotation(nblocks);
        std::vector<double> variance_explained(nblocks);
        std::vector<double> total_variance(nblocks);

        irlba::Irlba irb;
        irb.set_number(1);

        for (size_t b = 0; b < nblocks; ++b) {
            auto& emat = all_matrices[b];
            total_variance[b] = pca_utils::center_and_scale_by_dense_column(emat, scale, total_var, nthreads);

            Eigen::MatrixXd pcs;
            Eigen::VectorXd d;
            irlba::EigenThreadScope t(nthreads);
            irb.run(emat, pcs, rotation[b], d);

            variance_explained[b] = d[0] / static_cast<double>(emat.rows() - 1);
        }

        return compute_blockwise_scores(rotation, variance_explained, total_variance, 
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

    template<typename T, typename IDX, typename X>
    void run(const tatami::Matrix<T, IDX>* mat, const X* features, const B* block) const {
        auto NR = mat->nrow();
        auto NC = mat->nrow();

        std::vector<size_t> reverse_feature_map(NR);
        std::vector<size_t> which_features;
        {
            for (size_t r = 0; r < NR; ++r) {
                if (features[r]) {
                    reverse_feature_map[r] = which_features.size();
                    which_features.push_back(r);
                }
            }
        }

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
        if (mat->is_sparse()) {
            if (mat->prefer_rows()) {
                auto components = core_sparse_row(mat, features, which_features, reverse_feature_map, block, block_size.size(), reverse_block_map);
                return sparse_core_internal(which_features.size(), block_size, components);
            } else {
                auto components = core_sparse_column(mat, features, which_features, reverse_feature_map, block, block_size.size(), reverse_block_map);
                return sparse_core_internal(which_features.size(), block_size, components);
            }
        } else {
            if (mat->prefer_rows()) {
                auto matrices = core_dense_row(mat, which_features, reverse_feature_map, block, block_size.size(), reverse_block_map);
                temp = dense_core_internal(which_features.size(), block_size, components);
            } else {
                auto matrices = core_dense_column(mat, which_features, reverse_feature_map, block, block_size.size(), reverse_block_map);
                return dense_core_internal(which_features.size(), block_size, components);
            }
        }

        Results output;
        output.weights = std::move(temp.rotation);
        size_t nblocks = temp.block_scores.size();

        if (nblocks > 1) {
            size_t NC = mat->ncol();
            output.scores.resize(NC);
            std::vector<int> positions();
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
};

}

#endif
