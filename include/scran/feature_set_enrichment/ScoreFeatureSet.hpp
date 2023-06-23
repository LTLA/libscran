#ifndef SCRAN_SCORE_FEATURE_SET_HPP
#define SCRAN_SCORE_FEATURE_SET_HPP

#include "../utils/macros.hpp"

#include <algorithm>
#include <vector>
#include "tatami/tatami.hpp"

#include "irlba/irlba.hpp"
#include "Eigen/Dense"

#include "../dimensionality_reduction/pca_utils.hpp"

/**
 * @file ScoreFeatureSet.hpp
 * @brief Compute per-cell scores for a feature set.
 */

namespace scran {

/**
 * @brief Compute per-cell scores for a given feature set.
 *
 * Per-cell scores are defined as the column means of the rank-1 approximation of the input matrix for the subset of features in the set.
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
    /**
     * @param b Policy to use when combining rotation vectors from multiple blocks.
     * @return A reference to this `ScoreFeatureSet` instance.
     */
    ScoreFeatureSet& set_block_policy(BlockPolicy b = Defaults::block_policy) {
        block_policy = b;
        return *this;
    }

    /**
     * @param n Number of threads to use.
     * @return A reference to this `ScoreFeatureSet` object.
     */
    ScoreFeatureSet& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

    /**
     * @param s Whether to scale each block so that its features have unit variance.
     * @return A reference to this `ScoreFeatureSet` object.
     */
    ScoreFeatureSet& set_scale(bool s = Defaults::scale) {
        scale = s;
        return *this;
    }

protected:
    /**
     * @cond
     */
    std::vector<double> combine_rotation_vectors(const std::vector<Eigen::MatrixXd>& rotation, const std::vector<double>& variance_explained) const {
        std::vector<double> output;
        size_t nblocks = rotation.size();

        if (block_policy == BlockPolicy::MAXIMUM) {
            double best_var_prop = variance_explained[0];
            size_t best_index = 0;

            for (size_t b = 1; b < nblocks; ++b) {
                double var_prop = variance_explained[b];
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
                double var_prop = variance_explained[b];
                total_var_prop += var_prop;

                // Deciding whether we need to flip the rotation vector or not,
                // as the vectors are not defined w.r.t. their sign.
                double proj = 1;
                if (b && std::inner_product(output.begin(), output.end(), current_rotation.data(), 0.0) < 0.0) {
                    proj = -1;
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
        const std::vector<size_t>& block_size,
        Function mult,
        const std::vector<Eigen::VectorXd>& centers,
        const std::vector<Eigen::VectorXd>& scales 
    ) const {
        // This involves some mild copying of vectors... oh well.
        BlockwiseOutputs output;
        output.rotation = combine_rotation_vectors(rotation, variance_explained);

        size_t nblocks = rotation.size();
        output.block_scores.resize(nblocks);
        Eigen::VectorXd rotation_as_vector(output.rotation.size());
        std::copy(output.rotation.begin(), output.rotation.end(), rotation_as_vector.data()); 

        /* The low-rank representation is defined as (using R syntax here):
         * 
         * L = outer(R, P) * S + C
         * 
         * where P is the per-cell coordinates, R is the rotation vector, S is the scaling vector and C is the centering vector.
         * Our aim is to take the column sums of this matrix, so:
         *
         * colSums(L) = sum(R * S) * P + sum(C)
         *
         * sum(R * S) is called the 'pre-column sum', and is block-specific when 'scale = true'.
         */

        double pre_colsum;
        if (!scale) {
            pre_colsum = std::accumulate(output.rotation.begin(), output.rotation.end(), 0.0);
        }

        for (size_t b = 0; b < nblocks; ++b) {
            size_t cur_cells = block_size[b];
            Eigen::VectorXd scores(cur_cells);
            {
                irlba::EigenThreadScope t(nthreads);
                mult(b, rotation_as_vector, scores);
            }

            const auto& curcenter = centers[b];
            auto to_add = std::accumulate(curcenter.data(), curcenter.data() + curcenter.size(), 0.0);
            if (scale) {
                pre_colsum = std::inner_product(output.rotation.begin(), output.rotation.end(), scales[b].data(), 0.0);
            }

            auto& out = output.block_scores[b];
            out.reserve(scores.size());
            for (auto x : scores) {
                out.push_back(x * pre_colsum + to_add);
            }
        }

        return output;
    }

    static double compute_variance_explained(const Eigen::VectorXd& d, size_t nr, double total_var) {
        if (total_var == 0 || nr < 2) {
            return 0; 
        } else {
            return d[0] * d[0] / static_cast<double>(nr - 1) / total_var;
        }
    }
    /**
     * @endcond
     */

private:
    // Re-using the same two-pass philosophy from RunPCA, to save memory.
    template<typename T, typename IDX, typename B>
    std::vector<pca_utils::SparseComponents> core_sparse_row(const tatami::Matrix<T, IDX>* mat, const B* block, size_t nblocks, const std::vector<size_t>& reverse_block_map) const {
        IDX NC = mat->ncol();
        IDX NR = mat->nrow();

        std::vector<pca_utils::SparseComponents> output(nblocks);
        for (size_t b = 0; b < nblocks; ++b) {
            output[b].ptrs.resize(NR + 1);
        }

        /*** First round, to fetch the number of zeros in each row. ***/
        tatami::parallelize([&](size_t, IDX start, IDX length) -> void {
            tatami::Options opt;
            opt.sparse_extract_value = false;
            opt.sparse_ordered_index = false;
            auto ext = tatami::consecutive_extractor<true, true>(mat, start, length, opt);

            std::vector<IDX> ibuffer(NC);
            for (IDX r = 0, end = start + length; r < end; ++r) {
                auto range = ext->fetch(r, NULL, ibuffer.data());
                for (size_t i = 0; i < range.number; ++i) {
                    ++(output[block[range.index[i]]].ptrs[r + 1]);
                }
            }
        }, NR, nthreads);

        /*** Second round, to populate the vectors. ***/
        std::vector<std::vector<size_t> > ptr_copy;
        ptr_copy.reserve(nblocks);

        for (size_t b = 0; b < nblocks; ++b) {
            auto& curptrs = output[b].ptrs;
            for (size_t r = 0; r < NR; ++r) {
                curptrs[r + 1] += curptrs[r];
            }
            ptr_copy.push_back(curptrs);
            output[b].values.resize(curptrs.back());
            output[b].indices.resize(curptrs.back());
        }

        tatami::parallelize([&](size_t, IDX start, IDX length) -> void {
            std::vector<T> xbuffer(NC);
            std::vector<IDX> ibuffer(NC);
            auto ext = tatami::consecutive_extractor<true, true>(mat, start, length);

            for (IDX r = start, end = start + length; r < end; ++r) {
                auto range = ext->fetch(r, xbuffer.data(), ibuffer.data());
                for (size_t i = 0; i < range.number; ++i) {
                    auto c = range.index[i];
                    auto b = block[c];
                    auto& offset = ptr_copy[b][r];
                    output[b].values[offset] = range.value[i];
                    output[b].indices[offset] = reverse_block_map[c];
                    ++offset;
                }
            }
        }, NR, nthreads);

        return output;
    }

    template<typename T, typename IDX, typename B>
    std::vector<pca_utils::SparseComponents> core_sparse_column(const tatami::Matrix<T, IDX>* mat, const B* block, size_t nblocks, const std::vector<size_t>& reverse_block_map) const {
        IDX NR = mat->nrow();
        IDX NC = mat->ncol();

        /*** First round, to fetch the number of zeros in each row. ***/
        std::vector<std::vector<size_t> > nonzeros_per_row;
        {
            std::vector<std::vector<std::vector<size_t> > > threaded_nonzeros_per_row(nthreads);
            for (auto& x : threaded_nonzeros_per_row) {
                x.resize(nblocks);
                for (auto& current : x) {
                    current.resize(NR);
                }
            }

            tatami::parallelize([&](size_t t, IDX start, IDX length) -> void {
                tatami::Options opt;
                opt.sparse_extract_value = false;
                opt.sparse_ordered_index = false;
                auto wrk = tatami::consecutive_extractor<false, true>(mat, start, length, opt);

                std::vector<IDX> ibuffer(NR);
                auto& nonzeros_per_row = threaded_nonzeros_per_row[t];
                for (size_t c = start, end = start + length; c < end; ++c) {
                    auto range = wrk->fetch(c, NULL, ibuffer.data());
                    auto& current = nonzeros_per_row[block[c]];
                    for (size_t i = 0; i < range.number; ++i) {
                        ++(current[range.index[i]]);
                    }
                }
            }, NC, nthreads);

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
        std::vector<pca_utils::SparseComponents> output(nblocks);
        std::vector<std::vector<size_t> > ptr_copy;
        ptr_copy.reserve(nblocks);

        for (size_t b = 0; b < nblocks; ++b) {
            auto& block_ptrs = output[b].ptrs;
            block_ptrs.resize(NR + 1);

            size_t total_nzeros = 0;
            const auto& nonzeros_per_block = nonzeros_per_row[b];
            for (size_t r = 0; r < NR; ++r) {
                total_nzeros += nonzeros_per_block[r];
                block_ptrs[r + 1] = total_nzeros;
            }
            ptr_copy.push_back(block_ptrs);

            output[b].values.resize(total_nzeros);
            output[b].indices.resize(total_nzeros);
        }

        tatami::parallelize([&](size_t, IDX start, IDX length) -> void {
            std::vector<T> xbuffer(length);
            std::vector<IDX> ibuffer(length);
            auto wrk = tatami::consecutive_extractor<false, true>(mat, 0, NC, start, length);

            for (size_t c = 0; c < NC; ++c) {
                size_t b = block[c];
                auto& block_values = output[b].values;
                auto& block_indices = output[b].indices;
                auto& block_ptr_copy = ptr_copy[b];
                auto blocked_c = reverse_block_map[c];

                auto range = wrk->fetch(c, xbuffer.data(), ibuffer.data());
                for (size_t i = 0; i < range.number; ++i) {
                    auto r = range.index[i];
                    auto& offset = block_ptr_copy[r];
                    block_values[offset] = range.value[i];
                    block_indices[offset] = blocked_c;
                    ++offset;
                }
            }
        }, NR, nthreads);

        return output;
    }

    BlockwiseOutputs sparse_core(size_t num_features, const std::vector<size_t>& block_size, std::vector<pca_utils::SparseComponents> components) const {
        size_t nblocks = block_size.size();
        std::vector<Eigen::MatrixXd> rotation(nblocks);
        std::vector<double> variance_explained(nblocks);

        irlba::Irlba irb;
        irb.set_number(1);

        // Running through and computing the rotation vectors.
        std::vector<pca_utils::SparseMatrix> all_matrices;
        all_matrices.reserve(nblocks);
        std::vector<Eigen::VectorXd> centers;
        centers.reserve(nblocks);
        std::vector<Eigen::VectorXd> scales;
        scales.reserve(nblocks);

        for (size_t b = 0; b < nblocks; ++b) {
            auto& values = components[b].values;
            auto& indices = components[b].indices;
            auto& ptrs = components[b].ptrs;

            centers.emplace_back(num_features);
            auto& center_v = centers.back();
            scales.emplace_back(num_features);
            auto& scale_v = scales.back();
            pca_utils::compute_mean_and_variance_from_sparse_components(num_features, block_size[b], values, indices, ptrs, center_v, scale_v, nthreads);
            double total_var = pca_utils::process_scale_vector(scale, scale_v);

            all_matrices.emplace_back(
                block_size[b], // transposed; we want genes in the columns.
                num_features, 
                std::move(values), 
                std::move(indices), 
                std::move(ptrs),
                nthreads
            );
            const auto& A = all_matrices.back();

            auto& current_rotation = rotation[b];
            if (block_size[b] >= 2) {
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

                variance_explained[b] = compute_variance_explained(d, A.rows(), total_var);
            } else {
                // PCA is not defined here, so just make up whatever.
                current_rotation.resize(num_features, 1);
                current_rotation.fill(0);
                variance_explained[b] = 0;
            }
        }

        return compute_blockwise_scores(
            rotation, 
            variance_explained, 
            block_size,
            [&](size_t b, const Eigen::VectorXd& rotation_as_vector, Eigen::VectorXd& scores) -> void {
                const auto& A = all_matrices[b];
                irlba::Centered<std::remove_reference<decltype(A)>::type> centered(&A, &centers[b]);
                if (scale) {
                    irlba::Scaled<decltype(centered)> scaled(&centered, &scales[b]);
                    auto wrk = scaled.workspace();
                    scaled.multiply(rotation_as_vector, wrk, scores);
                } else {
                    auto wrk = centered.workspace();
                    centered.multiply(rotation_as_vector, wrk, scores);
                }
            },
            centers,
            scales
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
        IDX NC = mat->ncol();
        IDX NR = mat->nrow();

        std::vector<Eigen::MatrixXd> outputs;
        outputs.reserve(nblocks);
        for (size_t b = 0; b < nblocks; ++b) {
            outputs.emplace_back(block_size[b], NR);
        }

        tatami::parallelize([&](size_t, IDX start, IDX length) -> void {
            auto ext = tatami::consecutive_extractor<true, false>(mat, start, length);
            std::vector<T> buffer(NC);

            for (IDX r = start, end = start + length; r < end; ++r) {
                auto ptr = ext->fetch(r, buffer.data());
                for (size_t c = 0; c < NC; ++c) {
                    outputs[block[c]](reverse_block_map[c], r) = ptr[c];
                }
            }
        }, NR, nthreads);

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
        IDX NC = mat->ncol();
        IDX NR = mat->nrow();

        std::vector<Eigen::MatrixXd> outputs;
        outputs.reserve(nblocks);
        for (size_t b = 0; b < nblocks; ++b) {
            outputs.emplace_back(block_size[b], NR);
        }

        tatami::parallelize([&](size_t, IDX start, IDX length) -> void {
            auto ext = tatami::consecutive_extractor<false, false>(mat, 0, NC, start, length);
            std::vector<T> buffer(NR);

            for (size_t c = 0; c < NC; ++c) {
                auto b = block[c];
                auto c2 = reverse_block_map[c];
                auto& current = outputs[b];

                auto ptr = ext->fetch(c, buffer.data());
                for (size_t r = 0; r < length; ++r) {
                    current(c2, r + start) = ptr[r];
                }
            }
        }, NR, nthreads);

        return outputs;
    }

    BlockwiseOutputs dense_core(size_t num_features, const std::vector<size_t>& block_size, std::vector<Eigen::MatrixXd> all_matrices) const {
        size_t nblocks = block_size.size();
        std::vector<Eigen::MatrixXd> rotation(nblocks);
        std::vector<Eigen::VectorXd> centers(nblocks);
        std::vector<Eigen::VectorXd> scales(nblocks);
        std::vector<double> variance_explained(nblocks);

        irlba::Irlba irb;
        irb.set_number(1);

        for (size_t b = 0; b < nblocks; ++b) {
            auto& emat = all_matrices[b];
            centers[b].resize(num_features);
            scales[b].resize(num_features);
            pca_utils::compute_mean_and_variance_from_dense_columns(emat, centers[b], scales[b], nthreads);

            double total_var = pca_utils::process_scale_vector(scale, scales[b]);
            pca_utils::center_and_scale_dense_columns(emat, centers[b], scale, scales[b], nthreads);

            if (block_size[b] >= 2) {
                Eigen::MatrixXd pcs;
                Eigen::VectorXd d;
                irlba::EigenThreadScope t(nthreads);
                irb.run(emat, pcs, rotation[b], d);
                variance_explained[b] = compute_variance_explained(d, emat.rows(), total_var);
            } else {
                // PCA is not defined here, so just make up whatever.
                rotation[b].resize(num_features, 1);
                rotation[b].fill(0);
                variance_explained[b] = 0;
            }
        }

        return compute_blockwise_scores(
            rotation, 
            variance_explained, 
            block_size,
            [&](size_t b, const Eigen::VectorXd& rotation_as_vector, Eigen::VectorXd& scores) -> void {
                scores = all_matrices[b] * rotation_as_vector; 
            },
            centers,
            scales
        );
    }

public:
    /**
     * @brief Feature set scoring results.
     */
    struct Results {
        /**
         * Vector of per-cell scores for this feature set.
         * This has length equal to the number of scores in the dataset.
         */
        std::vector<double> scores;

        /**
         * Vector of weights of length equal to the number of features in the set.
         * Each entry contains the weight of each successive feature in the feature set.
         * Weights may be negative.
         */
        std::vector<double> weights;
    };

    /**
     * @tparam T Floating point type for the data.
     * @tparam IDX Integer type for the indices.
     * @tparam X Integer type for the feature filter.
     * @tparam Block Integer type for the block assignments.
     *
     * @param[in] mat Pointer to the input matrix.
     * Columns should contain cells while rows should contain genes.
     * @param[in] features Pointer to an array of length equal to the number of rows in `mat`, specifying the features in the set of interest.
     * Non-zero values indicate that the corresponding row is part of the feature set.
     * @param[in] block Pointer to an array of length equal to the number of columns in `mat`.
     * This should contain the blocking factor as 0-based block assignments 
     * (i.e., for `n` blocks, block identities should run from 0 to `n-1` with at least one entry for each block.)
     * If this is `NULL`, all cells are assumed to belong to the same block.
     *
     * @return A `Results` object containing the per-cell scores and per-feature weights.
     */
    template<typename T, typename IDX, typename X, typename Block>
    Results run_blocked(const tatami::Matrix<T, IDX>* mat, const X* features, const Block* block) const {
        std::shared_ptr<const tatami::Matrix<T, IDX> > subsetted = pca_utils::subset_matrix_by_features(mat, features);
        auto NR = subsetted->nrow();
        auto NC = subsetted->ncol();

        // Catching edge cases.
        if (NR == 0) {
            Results output;
            output.scores.resize(NC);
            return output;
        } else if (NR == 1) {
            Results output;
            output.weights.push_back(1);
            output.scores = subsetted->dense_row()->fetch(0);
            return output;
        } else if (NC == 0) {
            Results output;
            output.weights.resize(NR);
            return output;
        }

        std::vector<Block> dummy_block;
        if (block == NULL) {
            dummy_block.resize(mat->ncol());
            block = dummy_block.data();
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
        if (subsetted->sparse()) {
            if (subsetted->prefer_rows()) {
                auto components = core_sparse_row(subsetted.get(), block, block_size.size(), reverse_block_map);
                temp = sparse_core(NR, block_size, std::move(components));
            } else {
                auto components = core_sparse_column(subsetted.get(), block, block_size.size(), reverse_block_map);
                temp = sparse_core(NR, block_size, std::move(components));
            }
        } else {
            if (subsetted->prefer_rows()) {
                auto matrices = core_dense_row(subsetted.get(), block, block_size, reverse_block_map);
                temp = dense_core(NR, block_size, std::move(matrices));
            } else {
                auto matrices = core_dense_column(subsetted.get(), block, block_size, reverse_block_map);
                temp = dense_core(NR, block_size, std::move(matrices));
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

        for (auto& s : output.scores) {
            s /= NR; // no need to protect against NR = 0, as we caught that above.
        }

        return output;
    }

    /**
     * An overload of `run()` where all cells are assumed to belong to the same block.
     * 
     * @tparam T Floating point type for the data.
     * @tparam IDX Integer type for the indices.
     * @tparam X Integer type for the feature filter.
     *
     * @param[in] mat Pointer to the input matrix.
     * Columns should contain cells while rows should contain genes.
     * @param[in] features Pointer to an array of length equal to the number of rows in `mat`, specifying the features in the set of interest.
     * Non-zero values indicate that the corresponding row is part of the feature set.
     *
     * @return A `Results` object containing the per-cell scores and per-feature weights.
     */
    template<typename T, typename IDX, typename X>
    Results run(const tatami::Matrix<T, IDX>* mat, const X* features) const {
        return run_blocked(mat, features, static_cast<unsigned char*>(NULL));
    }
};

}

#endif
