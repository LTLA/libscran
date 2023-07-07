#ifndef SCRAN_NEIGHBOR_SCALING_HPP
#define SCRAN_NEIGHBOR_SCALING_HPP

#include "../utils/macros.hpp"

#include <vector>
#include <cmath>
#include <memory>
#include "knncolle/knncolle.hpp"
#include "tatami/stats/medians.hpp"

/**
 * @file ScaleByNeighbors.hpp
 *
 * @brief Scale multi-modal embeddings based on their relative variance.
 */

namespace scran {

/**
 * @brief Scale multi-modal embeddings to adjust for differences in variance.
 *
 * The premise is that we have multiple embeddings for the same set of cells, usually generated from different data modalities (e.g., RNA, protein, and so on).
 * We would like to combine these embeddings into a single embedding for downstream analyses such as clustering and t-SNE/UMAP.
 * The easiest combining strategy is to just concatenate the matrices together into a single embedding that contains information from all modalities.
 * However, this is complicated by the differences in the variance between modalities, whereby higher noise in one modality might drown out biological signal in another embedding.
 *
 * This class implements a scaling approach to equalize noise across embeddings prior to concatenation.
 * We compute the median distance to the $k$-th nearest neighbor in each embedding;
 * this is used as a proxy for the noise within a subpopulation containing at least $k$ cells.
 * We then compute a scaling factor as the ratio of the medians for a "target" embedding compared to a "reference" embedding.
 * The idea is to scale the target embedding by this factor before concatenation of matrices.
 *
 * This approach aims to remove differences in the magnitude of noise while preserving genuine differences in biological signal.
 * Embedding-specific subpopulations are preserved in the concatenation, provided that the difference between subpopulations is greater than that within subpopulations.
 * By contrast, a naive scaling based on the total variance would penalize embeddings with strong biological heterogeneity.
 */
class ScaleByNeighbors {
public:
    /**
     * @brief Default parameter settings.
     */
    struct Defaults { 
        /**
         * See `set_neighbors()` for more details.
         */
        static constexpr int neighbors = 20;

        /**
         * See `set_approximate()` for more details.
         */
        static constexpr bool approximate = false;

        /**
         * See `set_num_threads()`.
         */
        static constexpr int num_threads = 1;
    };

    /**
     * @param n Number of neighbors used in the nearest neighbor search.
     * This can be interpreted as the minimum size of each subpopulation.
     *
     * @return A reference to this `ScaleByNeighbors` instance.
     */
    ScaleByNeighbors& set_neighbors(int n = Defaults::neighbors) {
        num_neighbors = n;
        return *this;
    }

    /**
     * @param a Whether to perform an approximate neighbor search.
     *
     * @return A reference to this `ScaleByNeighbors` instance.
     *
     * This parameter only has an effect when an array is passed to `run()`,
     * otherwise the search uses the algorithm specified by the **knncolle** class.
     */
    ScaleByNeighbors& set_approximate(int a = Defaults::approximate) {
        approximate = a;
        return *this;
    }

    /**
     * @param n Number of threads to use. 
     * @return A reference to this `ScaleByNeighbors` object.
     */
    ScaleByNeighbors& set_num_threads(int n = Defaults::num_threads) {
        nthreads = n;
        return *this;
    }

private:
    int num_neighbors = Defaults::neighbors;
    bool approximate = Defaults::approximate;
    int nthreads = Defaults::num_threads;

public:
    /**
     * @param ndim Number of dimensions in the embedding.
     * @param ncells Number of cells in the embedding.
     * @param[in] data Pointer to an array containing the embedding matrix.
     * This should be stored in column-major layout where each row is a dimension and each column is a cell.
     *
     * @return Pair containing the median distance to the $k$-th nearest neighbor (first)
     * and the root-mean-squared distance across all cells (second).
     * These values can be used in `compute_scale()`.
     */
    std::pair<double, double> compute_distance(int ndim, size_t ncells, const double* data) const {
        std::unique_ptr<knncolle::Base<> > ptr;
        if (!approximate) {
            ptr.reset(new knncolle::VpTreeEuclidean<>(ndim, ncells, data));
        } else {
            ptr.reset(new knncolle::AnnoyEuclidean<>(ndim, ncells, data));
        }
        return compute_distance(ptr.get());
    }

    /**
     * @tparam Search Search index class, typically a `knncolle::Base` subclass.
     * @param search Search index for the embedding.
     *
     * @return Pair containing the median distance to the $k$-th nearest neighbor (first)
     * and the root-mean-squared distance across all cells (second).
     * These values can be used in `compute_scale()`.
     */
    template<class Search>
    std::pair<double, double> compute_distance(const Search* search) const {
        size_t nobs = search->nobs();
        std::vector<double> dist(nobs);

        tatami::parallelize([&](size_t, size_t start, size_t length) -> void {
            for (size_t i = start, end = start + length; i < end; ++i) {
                auto neighbors = search->find_nearest_neighbors(i, num_neighbors);
                dist[i] = neighbors.back().second;
            }
        }, nobs, nthreads);

        double med = tatami::stats::compute_median<double>(dist.data(), nobs);
        double rmsd = 0;
        for (auto d : dist) {
            rmsd += d * d;
        }
        rmsd = std::sqrt(rmsd);
        return std::make_pair(med, rmsd);
    }

public:
    /**
     * Compute the scaling factors for a group of embeddings, given the neighbor distances computed by `compute_distance()`.
     * This aims to scale each embedding so that the within-population variances are equal across embeddings.
     * The "reference" embedding is defined as the first embedding with a non-zero RMSD; 
     * other than this requirement, the exact choice of reference has no actual impact on the relative values of the scaling factors.
     *
     * @param distances Vector of distances for embeddings, as computed by `compute_scale()` on each embedding.
     *
     * @return Vector of scaling factors of length equal to that of `distances`, to be applied to each embedding.
     */
    static std::vector<double> compute_scale(const std::vector<std::pair<double, double> >& distances) {
        std::vector<double> output(distances.size());

        // Use the first entry with a non-zero RMSD as the reference.
        bool found_ref = false;
        size_t ref = 0;
        for (size_t e = 0; e < distances.size(); ++e) {
            if (distances[e].second) {
                found_ref = true;
                ref = e;
                break;
            }
        }

        // If all of them have a zero RMSD, then all scalings are zero, because it doesn't matter.
        if (found_ref) {
            const auto& dref = distances[ref];
            for (size_t e = 0; e < distances.size(); ++e) {
                output[e] = (e == ref ? 1 : compute_scale(dref, distances[e]));
            }
        }

        return output;
    }

    /**
     * Compute the scaling factor to be applied to a target embedding relative to a reference.
     * This aims to scale the target so that the within-population variance is equal to that of the reference.
     *
     * @param ref Output of `compute_distance()` for the reference embedding.
     * The first value contains the median distance while the second value contains the RMSD.
     * @param target Output of `compute_distance()` for the target embedding.
     *
     * @return A scaling factor to apply to the target embedding.
     * Scaling all values in the target matrix by the factor will equalize the magnitude of the noise to that of the reference embedding.
     *
     * If either of the median distances is zero, this function automatically switches to the root-mean-square-distance to the $k$-th neighbor.
     * The scaling factor is then defined as the ratio of the RMSDs between embeddings.
     * If the reference RMSDs is zero, this function will return zero;
     * if the target RMSD is zero, this function will return positive infinity.
     */
    static double compute_scale(const std::pair<double, double>& ref, const std::pair<double, double>& target) {
        if (target.first == 0 || ref.first == 0) {
            if (target.second == 0) {
                return std::numeric_limits<double>::infinity();
            } else if (ref.second == 0) {
                return 0;
            } else {
                return ref.second / target.second; 
            }
        } else {
            return ref.first / target.first;
        }
    }

    /**
     * Combine multiple embeddings into a single embedding matrix, possibly after scaling each embedding.
     * This is done row-wise, i.e., the coordinates are concatenated across embeddings for each column.
     * 
     * @tparam Embed Pointer type to the input embeddings.
     * 
     * @param ndims Vector containing the number of dimensions in each embedding.
     * @param ncells Number of cells in each embedding.
     * @param embeddings Vector of pointers of length equal to that of `ndims`.
     * Each pointer refers to an array containing an embedding matrix, which should be in column-major format with dimensions in rows and cells in columns.
     * The number of rows should be equal to the corresponding element in `ndims` and the number of columns should be equal to `ncells`.
     * @param scaling Scaling to apply to each embedding, usually from `compute_scale()`.
     * This should be of length equal to that of `ndims`.
     * @param[out] output Pointer to the output array.
     * This should be of length equal to the product of `ncells` and the sum of `ndims`.
     * On completion, `output` is filled with the combined embeddings in column-major format.
     * Each row corresponds to a dimension while each column corresponds to a cell.
     */
    template<typename Embed>
    static void combine_scaled_embeddings(const std::vector<int>& ndims, size_t ncells, const std::vector<Embed>& embeddings, const std::vector<double>& scaling, double* output) {
        size_t nembed = ndims.size();
        if (embeddings.size() != nembed || scaling.size() != nembed) {
            throw std::runtime_error("'ndims', 'embeddings' and 'scale' should have the same length");
        }

        int ntotal = std::accumulate(ndims.begin(), ndims.end(), 0);
        size_t offset = 0;

        for (size_t e = 0; e < nembed; ++e) {
            size_t curdim = ndims[e];
            auto inptr = embeddings[e];
            auto outptr = output + offset;
            auto s = scaling[e];

            if (std::isinf(s)) {
                // If the scaling factor is infinite, it implies that the current
                // embedding is all-zero, so we just fill with zeros, and move on.
                for (size_t c = 0; c < ncells; ++c, inptr += curdim, outptr += ntotal) {
                    std::fill(outptr, outptr + curdim, 0);
                }
            } else {
                for (size_t c = 0; c < ncells; ++c, inptr += curdim, outptr += ntotal) {
                    for (size_t d = 0; d < curdim; ++d) {
                        outptr[d] = inptr[d] * s;
                    }
                }
            }

            offset += curdim;
        }
    }
};

}

#endif
