#ifndef SCRAN_BUILDSNNGRAPH_HPP
#define SCRAN_BUILDSNNGRAPH_HPP

#include "knncolle/VpTree/VpTree.hpp"

namespace scran {

class BuildSNNGraph {
public:
    BuildSNNGraph& set_neighbors(int k = 10) {
        num_neighbors = k;
        return *this;
    }

    BuildSNNGraph& set_shared(bool s = true) {
        shared = s;
        return *this;
    }

public:
    enum Scheme { RANKED, NUMBER, JACCARD };

    BuildSNNGraph& set_weighting_scheme(Scheme w = RANKED) {
        weight_scheme = w;
        return *this;
    }

private:
    typedef std::vector<std::vector<std::pair<size_t, int> > > neighbor2index;
    typedef std::vector<std::vector<knncolle::CellIndex_t> > index2neighbor;

public:
    // Build a NN graph from a feature x cell matrix.
    void run(size_t ndims, size_t ncells, const double* mat) {
        knncolle::VpTreeEuclidean vp(ndims, ncells, mat);

        // Collecting neighbors.
        std::vector<double> distances;
        index2neighbor indices;
        neighbor2index hosts(ncells);

        for (size_t i = 0; i < ncells; ++i) {
            auto& current = indices[i];
            vp.find_nearest_neighbors(i, num_neighbors, current, distances, true, false, false);

            int counter = 1;
            for (auto x : current) {
                hosts[x].push_back(std::make_pair(counter, i));
                ++counter;
            }
        }

        construct_shared_neighbor_edges(indices, hosts);
        return;
    }

private:
    void construct_shared_neighbor_edges(const index2neighbor& neighbors, const neighbor2index& hosts) {
        std::deque<size_t> output_pairs;
        std::deque<double> output_weights;
        std::deque<size_t> current_added;
        std::vector<int> current_score(neighbors.size());

        for (size_t j = 0; j < neighbors.size(); ++j) {
            for (int i = 0; i <= num_neighbors; ++i) {
                // First iteration treats 'j' as the zero-th neighbor.
                // Remaining iterations go through the neighbors of 'j'.
                int cur_neighbor;
                if (i==0) {
                    cur_neighbor = j;
                } else {
                    cur_neighbor = neighbors[j][i-1];

                    if (cur_neighbor < j) { // avoid duplicates from symmetry in the SNN calculations.
                        auto& existing_other=current_score[cur_neighbor];
                        if (weight_scheme == RANKED) {
                            // Computing the weight of the edge to neighbor 'i'. In this case,
                            // 'i' is the rank of 'i' in 'j's list, and the rank of 'i' in its
                            // own list is zero, so the weight is just 'i + 0 = i'.
                            const auto& currank = i;
                            if (existing_other == 0) { 
                                existing_other = currank;
                                current_added.push_back(cur_neighbor);
                            } else if (existing_other > currank) {
                                existing_other = currank;
                            }
                        } else {
                            // Weight is just the number of shared neighbors; obviously, 'i' itself
                            // is a shared neighbor of 'j' and itself, so we increment by 1.
                            if (existing_other==0) { 
                                current_added.push_back(cur_neighbor);
                            } 
                            ++existing_other;
                        }
                    }
                }

                // Going through all observations 'h' for which 'cur_neighbor'
                // is a nearest neighbor, a.k.a., 'cur_neighbor' is a shared
                // neighbor of both 'h' and 'j'.
                for (auto& h : hosts[cur_neighbor]) {
                    const auto& othernode = h.second;

                    if (othernode < j) { // avoid duplicates from symmetry in the SNN calculations.
                        size_t& existing_other = current_score[othernode];
                        if (weight_scheme == RANKED) {
                            // Recording the lowest combined rank per neighbor.
                            int currank = h.first + i;
                            if (existing_other == 0) { 
                                existing_other = currank;
                                current_added.push_back(othernode);
                            } else if (existing_other > currank) {
                                existing_other = currank;
                            }
                        } else {
                            // Recording the number of shared neighbors.
                            if (existing_other==0) { 
                                current_added.push_back(othernode);
                            } 
                            ++existing_other;
                        }
                    }
                }
            }
           
            for (auto othernode : current_added) {
                // Converting to edges.
                output_pairs.push_back(j);
                output_pairs.push_back(othernode);

                // Ensuring that an edge with a positive weight is always reported.
                size_t& otherscore=current_score[othernode];
                double finalscore;
                if (weight_scheme == RANKED) {
                    finalscore = static_cast<double>(num_neighbors) - 0.5 * static_cast<double>(otherscore);
                } else {
                    finalscore = otherscore;
                    if (weight_scheme == JACCARD) {
                        finalscore = finalscore / (2 * (num_neighbors + 1) - finalscore);
                    }
                }
                output_weights.push_back(std::max(finalscore, 1e-6));

                // Resetting all those added to zero.
                otherscore=0;
            }
            current_added.clear();
        }
        return;
    }
private:
    int num_neighbors = 10;
    int weight_scheme = RANKED;
    bool shared = true;
    bool transposed = false;
};

};

#endif
