#ifndef SCRAN_BUILDSNNGRAPH_HPP
#define SCRAN_BUILDSNNGRAPH_HPP

#include <vector>
#include <algorithm>

#include "knncolle/knncolle.hpp"

namespace scran {

class BuildSNNGraph {
private:
    knncolle::DispatchAlgorithm algo = knncolle::VPTREE;
    int num_neighbors = 10;
    int weight_scheme = RANKED;

public:
    BuildSNNGraph& set_neighbors(int k = 10) {
        num_neighbors = k;
        return *this;
    }

    BuildSNNGraph& set_algorithm(knncolle::DispatchAlgorithm a = knncolle::VPTREE) {
        algo = a;
        return *this;
    }

    enum Scheme { RANKED, NUMBER, JACCARD };

    BuildSNNGraph& set_weighting_scheme(Scheme w = RANKED) {
        weight_scheme = w;
        return *this;
    }

public:
    typedef std::tuple<size_t, size_t, double> WeightedEdge;

    std::deque<WeightedEdge> run(size_t ndims, size_t ncells, const double* mat) const {
        std::deque<WeightedEdge> store;

        typedef knncolle::Base<> knnbase;
        std::shared_ptr<knnbase> ptr;
        if (algo == knncolle::VPTREE) {
            ptr = std::shared_ptr<knnbase>(new knncolle::VpTreeEuclidean<>(ndims, ncells, mat));
        } else if (algo == knncolle::ANNOY) {
            ptr = std::shared_ptr<knnbase>(new knncolle::AnnoyEuclidean<>(ndims, ncells, mat));
        } else {
            throw std::runtime_error("only Annoy and VP-tree searches are supported");
        }

        // Collecting neighbors.
        std::vector<std::vector<int> > indices(ncells);
        std::vector<std::vector<std::pair<int, size_t> > > hosts(ncells);

        for (size_t i = 0; i < ncells; ++i) {
            auto neighbors = ptr->find_nearest_neighbors(i, num_neighbors);
            auto& current = indices[i];

            hosts[i].push_back(std::make_pair(0, i)); // each point is its own 0-th nearest neighbor
            int counter = 1;
            for (auto x : neighbors) {
                current.push_back(x.first);
                hosts[x.first].push_back(std::make_pair(counter, i));
                ++counter;
            }
        }

        // Constructing the shared neighbor graph.
        std::vector<size_t> current_added;
        current_added.reserve(ncells);
        std::vector<int> current_score(ncells);

        for (size_t j = 0; j < ncells; ++j) {
            const auto& current_neighbors = indices[j];

            for (int i = 0; i <= current_neighbors.size(); ++i) {
                // First iteration treats 'j' as the zero-th neighbor.
                // Remaining iterations go through the neighbors of 'j'.
                const int cur_neighbor = (i==0 ? j : current_neighbors[i-1]);

                // Going through all observations 'h' for which 'cur_neighbor'
                // is a nearest neighbor, a.k.a., 'cur_neighbor' is a shared
                // neighbor of both 'h' and 'j'.
                for (auto& h : hosts[cur_neighbor]) {
                    const auto& othernode = h.second;

                    if (othernode < j) { // avoid duplicates from symmetry in the SNN calculations.
                        int& existing_other = current_score[othernode];
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
           
            // Converting to edges.
            for (auto othernode : current_added) {
                int& otherscore=current_score[othernode];
                double finalscore;
                if (weight_scheme == RANKED) {
                    finalscore = static_cast<double>(current_neighbors.size()) - 0.5 * static_cast<double>(otherscore);
                } else {
                    finalscore = otherscore;
                    if (weight_scheme == JACCARD) {
                        finalscore = finalscore / (2 * (current_neighbors.size() + 1) - finalscore);
                    }
                }

                // Ensuring that an edge with a positive weight is always reported.
                store.push_back(WeightedEdge(j, othernode, std::max(finalscore, 1e-6)));

                // Resetting all those added to zero.
                otherscore=0;
            }
            current_added.clear();
        }

        return store;
    }
};

};

#endif
