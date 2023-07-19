#include "scran/scran.hpp"
#include "tatami_mtx/tatami_mtx.hpp"
#include <iostream>

int main(int argc, char * argv[]) {
    if (argc < 2) {
        std::cerr << "Hey, I need a MatrixMarket file!" << std::endl;
        return 1;
    }

    // Loading the data from an unzipped MatrixMarket file.
    auto mat = tatami_mtx::load_matrix_from_file<false, double, int>(argv[1]);

    // Filtering out low-quality cells. 
    auto qc_res = scran::PerCellRnaQcMetrics().run(mat.get(), { /* mito subset definitions go here */ });
    auto qc_filters = scran::SuggestRnaQcFilters().run(qc_res);
    auto low_quality = qc_filters.filter(qc_res);
    auto filtered = scran::FilterCells().run(mat, low_quality.data());

    // Computing log-normalized expression values, re-using the total count from the QC step.
    auto size_factors = scran::subset_vector<false>(qc_res.sums, low_quality.data());
    auto normalized = scran::LogNormCounts().run(filtered, std::move(size_factors));

    // Identifying highly variable genes.
    auto var_res = scran::ModelGeneVariances().run(normalized.get());
    auto keep = scran::ChooseHvgs().run(var_res.residuals.size(), var_res.residuals.data());

    // Performing a PCA on the HVGs.
    int npcs = 20;
    auto pca_res = scran::SimplePca().set_rank(npcs).run(normalized.get(), keep.data());

    // Performing clustering.
    auto graph = scran::BuildSnnGraph().run(npcs, pca_res.pcs.cols(), pca_res.pcs.data());
    auto clust_res = scran::ClusterSnnGraphMultiLevel().run(graph);
    const auto& best_clustering = clust_res.membership[clust_res.max];

    std::vector<int> counters;
    for (auto x : best_clustering) {
        if (x >= counters.size()) {
            counters.resize(x + 1);
        }
        ++(counters[x]);
    }
    std::cout << "Detected " << counters.size() << " clusters in '" << argv[1] << "'\nSizes are ";
    for (size_t i = 0; i < counters.size(); ++i) {
        if (i) {
            std::cout << ", ";
        }
        std::cout << counters[i];
    }
    std::cout << std::endl;

    // Throw in some marker detection.
    auto marker_res = scran::ScoreMarkers().run(normalized.get(), best_clustering.data());

    return 0;
}
