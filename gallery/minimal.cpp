#include "scran/scran.hpp"
#include "tatami/ext/MatrixMarket.hpp"
#include <iostream>

int main(int argc, char * argv[]) {
    if (argc < 2) {
        std::cerr << "Hey, I need a MatrixMarket file!" << std::endl;
        return 1;
    }

    // Loading the data from an (unzipped) MatrixMarket file.
    auto mat = tatami::MatrixMarket::load_sparse_matrix(argv[1]);

    // Filtering out low-quality cells. 
    auto qc_res = scran::PerCellQCMetrics().run(mat.get(), { /* mito subset definitions go here */ });
    auto qc_filters = scran::PerCellQCFilters().run(qc_res);
    auto filtered = scran::FilterCells().run(mat, qc_filters.overall_filter.data());

    // Computing log-normalized expression values, re-using the total count from the QC step.
    auto size_factors = scran::subset_vector<false>(qc_res.sums, qc_filters.overall_filter.data());
    auto normalized = scran::LogNormCounts().run(filtered, std::move(size_factors));

    // Identifying highly variable genes.
    auto var_res = scran::ModelGeneVar().run(normalized.get());
    auto keep = scran::ChooseHVGs().run(var_res.residuals[0].size(), var_res.residuals[0].data());

    // Performing a PCA on the HVGs.
    int npcs = 20;
    auto pca_res = scran::RunPCA().set_rank(npcs).run(normalized.get(), keep.data());
    pca_res.pcs.adjointInPlace(); 

    // Performing clustering.
    auto clust_res = scran::ClusterSNNGraphMultiLevel().run(npcs, pca_res.pcs.cols(), pca_res.pcs.data());
    const auto& best_clustering = clust_res.membership[clust_res.max];

    int nclusters = *std::max_element(best_clustering.begin(), best_clustering.end()) + 1;
    std::cout << "Detected " << nclusters << " clusters in '" << argv[1] << "'" << std::endl;

    // Throw in some marker detection.
    auto marker_res = scran::ScoreMarkers().run(normalized.get(), best_clustering.data());

    return 0;
}
