#include "scran/scran.hpp"
#include "eminem/eminem.hpp"
#include "tatami/tatami.hpp"
#include <iostream>

int main(int argc, char * argv[]) {
    if (argc < 2) {
        std::cerr << "Hey, I need a MatrixMarket file!" << std::endl;
        return 1;
    }

    // Loading the data from an unzipped MatrixMarket file.
    // TODO: move to a tatami extension.
    std::shared_ptr<tatami::NumericMatrix> mat;

    {
        eminem::TextFileParser parser(argv[1]);
        parser.scan_preamble();
        int NR = parser.get_nrows(), NC = parser.get_ncols();
        std::vector<int> values(parser.get_nlines()), rows(parser.get_nlines()), columns(parser.get_nlines());

        if (parser.get_banner().field != eminem::Field::INTEGER) {
            std::cerr << "Expected an integer Matrix Market file" << std::endl;
            return 1;
        }

        parser.scan_integer([&](int r, int c, int v) -> void {
            values.push_back(v);
            rows.push_back(r - 1);
            columns.push_back(c - 1);
        });

        auto ptr = tatami::compress_sparse_triplets<false>(NR, NC, values, rows, columns);
        mat.reset(new tatami::CompressedSparseColumnMatrix<double, int, decltype(values)>(NR, NC, std::move(values), std::move(rows), std::move(ptr)));
    }

    // Filtering out low-quality cells. 
    auto qc_res = scran::PerCellRnaQcMetrics().run(mat.get(), { /* mito subset definitions go here */ });
    auto qc_filters = scran::SuggestRnaQcFilters().run(qc_res);
    auto low_quality = qc_filters.filter(qc_res);
    auto filtered = scran::FilterCells().run(mat, low_quality.data());

    // Computing log-normalized expression values, re-using the total count from the QC step.
    auto size_factors = scran::subset_vector<false>(qc_res.sums, low_quality.data());
    auto normalized = scran::LogNormCounts().run(filtered, std::move(size_factors));

    // Identifying highly variable genes.
    auto var_res = scran::ModelGeneVar().run(normalized.get());
    auto keep = scran::ChooseHVGs().run(var_res.residuals[0].size(), var_res.residuals[0].data());

    // Performing a PCA on the HVGs.
    int npcs = 20;
    auto pca_res = scran::RunPCA().set_rank(npcs).run(normalized.get(), keep.data());

    // Performing clustering.
    auto graph = scran::BuildSNNGraph().run(npcs, pca_res.pcs.cols(), pca_res.pcs.data());
    auto clust_res = scran::ClusterSNNGraphMultiLevel().run(graph);
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
