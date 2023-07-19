# A C++ library for single-cell data analysis

![Unit tests](https://github.com/LTLA/libscran/actions/workflows/run-tests.yaml/badge.svg)
![Documentation](https://github.com/LTLA/libscran/actions/workflows/doxygenate.yaml/badge.svg)
[![Codecov](https://codecov.io/gh/LTLA/libscran/branch/master/graph/badge.svg?token=CPER7Q7FEH)](https://codecov.io/gh/LTLA/libscran)

## Overview 

The **libscran** library takes core parts of the [**scran**](https://github.com/MarioniLab/scran) Bioconductor package (as well as other useful bits from other packages) and implements them in C++.
The idea is to provide a light-weight library that can be easily embedded into other applications without including the entire R/Bioconductor runtime.
For example, we can compile [**libscran** to WebAssembly](https://github.com/jkanche/scran.js) to perform single-cell analyses in the browser;
or we can wrap [**libscran** into an R package](https://github.com/LTLA/scran.chan) for a minimal-dependency version of the basic Bioconductor single-cell analysis stack.
The library itself is compatible with any CMake-based build system and can be turned into a fully header-only library for easy deployment.

## Quick start

The example below demonstrates how to use **libscran** to run a standard analysis of single-cell RNA-seq data.

```cpp
#include "scran/scran.hpp"

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

// Throw in some marker detection.
auto marker_res = scran::ScoreMarkers().run(normalized.get(), best_clustering.data());
```

Each class represents a step in the analysis and has tunable parameters, e.g., `RunPCA::set_rank` to set the number of PCs.
See the [reference documentation](https://ltla.github.io/libscran/) for more details.

## Available functions

Most of the functions are motivated by the theory in the [**Orchestrating single-cell analysis with Bioconductor** book](https://bioconductor.org/books/release/OSCA/).

**Identification and filtering of low-quality cells** are performed using an outlier-based approach.
The `PerCellQCMetrics` class will compute common QC metrics, 
the `PerCellQCFilters` class will identify filtering thresholds from the distribution of such metrics,
and the `FilterCells` class will apply those filters to the count matrix.

**Log-transformed normalized expression values** are computed from the count matrix,
using size factors derived from the library size.
This is performed using the `LogNormCounts` class.

**Variance modelling and selection of highly variable genes** is performed on the log-expression values.
The `ModelGeneVar` class will fit a mean-dependent trend to the variances across genes,
while the `ChooseHVGs` class will choose the top set of HVGs based on the residuals from the trend.

**Principal component analysis** is used to compress and denoise the data based on the first few PCs.
The `RunPCA` class will use an approximate PCA algorithm to efficiently compute the top PCs from the HVG-subsetted matrix.
Alternatively, the `BlockedPCA` and `MultiBatchPCA` classes can be used when dealing with multiple batches.

**Clustering of cells** is performed using the per-cell PC scores.
We provide several flavors of graph-based clustering from a shared-nearest neighbor graph,
using community detection algorithms such as multi-level (`ClusterSnnGraphMultiLevel`), Leiden (`ClusterSnnGraphLeiden`) or Walktrap clustering (`ClusterSnnGraphWalktrap`).
Developers can also easily apply other algorithms, e.g., [k-means](https://github.com/LTLA/CppKmeans).

**Per-cluster marker detection** is performed based on pairwise comparisons between clusters.
The `ScoreMarkers` class will aggregate the set of pairwise comparisons into a single suite of summary statistics for each cluster.
Users can then rank by a statistic of interest to obtain a marker listing for each cluster.

The output of PCA is also directly compatible with [UMAP](https://github.com/LTLA/umappp) and [t-SNE](https://github.com/LTLA/qdtsne) C++ implementations.
Readers are referred to the documentation for those libraries for more details.

## Example analysis

Compile the [`minimal.cpp`](https://github.com/LTLA/libscran/blob/master/gallery/minimal.cpp) example by running the following commands at the root of the **libscran** directory:

```sh
cmake -S . -B example -DBUILD_TESTING=OFF -DBUILD_GALLERY=ON
cmake --build example --target minimal
```

Download and decompress a Matrix Market file containing a scRNA-seq count matrix:

```sh
mkdir example/data
wget https://cf.10xgenomics.com/samples/cell-exp/2.1.0/pbmc4k/pbmc4k_filtered_gene_bc_matrices.tar.gz -P example/data
tar -xvf example/data/pbmc4k_filtered_gene_bc_matrices.tar.gz --directory example/data
```

Run the minimal pipeline:

```sh
time example/gallery/minimal example/data/filtered_gene_bc_matrices/GRCh38/matrix.mtx

## Detected 11 clusters in 'example/data/filtered_gene_bc_matrices/GRCh38/matrix.mtx'
## Sizes are 937, 534, 1018, 37, 135, 384, 537, 225, 190, 123, 191
##
## real	0m2.340s
## user	0m9.613s
## sys	0m0.104s
```

## Building projects 

If you're using CMake, you just need to add something like this to your `CMakeLists.txt`:

```cmake
include(FetchContent)

FetchContent_Declare(
  libscran
  GIT_REPOSITORY https://github.com/LTLA/libscran
  GIT_TAG master # or any version of interest 
)

FetchContent_MakeAvailable(libscran)
```

Then you can link to **libscran** to make the headers available during compilation:

```cmake
# For executables:
target_link_libraries(myexe libscran)

# For libaries
target_link_libraries(mylib INTERFACE libscran)
```

Developers are responsible for linking to the [**igraph**](https://igraph.org) C library themselves, either with `find_package()` or `FetchContent`.
We expect **igraph** versions from the 0.10 series - see [`tests/CMakeLists.txt`](tests/CMakeLists.txt) for the specific version being tested.
