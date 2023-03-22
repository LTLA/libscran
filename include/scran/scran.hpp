#ifndef SCRAN_SCRAN_HPP
#define SCRAN_SCRAN_HPP

#include "utils/macros.hpp"

#include "aggregation/AggregateAcrossCells.hpp"

#include "feature_set_enrichment/HypergeometricTail.hpp"
#include "feature_set_enrichment/ScoreFeatureSet.hpp"

#include "clustering/BuildSNNGraph.hpp"
#include "clustering/ClusterKmeans.hpp"
#include "clustering/ClusterSNNGraph.hpp"
#include "differential_analysis/ScoreMarkers.hpp"
#include "dimensionality_reduction/BlockedPCA.hpp"
#include "dimensionality_reduction/MultiBatchPCA.hpp"
#include "dimensionality_reduction/RunPCA.hpp"
#include "feature_selection/ChooseHVGs.hpp"
#include "feature_selection/FitTrendVar.hpp"
#include "feature_selection/ModelGeneVar.hpp"
#include "normalization/LogNormCounts.hpp"

#include "quality_control/FilterCells.hpp"
#include "quality_control/SuggestRnaQcFilters.hpp"
#include "quality_control/PerCellRnaQcMetrics.hpp"
#include "quality_control/SuggestAdtQcFilters.hpp"
#include "quality_control/PerCellAdtQcMetrics.hpp"
#include "quality_control/SuggestCrisprQcFilters.hpp"
#include "quality_control/PerCellCrisprQcMetrics.hpp"

#include "utils/average_vectors.hpp"
#include "utils/subset_vector.hpp"
#include "utils/vector_to_pointers.hpp"

/**
 * @namespace scran
 * @brief Functions for single-cell RNA-seq analyses.
 */
namespace scran {}

#endif
