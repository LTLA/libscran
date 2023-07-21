#ifndef SCRAN_SCRAN_HPP
#define SCRAN_SCRAN_HPP

#include "utils/macros.hpp"

#include "aggregation/AggregateAcrossCells.hpp"
#include "aggregation/DownsampleByNeighbors.hpp"

#include "feature_set_enrichment/HypergeometricTail.hpp"
#include "feature_set_enrichment/ScoreFeatureSet.hpp"

#include "clustering/BuildSnnGraph.hpp"
#if __has_include("igraph.h")
#include "clustering/ClusterSnnGraph.hpp"
#endif

#include "differential_analysis/ScoreMarkers.hpp"

#include "feature_selection/ChooseHvgs.hpp"
#include "feature_selection/FitVarianceTrend.hpp"
#include "feature_selection/ModelGeneVariances.hpp"

#include "dimensionality_reduction/SimplePca.hpp"
#include "dimensionality_reduction/ResidualPca.hpp"
#include "dimensionality_reduction/MultiBatchPca.hpp"

#include "normalization/LogNormCounts.hpp"
#include "normalization/GroupedSizeFactors.hpp"
#include "normalization/MedianSizeFactors.hpp"
#include "normalization/CenterSizeFactors.hpp"
#include "normalization/ChoosePseudoCount.hpp"

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
 * @file scran.hpp
 * @brief Umbrella header for all **libscran** functionality.
 */

/**
 * @namespace scran
 * @brief Functions for single-cell RNA-seq analyses.
 */
namespace scran {}

#endif
