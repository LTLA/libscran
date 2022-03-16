#ifndef SCRAN_SCRAN_HPP
#define SCRAN_SCRAN_HPP

#include "aggregation/AggregateAcrossCells.hpp"
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
#include "quality_control/IsOutlier.hpp"
#include "quality_control/PerCellQCFilters.hpp"
#include "quality_control/PerCellQCMetrics.hpp"
#include "utils/average_vectors.hpp"
#include "utils/subset_vector.hpp"
#include "utils/vector_to_pointers.hpp"

#endif
