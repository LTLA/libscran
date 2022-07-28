#ifndef SCRAN_KMEANSCLUSTERING_HPP
#define SCRAN_KMEANSCLUSTERING_HPP

#include "../utils/macros.hpp"

#include "kmeans/Kmeans.hpp"

/**
 * @file ClusterKmeans.hpp
 *
 * @brief Identify clusters of cells with k-means.
 */

namespace scran {

/**
 * Provides the `kmeans::Kmeans` class under the **scran** namespace.
 * This is mostly for the sake of having a consistent naming scheme - 
 * see the [**kmeans** documentation](https://ltla.github.io/CppKmeans/) for specifics.
 *
 * Note that applications using a custom parallelization scheme will need to set the `KMEANS_CUSTOM_PARALLEL` macro;
 * the setting of the `SCRAN_CUSTOM_PARALLEL` macro will not be propagated automatically.
 */
typedef kmeans::Kmeans<> ClusterKmeans;

}

#endif
