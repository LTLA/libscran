#ifndef SCRAN_KMEANSCLUSTERING_HPP
#define SCRAN_KMEANSCLUSTERING_HPP

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
 */
typedef kmeans::Kmeans<> ClusterKmeans;

}

#endif
