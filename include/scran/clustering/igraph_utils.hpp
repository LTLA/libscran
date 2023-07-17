#ifndef SCRAN_IGRAPH_UTILS_HPP
#define SCRAN_IGRAPH_UTILS_HPP

#include "../utils/macros.hpp"

#include "igraph.h"

/**
 * @file igraph_utils.hpp
 *
 * @brief Utilities for manipulating **igraph** data structures.
 */

namespace scran {

/**
 * @namespace scran::igraph
 * @brief Utilities for working with **igraph** in **libscran**.
 */
namespace igraph {

/**
 * @cond
 */
struct RNGScope {
    RNGScope(int seed) : previous(igraph_rng_default()) {
        if (igraph_rng_init(&rng, &igraph_rngtype_mt19937)) {
            throw std::runtime_error("failed to initialize an instance of igraph's RNG");
        }

        if (igraph_rng_seed(&rng, seed)) {
            igraph_rng_destroy(&rng);
            throw std::runtime_error("failed to set the seed on igraph's RNG");
        }

        igraph_rng_set_default(&rng);
    }

    // Just deleting the methods here, because the RNGScope
    // is strictly internal and we don't do any of these.
    RNGScope(const RNGScope& other) = delete;
    RNGScope& operator=(const RNGScope& other) = delete;
    RNGScope(RNGScope&& other) = delete;
    RNGScope& operator=(RNGScope&& other) = delete;

    ~RNGScope() {
        if (active) {
            igraph_rng_set_default(previous);
            igraph_rng_destroy(&rng);
        }
    }

    igraph_rng_t* previous;
    igraph_rng_t rng;
    bool active = true;
};

/*****************************************************
 *****************************************************/

template<class Overlord>
struct Vector_ {
private:
    static void try_copy(typename Overlord::Vector& dest, const typename Overlord::Vector& source, bool source_active) {
        if (source_active) {
            if (Overlord::copy(dest, source)) {
                throw std::runtime_error("failed to copy igraph vector of size " + std::to_string(Overlord::size(source)));
            }
        }
    }

    static void try_destroy(typename Overlord::Vector& vector, bool active) {
        if (active) {
            Overlord::destroy(vector);
        }
    }

public:
    Vector_(size_t size = 0) {
        if (Overlord::initialize(vector, size)) {
            throw std::runtime_error("failed to initialize igraph vector of size " + std::to_string(size));
        }
    }
    
    // Just deleting the methods here, because the Vector_ 
    // is strictly internal and we don't do any of these.
    Vector_(const Vector_& other) = delete;
    Vector_& operator=(const Vector_& other) = delete;
    Vector_(Vector_&& other) = delete;
    Vector_& operator=(Vector_&& other) = delete;

    ~Vector_() {
        try_destroy(vector, active);
    }

    typename Overlord::Vector vector;
    bool active = true;
};

struct IntegerVectorOverlord {
    typedef igraph_vector_int_t Vector;

    static igraph_error_t copy(igraph_vector_int_t& dest, const igraph_vector_int_t& source) {
        return igraph_vector_int_init_copy(&dest, &source);
    }

    static void destroy(igraph_vector_int_t& x) {
        return igraph_vector_int_destroy(&x);
    }

    static igraph_integer_t size(igraph_vector_int_t& x) {
        return igraph_vector_int_size(&x);
    }

    static igraph_error_t initialize(igraph_vector_int_t& x, size_t size) {
        return igraph_vector_int_init(&x, size);
    }
};

using IntegerVector = Vector_<IntegerVectorOverlord>;

struct RealVectorOverlord {
    typedef igraph_vector_t Vector;

    static igraph_error_t copy(igraph_vector_t& dest, const igraph_vector_t& source) {
        return igraph_vector_init_copy(&dest, &source);
    }

    static void destroy(igraph_vector_t& x) {
        return igraph_vector_destroy(&x);
    }

    static igraph_integer_t size(igraph_vector_t& x) {
        return igraph_vector_size(&x);
    }

    static igraph_error_t initialize(igraph_vector_t& x, size_t size) {
        return igraph_vector_init(&x, size);
    }
};

using RealVector = Vector_<RealVectorOverlord>;

/*****************************************************
 *****************************************************/

struct IntegerMatrix {
    IntegerMatrix(size_t nrows = 0, size_t ncols = 0) {
        if (igraph_matrix_int_init(&matrix, nrows, ncols)) {
            throw std::runtime_error("failed to initialize igraph " + std::to_string(nrows) + "x" + std::to_string(ncols) + " matrix");
        }
    }

    // Just deleting the methods here, because the Matrix
    // is strictly internal and we don't do any of these.
    IntegerMatrix(const IntegerMatrix& other) = delete;
    IntegerMatrix& operator=(const IntegerMatrix& other) = delete;
    IntegerMatrix(IntegerMatrix&& other) = delete;
    IntegerMatrix& operator=(IntegerMatrix&& other) = delete;

    ~IntegerMatrix() {
        if (active) {
            igraph_matrix_int_destroy(&matrix);
        }
    }

    igraph_matrix_int_t matrix;
    bool active = true;
};
/**
 * @endcond
 */

/*****************************************************
 *****************************************************/

/**
 * @brief Wrapper around the `igraph_t` class from **igraph**.
 */
struct Graph {
private:
    static void try_copy(igraph_t& dest, const igraph_t& source, bool source_active) {
        if (source_active) {
            if (igraph_copy(&dest, &source)) {
                throw std::runtime_error("failed to copy igraph's graph");
            }
        }
    }

    static void try_destroy(igraph_t& graph, bool active) {
        if (active) {
            igraph_destroy(&graph);
        }
    }

public:
    /**
     * @cond
     */
    Graph() : active(false) {}

    Graph(const IntegerVector& edges, size_t nvertices, bool directed) : Graph(&(edges.vector), nvertices, directed) {} 

    Graph(const igraph_vector_int_t* edges, size_t nvertices, bool directed) { 
        if (igraph_create(&graph, edges, nvertices, directed)) {
            throw std::runtime_error("failed to initialize igraph's graph object"); 
        }
    }

    Graph(const Graph& other) : active(other.active) {
        try_copy(graph, other.graph, other.active);
    }

    Graph& operator=(const Graph& other) {
        if (this != &other) {
            try_destroy(graph, active);
            try_copy(graph, other.graph, other.active);
            active = other.active;
        }
        return *this;
    }

    // See https://docs.microsoft.com/en-us/cpp/cpp/move-constructors-and-move-assignment-operators-cpp?view=msvc-170
    Graph(Graph&& other) : graph(std::move(other.graph)), active(other.active) {
        other.active = false;
    }

    Graph& operator=(Graph&& other) {
        if (this != &other) {
            try_destroy(graph, active);
            graph = std::move(other.graph);
            active = other.active;
            other.active = false;
        }
        return *this;
    }

    ~Graph() {
        try_destroy(graph, active);
    }

    bool active = true;

    igraph_t graph;
    /**
     * @endcond
     */

    /**
     * Get the graph.
     * Users should not pass this pointer to `igraph_destroy`; the `Graph` destructor will handle the freeing automatically.
     *
     * @return Pointer to an **igraph** graph.
     * Nodes are cells with edges being formed between its nearest neighbors.
     */
    const igraph_t* get_graph() const {
        return &graph;
    }
};

}

}

#endif
