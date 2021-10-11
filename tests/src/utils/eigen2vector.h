#ifndef EIGEN2VECTOR_H
#define EIGEN2VECTOR_H

#include <vector>
#include "Eigen/Dense"

inline std::vector<double> eigen2vector(const Eigen::MatrixXd& input) {
    auto ptr = input.data();
    return std::vector<double>(ptr, ptr + input.rows() * input.cols());
}

inline std::vector<double> eigen2vector(const Eigen::VectorXd& input) {
    return std::vector<double>(input.begin(), input.end());
}

#endif
