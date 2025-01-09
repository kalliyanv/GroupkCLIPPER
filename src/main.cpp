#include <xtensor/xarray.hpp>
// #include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xrandom.hpp> 
#include <xtensor/xmath.hpp>
#include <xtensor/xio.hpp>
#include <iostream>

#include <Eigen/Dense> 
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp> 

#include "GkCLIPPER.h"


Eigen::VectorXd xtensor_to_eigen(const xt::xarray<double>& xtensor_array) {
    // Ensure the xtensor array is 1D
    if (xtensor_array.dimension() != 1) {
        throw std::invalid_argument("xt::xarray must be one-dimensional for conversion to Eigen::VectorXd.");
    }

    // Create an Eigen::VectorXd with the same size
    Eigen::VectorXd eigen_vector(xtensor_array.size());

    // Copy data from the xtensor array to the Eigen vector
    for (std::size_t i = 0; i < xtensor_array.size(); ++i) {
        eigen_vector(i) = xtensor_array(i);
    }

    return eigen_vector;
}

Eigen::MatrixXd xtensor_to_eigen_matrix(const xt::xarray<double>& xtensor_array) {
    // Ensure the xtensor array is 2D
    if (xtensor_array.dimension() != 2) {
        throw std::invalid_argument("xt::xarray must be two-dimensional for conversion to Eigen::MatrixXd.");
    }

    // Get the number of rows and columns from xtensor
    std::size_t rows = xtensor_array.shape()[0];
    std::size_t cols = xtensor_array.shape()[1];

    // Create an Eigen::MatrixXd with the same dimensions
    Eigen::MatrixXd eigen_matrix(rows, cols);

    // Copy data from the xtensor array to the Eigen matrix
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            eigen_matrix(i, j) = xtensor_array(i, j);
        }
    }

    return eigen_matrix;
}

// xt::xarray<double> get_grad(const xt::xarray<double>& A, const xt::xarray<double>& X) {
//     // std::cout << "Shapes " << A.shape()[0] << " " << X.size() << std::endl;
//     // // Ensure dimensions are compatible
//     // if (A.dimension() < 2 || A.shape()[0] != X.size()) {
//     //     throw std::invalid_argument("Incompatible dimensions for tensor and vector.");
//     // }

//     size_t k = A.dimension();           // Tensor order
//     size_t n = pow(A.size(), 1.0/k);    // Tensor dimension
//     xt::xarray<double> result = xt::xarray<int>::from_shape({n});
//     xt::xarray<double> A_flat = xt::flatten(A);

//     // Each entry in resulting vector
//     for (int i = 0; i < n; i++) {
//         // Iterate through each order
//         for (int k_it = 1; k_it < k; k_it++) {
//             // Iterate through each item in this dimension
//             for (int n_it = 0; n_it < n; n_it++) {
//                 result(i) = i;

//             }
//         }
//         // // Iterate over all higher-order indices (i_2, ..., i_k)
//         // std::vector<std::size_t> indices(A.dimension() - 1, 0);
//         // do {
//         //     // Compute the product x_{i_2} * x_{i_3} * ... * x_{i_k}
//         //     double product = 1.0;
//         //     for (std::size_t d = 1; d < A.dimension(); ++d) {
//         //         product *= X[indices[d - 1]];
//         //     }

//         //     // Access the tensor entry a_{i, i_2, ..., i_k}
//         //     std::vector<std::size_t> tensor_indices = {i};
//         //     tensor_indices.insert(tensor_indices.end(), indices.begin(), indices.end());
//         //     result(i) += A(tensor_indices) * product;

//         //     // Increment indices
//         //     for (std::size_t d = indices.size(); d-- > 0;) {
//         //         if (++indices[d] < shape[d + 1]) break;
//         //         indices[d] = 0;
//         //     }
//         // } while (indices[0] < shape[1]); // Stop when we've exhausted all combinations

//     }

//     return result;


// }

int main() {
    // Get ajacency tensor
    // xt::xarray<int> M = {
    //     {
    //         {1, 0, 0, 0}, 
    //         {0, 0, 1, 0}, 
    //         {0, 1, 0, 1}, 
    //         {0, 0, 1, 0}
    //     },
    //     {
    //         {0, 0, 1, 0}, 
    //         {0, 1, 0, 0}, 
    //         {1, 0, 0, 0}, 
    //         {0, 0, 0, 0}
    //     },
    //     {
    //         {0, 1, 0, 1}, 
    //         {1, 0, 0, 0}, 
    //         {0, 0, 1, 0}, 
    //         {1, 0, 0, 0}
    //     },
    //     {
    //         {0, 0, 1, 0}, 
    //         {0, 0, 0, 0}, 
    //         {1, 0, 0, 0}, 
    //         {0, 0, 0, 1}
    //     }
    // };

    // xt::xarray<int> M = {
    //                         {0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0},
    //                         {0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0},
    //                         {0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    //                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0},
    //                         {0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0},
    //                         {0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0},
    //                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //                         {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
    //                     };

    // xt::xarray<int> M = {
    //                         {2, -1, 0},
    //                         {-1, 2, -1},
    //                         {0, -1, 2}
    //                     };

    xt::xarray<double> M = {{{{-4.0,  0.0,  0.},
                            {0.,  0.,  0.},
                            {0.,  0.,  0.}},

                            {{0.,  0.,  0.},
                            {0.,  0.,  0.},
                            {0.,  0.,  0.}},

                            {{ 0.,  0.,  0.},
                            { 0.,  0.,  0.},
                            { 0.,  0.,  1.}}},


                            {{{ 0.,  0.,  0.},
                            { 0.,  0.,  0.},
                            { 0.,  0.,  0.}},

                            {{ 0.,  0.,  0.},
                            { 0., -4.,  0.},
                            { 0.,  0.,  0.}},

                            {{ 0.,  0.,  0.},
                            { 0.,  0.,  0.},
                            { 0.,  0.,  0.}}},


                            {{{ 0.,  0.,  0.},
                            { 0.,  0.,  0.},
                            { 0.,  0.,  1.}},

                            {{ 0.,  0.,  0.},
                            { 0.,  0.,  0.},
                            { 0.,  0.,  0.}},

                            {{ 0.,  0.,  1.},
                            { 0.,  0.,  0.},
                            { 1.,  0., -4.}}}};

    // Print the tensor
    // std::cout << "M Tensor:" << std::endl;
    // std::cout << M << std::endl;

    // Init solution randomly. TODO add option to input init vector
    size_t n = 3;
    size_t k = M.dimension();
    // std::cout << "k " << k << std::endl;

    GkCLIPPER clipper(n, k, M);
    std::cout << "attempting to solve" << std::endl;
    clipper.solve();

    // Initialize memory. These need to all be tensors, with last dimension being 1
    xt::xarray<double> gradF = xt::xarray<int>::from_shape({n, 1});
    xt::xarray<double> gradFnew = xt::xarray<int>::from_shape({n, 1});
    xt::xarray<double> u = xt::xarray<int>::from_shape({n, 1});
    xt::xarray<double> unew = xt::xarray<int>::from_shape({n, 1});
    xt::xarray<double> Mu = xt::xarray<int>::from_shape({n, k});
    xt::xarray<double> num = xt::xarray<int>::from_shape({n, k});
    xt::xarray<double> den = xt::xarray<int>::from_shape({n, k});

    // Normal matrix version

    // Compute eigenvalues and eigenvectors using EigenSolver
    // Eigen::EigenSolver<Eigen::Matrix3d> solver(M_e);
    // Eigen::Vector3cd eigenvalues = solver.eigenvalues();
    // std::cout << "Eigenvalues of the second order matrix:\n" << eigenvalues << std::endl;  // Real and imaginary parts

    Eigen::VectorXd gradF_e(n);
    Eigen::VectorXd gradFnew_e(n);
    Eigen::VectorXd u_e(n);
    Eigen::VectorXd unew_e(n);
    Eigen::VectorXd Mu_e(n);
    Eigen::VectorXd num_e(n);
    Eigen::VectorXd den_e(n);


    xt::xarray<double> u0 =  {-0.23197069, -0.78583024,  0.40824829};//xt::random::rand<double>({n}, 0.0, 1.0); //{1.0, 2.0, 3.0, 4.0}; // xt::ones<double>({n}); //{1.0, 2.0, 3.0}; //
    std::cout << "u0" << u0 << " " << u0.shape()[1] << std::endl;

    u = u0 / clipper.calculate_k_norm(u0, k);

    // Gradient ascent
    double F = 0;       // objective value
    double alpha = 0.01;   // step size scaling

    // u_e = xtensor_to_eigen(u);

    int max_iters = 10;

    // Eigen::MatrixXd M_e = xtensor_to_eigen_matrix(M);
    // Eigen::MatrixXd M_full_e = M_e.selfadjointView<Eigen::Upper>();
    // std::cout << M_e << std::endl;
    // gradF_e = M_e*u_e;

    gradF = clipper.get_grad(M, u);
    std::cout << "First grad " << gradF << std::endl;

    // std::cout << "M " << M << std::endl;
    // std::cout << "M_e " << M_e << std::endl;
    // std::cout << "u " << u << std::endl;
    // std::cout << "u_e " << u_e << std::endl;
    // std::cout << "Gradient F " << gradF << std::endl;
    // std::cout << "Gradient F_e " << gradF_e << std::endl;

    for (int i=0; i < max_iters; i++) {
        u = u + alpha * gradF;
        u = u / clipper.calculate_k_norm(u, k);
        F = xt::sum(gradF * u)();
        std::cout << "Current e-value " << F << std::endl;
    }


    // try {
    //     double k_norm = calculate_k_norm(u0, k);
    //     std::cout << "The " << k << "-norm of the vector is: " << k_norm << std::endl;
    // } catch (const std::invalid_argument& e) {
    //     std::cerr << e.what() << std::endl;
    // }

    // try {
    //     // Convert xt::xarray<double> to Eigen::VectorXd
    //     Eigen::VectorXd eigen_vector = xtensor_to_eigen(u0);

    //     // Print the Eigen::VectorXd
    //     // std::cout << "Eigen::VectorXd:\n" << eigen_vector << std::endl;
    //     std::cout << "The " << 2 << "-norm of the Eigen vector is: " << eigen_vector.norm() << std::endl;
    //     eigen_vector /= eigen_vector.norm();
    //     std::cout << "Normalized Random Eigen Vector:\n" << eigen_vector << std::endl;

    // } catch (const std::exception& e) {
    //     std::cerr << e.what() << std::endl;
    // }

    // // // one step of power method to have a good scaling of u
    // // if (params_.rescale_u0) {
    // //     u = M_.selfadjointView<Eigen::Upper>() * u0 + u0;
    // // } else {
    // //     u = u0;
    // // }

    // // TODO Enforce clique constraints using penalty parameter
    // // double d = 0;

    return 0;
}
