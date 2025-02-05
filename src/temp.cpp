#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>

int main() {
    // Define a simple vector
    xt::xarray<double> vec = {1.0, 2.0, 3.0};

    // Compute the L2 norm (Euclidean norm)
    double norm_value = xt::linalg::norm(vec);

    // Print the result
    std::cout << "L2 Norm: " << norm_value << std::endl;

    return 0;
}
