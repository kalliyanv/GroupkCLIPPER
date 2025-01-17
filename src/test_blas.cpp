// #include <iostream>
// #include <xtensor/xarray.hpp>
// #include <xtensor-blas/xlinalg.hpp>

// int main() {
//     xt::xarray<double> a = {{1, 2}, {3, 4}};
//     xt::xarray<double> b = {{5, 6}, {7, 8}};
//     xt::xarray<double> c = xt::linalg::dot(a, b);
//     std::cout << c << std::endl;
//     return 0;
// }

#include <iostream>
#include <cblas.h>

int main() {
    std::cout << "BLAS Implementation: " << cblas_get_config() << std::endl;
    return 0;
}