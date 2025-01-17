#include <xtensor/xarray.hpp>
// #include <xtensor-blas/xlinalg.hpp>
#include <xtensor/xrandom.hpp> 
#include <xtensor/xsort.hpp>
#include <xtensor/xmath.hpp>
#include <xtensor/xview.hpp>
#include <xtensor/xio.hpp> 

#include <xtensor-blas/xlinalg.hpp>

#include <iostream>

#include <Eigen/Dense> 
#include <vector>
using namespace Eigen;

#include <Fastor/Fastor.h>
using namespace Fastor;
enum {I,J,K,L,M,N};

#include <chrono>


#include "gen_max_clique/gmc.hpp"

using namespace std;
using namespace std::chrono;

// General function to create a Fastor tensor
template <typename T, size_t... Dims>
Tensor<T, Dims...> createFastorTensor(const std::vector<T>& values) {
    // Ensure the number of values matches the total size of the tensor
    constexpr size_t total_size = (... * Dims); // Compile-time product of dimensions
    if (values.size() != total_size) {
        throw std::invalid_argument("The number of values does not match the tensor size.");
    }

    // Initialize the tensor
    Tensor<T, Dims...> tensor;
    std::copy(values.begin(), values.end(), tensor.data());
    return tensor;
}


// Helper to dynamically construct dimensions and call the tensor creator
template <typename T>
void createDynamicFastorTensor(const std::vector<size_t>& dimensions, const std::vector<T>& values) {
    // Handle cases dynamically (this would require dynamic Fastor-like abstraction, but Fastor requires static dimensions)
    std::cout << "Fastor requires compile-time known dimensions. Use templates for specific tensor sizes.\n";
}


class GkCLIPPER
{
    public:

        size_t n = 2;   // Number nodes in the graph
        size_t k;   // Number nodes in an edge

        xt::xarray<double> M;
        xt::xarray<double> u = {};
        double rho;

        GMC::KUniformHypergraph<int> graph;

        // Constructor to initialize graph
        GkCLIPPER(size_t n, size_t k, const xt::xarray<double>& M)
        : n(n), k(k), M(M), graph(k) {}
        
        // Math helpers
        double calculate_k_norm(const xt::xarray<double>& x, int k);
        xt::xarray<double> tensor_contract(const xt::xarray<double>& tensor, const xt::xarray<double>& x, int k);
        xt::xarray<double> gradient(const xt::xarray<double>& tensor, const xt::xarray<double>& x, int k, double d);
        double rayleigh_quotient(const xt::xarray<double>& tensor, const xt::xarray<double>& x, int k);
        xt::xarray<double> get_grad(const xt::xarray<double>& tensor, const xt::xarray<double>& vector); // DELETE 

        // Solve
        double get_new_d(const xt::xarray<double>& u, const xt::xarray<double>& C, const xt::xarray<double>& M, double d_prev, double eps = 1e-9);
        void h_eigenvalue_rayleigh(
            xt::xarray<double> x_init = {},
            double d = 0.0,
            int max_iter = 10000, //10000,
            double tol = 1e-6,
            double step_size = 0.01,
            double alpha = 1e-4,
            double beta = 0.5
        );
        void solve();
        xt::xarray<size_t> get_top_n_indices(const xt::xarray<double>& x, size_t omega);

        // Fastor version of everything
        void h_eigenvalue_rayleigh_Fastor(
            std::vector<double> x_init = {},
            double d = 0.0,
            int max_iter = 10000, //10000,
            double tol = 1e-6,
            double step_size = 0.01,
            double alpha = 1e-4,
            double beta = 0.5
        );
    private:

};