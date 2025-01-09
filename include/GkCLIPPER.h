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

#include "gen_max_clique/gmc.hpp"

using namespace std;

class GkCLIPPER
{
    public:

        size_t n;   // Number nodes in the graph
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
        double tensor_contract(const xt::xarray<double>& tensor, const xt::xarray<double>& x, int k);
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


    private:

};