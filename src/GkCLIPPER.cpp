#include "GkCLIPPER.h"

// TODO delete and remove from main later
xt::xarray<double> GkCLIPPER::get_grad(const xt::xarray<double>& tensor, const xt::xarray<double>& vector) {
    // Validate input dimensions
    size_t n = vector.size(); // Size of the vector
    size_t k = tensor.dimension(); // Order of the tensor
    // if (tensor.shape() != std::vector<size_t>(k, n)) {
    //     throw std::invalid_argument("Tensor dimensions and vector length do not match.");
    // }

    // Initialize result vector (1D array of size n)
    xt::xarray<double> result = xt::zeros<double>({n});

    // Compute AX^{k-1}
    for (size_t i = 0; i < n; ++i) {
        double contraction = 0.0;

        // Iterate over all combinations of remaining indices (k-1 dimensions)
        for (size_t j2 = 0; j2 < n; ++j2) {
            for (size_t j3 = 0; j3 < n; ++j3) {
                // Generalize for higher-order tensors by summing over remaining indices
                contraction += tensor(i, j2, j3) * vector(j2) * vector(j3);
            }
        }

        // Store the contraction result for index i
        result(i) = contraction;
    }

    return result;
}

double GkCLIPPER::calculate_k_norm(const xt::xarray<double>& x, int k) {
    // Ensure k > 0 to avoid invalid mathematical operations
    if (k <= 0) {
        throw std::invalid_argument("k must be greater than 0.");
    }

    // Compute the sum of absolute values raised to the power of k
    double sum_of_powers = xt::sum(xt::pow(xt::abs(x), k))();

    // Return the k-norm
    return std::pow(sum_of_powers, 1.0 / k);
}

// Function to perform tensor contraction
xt::xarray<double> GkCLIPPER::tensor_contract(const xt::xarray<double>& tensor, const xt::xarray<double>& x, int k) {
    // Simplified contraction for demonstration purposes
    auto result = tensor;
    for (int i = 0; i < k; i++) {
        result = xt::linalg::tensordot(result, x, {0}, {0});
    }
    return result;
}

// Gradient computation
xt::xarray<double> GkCLIPPER::gradient(const xt::xarray<double>& tensor, const xt::xarray<double>& x, int k, double d) {
    auto grad = tensor;
    for (int i = 0; i < k - 1; ++i) {
        // grad = xt::sum(grad * x, {0});
        grad = xt::linalg::tensordot(grad, x, {0}, {0});
    }
    // cout << "\ngrad 1 " << grad;
    auto ones = xt::ones_like(x);
    grad = k * grad + d * x - d * ones * xt::sum(x)();
    
    return k * grad;
}


// Rayleigh quotient computation
double GkCLIPPER::rayleigh_quotient(const xt::xarray<double>& tensor, const xt::xarray<double>& x, int k) {
    // Contract the tensor k times for the numerator
    return tensor_contract(tensor, x, k)();
}

// H-eigenvalue computation
void GkCLIPPER::h_eigenvalue_rayleigh(
    xt::xarray<double> x_init,
    double d,
    int max_iter,
    double tol,
    double step_size,
    double alpha,
    double beta
) {
    bool converged_flag = false;
    xt::xarray<double> x;

    if (x_init.dimension() == 0) {
        x = xt::random::rand<double>({n});
    } else {
        x = x_init;
    }

    x = xt::ones<double>({n}); // TODO delete!

    x = xt::maximum(x, 0.0);
    double norm = calculate_k_norm(x, k);
    x /= norm;

    xt::xarray<double> C = M;
    xt::xarray<double> Md = M + d*C;

    double f_x; 

    for (int iter = 0; iter < max_iter; ++iter) {
        // cout << "\neigenval iters " << iter << endl;

        // Gradient at the current point
        auto start = high_resolution_clock::now();
        auto grad = gradient(Md, x, k, d);
        // cout << "grad " << grad << endl;
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
        // std::cout << "Time grad " << duration.count() << std::endl;

        // Backtracking line search to find step size
        double t = 1.0; // Initial step size
        start = high_resolution_clock::now();
        f_x = rayleigh_quotient(M, x, k);
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        // std::cout << "Time rayleigh " << duration.count() << std::endl;

        while (rayleigh_quotient(M, x + t * grad, k) < f_x + alpha * t * xt::sum(grad * grad)()) {
            t *= beta;
        }

        // cout << "step size " << t << endl;
        // Update x using the computed step size. TODO use backtracking or not? t * grad vs alpha * grad
        start = high_resolution_clock::now();
        xt::xarray<double> x_new = x + t * grad;
        x_new = xt::maximum(x_new, 0.0);        // project onto positive orthant
        // cout << "x new " << x_new << endl;
        norm = calculate_k_norm(x_new, k);
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        // std::cout << "Time max and norm " << duration.count() << std::endl;
        
        if (norm > 0) {
            x_new /= norm;
        } else {
            cerr << "Cannot normalize!" << endl;
            exit(1);
        }

        // Desired accuracy reached by gradient ascent
        start = high_resolution_clock::now();
        if (abs(rayleigh_quotient(M, x_new, k) - rayleigh_quotient(M, x, k)) < tol) {
            converged_flag = true;
            break;
        }
        stop = high_resolution_clock::now();
        duration = duration_cast<microseconds>(stop - start);
        // std::cout << "Time converge check " << duration.count() << std::endl;
        // cout << endl;
        // cout << "\nx old " << x << endl;
        x = x_new;

        // deltaF = Fnew - F;    // change in objective value

        // if (deltaF < -params_.eps) {
        //   // objective value decreased---we need to backtrack, so reduce step size
        //   alpha = alpha * params_.beta;
        // } else {
        //   break; // obj value increased, stop line search
        // }

        // cout << "grad " << grad << endl;
        // cout << "rayleigh " << std::fixed << std::setprecision(15) << f_x << endl;
        // cout << "k norm " << norm << endl;
        // cout << "x new " << x_new << endl;
        // cout << "next x old " << x << endl;
    }

    // Save final results
    rho = rayleigh_quotient(M, x, k);  // Spectral radius
    u = x;      // Corresponding eigenvector
    
    // cout << "u final " << u << endl;
    // cout << "rho final " << rho << endl;
    // exit(1);

    // if (converged_flag) {
    //     cout << "\nH Rayleigh Converged" << endl;
    // } else {
    //     cout << "\nH Rayleigh Not Converged" << endl;
    // }

}

// H-eigenvalue computation
void GkCLIPPER::h_eigenvalue_rayleigh_Fastor(
    std::vector<double> x_init,
    double d,
    int max_iter,
    double tol,
    double step_size,
    double alpha,
    double beta
) {
    bool converged_flag = false;

    std::vector<double> vec(n); 

    if (x_init.empty()) {
        // Create a random number generator
        std::random_device rd; 
        std::mt19937 gen(rd()); 
        std::uniform_real_distribution<double> dis(0.0, 1.0); // Adjust range as needed

        // Fill the vector with random doubles
        for (int i = 0; i < n; ++i) {
            vec[i] = dis(gen);
        }
        cout << "random x_init" << endl;
    } else {
        vec = x_init;
        cout << "use x_init " << endl;
    }

    // auto x = createFastorTensor<double, 2>(vec);
    cout << endl;

    // auto x = initializeTensor<double, n>(x_init);
    // std::cout << "Final tensor: " << x << std::endl;

    // exit(1);

    // x = xt::ones<double>({n}); 

    // x =SingleValueTensor<double, n> a(1.0); // TODO delete!

    // x = xt::maximum(x, 0.0);
    // double norm = calculate_k_norm(x, k);
    // x /= norm;

    // xt::xarray<double> C = M;
    // xt::xarray<double> Md = M + d*C;

    // double f_x; 

    // for (int iter = 0; iter < max_iter; ++iter) {
    //     // cout << "\neigenval iters " << iter << endl;

    //     // Gradient at the current point
    //     auto start = high_resolution_clock::now();
    //     auto grad = gradient(Md, x, k, d);
    //     // cout << "grad " << grad << endl;
    //     auto stop = high_resolution_clock::now();
    //     auto duration = duration_cast<microseconds>(stop - start);
    //     // std::cout << "Time grad " << duration.count() << std::endl;

    //     // Backtracking line search to find step size
    //     double t = 1.0; // Initial step size
    //     start = high_resolution_clock::now();
    //     f_x = rayleigh_quotient(M, x, k);
    //     stop = high_resolution_clock::now();
    //     duration = duration_cast<microseconds>(stop - start);
    //     // std::cout << "Time rayleigh " << duration.count() << std::endl;

    //     while (rayleigh_quotient(M, x + t * grad, k) < f_x + alpha * t * xt::sum(grad * grad)()) {
    //         t *= beta;
    //     }

    //     // cout << "step size " << t << endl;
    //     // Update x using the computed step size. TODO use backtracking or not? t * grad vs alpha * grad
    //     start = high_resolution_clock::now();
    //     xt::xarray<double> x_new = x + t * grad;
    //     x_new = xt::maximum(x_new, 0.0);        // project onto positive orthant
    //     // cout << "x new " << x_new << endl;
    //     norm = calculate_k_norm(x_new, k);
    //     stop = high_resolution_clock::now();
    //     duration = duration_cast<microseconds>(stop - start);
    //     // std::cout << "Time max and norm " << duration.count() << std::endl;
        
    //     if (norm > 0) {
    //         x_new /= norm;
    //     } else {
    //         cerr << "Cannot normalize!" << endl;
    //         exit(1);
    //     }

    //     // Desired accuracy reached by gradient ascent
    //     start = high_resolution_clock::now();
    //     if (abs(rayleigh_quotient(M, x_new, k) - rayleigh_quotient(M, x, k)) < tol) {
    //         converged_flag = true;
    //         break;
    //     }
    //     stop = high_resolution_clock::now();
    //     duration = duration_cast<microseconds>(stop - start);
    //     // std::cout << "Time converge check " << duration.count() << std::endl;
    //     // cout << endl;
    //     // cout << "\nx old " << x << endl;
    //     x = x_new;

    //     // deltaF = Fnew - F;    // change in objective value

    //     // if (deltaF < -params_.eps) {
    //     //   // objective value decreased---we need to backtrack, so reduce step size
    //     //   alpha = alpha * params_.beta;
    //     // } else {
    //     //   break; // obj value increased, stop line search
    //     // }

    //     // cout << "grad " << grad << endl;
    //     // cout << "rayleigh " << std::fixed << std::setprecision(15) << f_x << endl;
    //     // cout << "k norm " << norm << endl;
    //     // cout << "x new " << x_new << endl;
    //     // cout << "next x old " << x << endl;
    // }

    // Save final results
    // rho = rayleigh_quotient(M, x, k);  // Spectral radius
    // u = x;      // Corresponding eigenvector
    
    // cout << "u final " << u << endl;
    // cout << "rho final " << rho << endl;
    // exit(1);

    // if (converged_flag) {
    //     cout << "\nH Rayleigh Converged" << endl;
    // } else {
    //     cout << "\nH Rayleigh Not Converged" << endl;
    // }

}

// Function to get the indices of the top omega values in an xtensor array
xt::xarray<size_t> GkCLIPPER::get_top_n_indices(const xt::xarray<double>& x, size_t omega) {
    if (omega > x.size()) {
        throw std::invalid_argument("omega cannot be greater than the length of the array.");
    }

    // Create an array of indices
    xt::xarray<size_t> indices = xt::arange<size_t>(x.size());

    // Sort the indices based on values in x (descending order)
    xt::xarray<size_t> sorted_indices = xt::argsort(x);
    xt::xarray<size_t> top_indices = xt::view(sorted_indices, xt::range(x.size() - omega, x.size()));

    // Reverse to get indices in descending order of values
    std::reverse(top_indices.begin(), top_indices.end());

    return top_indices;
}


// Function to perform the update calculations
double GkCLIPPER::get_new_d(const xt::xarray<double>& u, const xt::xarray<double>& C, const xt::xarray<double>& M, double d_prev, double eps) {
    double d = d_prev;
    int k = M.dimension();

    // Calculate Cbu
    auto ones = xt::ones_like(u);
    auto Cbu = ones * xt::sum(u)() - tensor_contract(C, u, k - 1) - u;
    // cout << "Cbu " << Cbu << endl;
    // cout << "u " << u << endl;
    // cout << "sum " << ones * xt::sum(u)() << endl;
    // cout << "contract " << tensor_contract(C, u, k - 1) << endl;
    // exit(1);

    // Calculate idxD as a boolean mask
    auto idxD = (Cbu > eps) & (u > eps);

    if (xt::sum(idxD)() > 0) {
        // Calculate Mu
        auto Mu = tensor_contract(M, u, k - 1);

        // Select elements from Mu and Cbu using the idxD mask
        auto num = xt::filter(Mu, idxD);
        auto den = xt::filter(Cbu, idxD);

        // Calculate deltad
        double deltad = xt::mean(xt::abs(num / den))();

        // Update d
        d += deltad;
    } else {
        d = std::numeric_limits<double>::quiet_NaN();
    }

    return d;
}

void GkCLIPPER::solve()
{

    // TODO: init properly
    int max_iter = 1000;
    double eps = 1e-9;

    xt::xarray<double> C = M;

    // Get inital value of u
    auto start = high_resolution_clock::now();
    h_eigenvalue_rayleigh();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    // std::cout << "Time first rayleigh " << duration.count() << std::endl;
    // exit(1);

    double d = get_new_d(u, C, M, 0);
    auto ones = xt::ones_like(u);

    for (int i = 0; i < max_iter; ++i) {
        if (std::isnan(d)) {
            // cout << "d is None" << endl;
            break;
        } else {
            // cout << "INITING X " << u << "  NEW PENATLY " << d << endl;
            h_eigenvalue_rayleigh(u, d);

            double upper_bound = rho / std::tgamma(k + 1) + k;

            auto Cu = C;
            for (int j = 0; j < k - 1; ++j) {
                // Cu = xt::sum(Cu * u, {0});
                Cu = xt::linalg::tensordot(Cu, u, {0}, {0});
            }

            auto Cbu = ones * xt::sum(u)() - Cu - u;
            xt::xarray<int> idxD = xt::cast<int>(((Cbu > eps) & (u > eps)));

            double d_temp = get_new_d(u, C, M, d);
            if (std::isnan(d_temp)) {
                // Clique constraints satisfied
                break;
            } else {
                d = d_temp;
            }
        }
        cout << "solve iters " << i << endl;
    }


}