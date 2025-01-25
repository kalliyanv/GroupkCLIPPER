#include <cassert>
#include <coin/IpIpoptApplication.hpp>
#include <coin/IpTNLP.hpp>
#include <cmath>
#include <vector>

using namespace Ipopt;

// Define the optimization problem
class TensorOptimization : public TNLP {
public:
    TensorOptimization(const std::vector<std::vector<double>>& M, int dim, double k)
        : M_(M), dim_(dim), k_(k) {}

    // Destructor
    virtual ~TensorOptimization() {}

    // Problem dimensions
    virtual bool get_nlp_info(Index& n, Index& m, Index& nnz_jac_g, Index& nnz_h_lag, IndexStyleEnum& index_style) {
        n = dim_;                     // Number of variables (size of vector v)
        m = 1;                        // Number of constraints (\|v\|_2 = 1)
        nnz_jac_g = dim_;             // Nonzero entries in Jacobian (gradient of constraint)
        nnz_h_lag = dim_ * dim_;      // Nonzero entries in Hessian (full matrix for simplicity)
        index_style = TNLP::C_STYLE;  // Use C-style indexing
        return true;
    }

    // Variable bounds
    virtual bool get_bounds_info(Index n, Number* x_l, Number* x_u, Index m, Number* g_l, Number* g_u) {
        assert(n == dim_);
        assert(m == 1);

        // Bounds for variables (no bounds here, use -inf/inf)
        for (Index i = 0; i < n; i++) {
            x_l[i] = -2e19;  // Unbounded
            x_u[i] = 2e19;   // Unbounded
        }

        // Bounds for the constraint (\|v\|_2 = 1)
        g_l[0] = 1.0;
        g_u[0] = 1.0;
        return true;
    }

    // Initial values
    virtual bool get_starting_point(Index n, bool init_x, Number* x, bool init_z, Number* z_L, Number* z_U,
                                    Index m, bool init_lambda, Number* lambda) {
        assert(init_x == true);
        assert(n == dim_);

        // Start with a unit vector
        for (Index i = 0; i < n; i++) {
            x[i] = 1.0 / dim_;  // Example initialization
        }
        return true;
    }

    // Objective function
    virtual bool eval_f(Index n, const Number* x, bool new_x, Number& obj_value) {
        assert(n == dim_);

        obj_value = 0.0;
        for (Index i = 0; i < dim_; i++) {
            double sum = 0.0;
            for (Index j = 0; j < dim_; j++) {
                sum += M_[i][j] * std::pow(x[j], k_);
            }
            obj_value += sum;
        }
        obj_value = -obj_value;  // Negate for maximization
        return true;
    }

    // Gradient of the objective function
    virtual bool eval_grad_f(Index n, const Number* x, bool new_x, Number* grad_f) {
        assert(n == dim_);

        for (Index i = 0; i < dim_; i++) {
            grad_f[i] = 0.0;
            for (Index j = 0; j < dim_; j++) {
                grad_f[i] += M_[j][i] * k_ * std::pow(x[i], k_ - 1);
            }
        }
        for (Index i = 0; i < n; i++) grad_f[i] *= -1.0;  // Negate for maximization
        return true;
    }

    // Constraints
    virtual bool eval_g(Index n, const Number* x, bool new_x, Index m, Number* g) {
        assert(m == 1);
        assert(n == dim_);

        g[0] = 0.0;
        for (Index i = 0; i < n; i++) {
            g[0] += x[i] * x[i];
        }
        return true;
    }

    // Jacobian of constraints
    virtual bool eval_jac_g(Index n, const Number* x, bool new_x, Index m, Index nele_jac, Index* iRow, Index* jCol, Number* values) {
        assert(m == 1);
        assert(n == dim_);

        if (values == nullptr) {
            // Structure of Jacobian
            for (Index i = 0; i < n; i++) {
                iRow[i] = 0;
                jCol[i] = i;
            }
        } else {
            // Values of Jacobian
            for (Index i = 0; i < n; i++) {
                values[i] = 2.0 * x[i];
            }
        }
        return true;
    }

    // Hessian (not required for now)
    virtual bool eval_h(Index n, const Number* x, bool new_x, Number obj_factor, Index m, const Number* lambda,
                        bool new_lambda, Index nele_hess, Index* iRow, Index* jCol, Number* values) {
        return false;  // Use quasi-Newton
    }

    // Finalize solution
    virtual void finalize_solution(SolverReturn status, Index n, const Number* x, const Number* z_L, const Number* z_U,
                                   Index m, const Number* g, const Number* lambda, Number obj_value,
                                   const IpoptData* ip_data, IpoptCalculatedQuantities* ip_cq) {
        std::cout << "Solution:" << std::endl;
        for (Index i = 0; i < n; i++) {
            std::cout << "v[" << i << "] = " << x[i] << std::endl;
        }
        std::cout << "Objective value = " << -obj_value << std::endl;  // Negate back
    }

private:
    const std::vector<std::vector<double>>& M_;
    int dim_;
    double k_;
};

int main() {
    // Example tensor and parameters
    int dim = 3;
    std::vector<std::vector<double>> M = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    double k = 2.0;

    // Create the optimization problem
    SmartPtr<TNLP> mynlp = new TensorOptimization(M, dim, k);

    // Create and solve the problem
    SmartPtr<IpoptApplication> app = IpoptApplicationFactory();
    app->Initialize();
    ApplicationReturnStatus status = app->OptimizeTNLP(mynlp);

    if (status == Solve_Succeeded) {
        std::cout << "Optimization succeeded!" << std::endl;
    } else {
        std::cout << "Optimization failed!" << std::endl;
    }

    return 0;
}
