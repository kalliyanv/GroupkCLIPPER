#include "gen_max_clique/gmc.hpp"
#include <random>
#include <iostream>

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <clipper/clipper.h>
#include <clipper/utils.h>

#include "GkCLIPPER.h"



#include <iostream>
#include <xtensor/xarray.hpp>
#include <xtensor-blas/xlinalg.hpp>

// Example from Brendon's code src/gen_max_clique_tests/max_clique_exact_tests.cpp

// Function to generate permutations of a vector
void generatePermutations(const std::vector<int>& elements, int k, std::vector<std::vector<int>>& result) {
    std::vector<int> temp(elements);
    std::sort(temp.begin(), temp.end());
    do {
        result.push_back(std::vector<int>(temp.begin(), temp.begin() + k));
    } while (std::next_permutation(temp.begin(), temp.end()));
}

void add_edges_from_tensor(const xt::xarray<int>& tensor, GMC::KUniformHypergraph<int>& graph, std::vector<int> indices = {}) {
 
    if (tensor.element(indices.begin(), indices.end()) == 1) {
        graph.add_edge(indices);
    }
}


int main(int argc, char* argv[]) {
    int num_runs = 100;
    int num_correct = 0;
    double time_h = 0;
    double time_c = 0;

    for(int run_idx=0; run_idx < num_runs; run_idx++) {

        // xt::random::seed(0);
    
        auto start = high_resolution_clock::now();
        auto stop = high_resolution_clock::now();
        auto duration = duration_cast<microseconds>(stop - start);
    
    
        // -------------Generalized Example-------------
        cout << "Making graph" << endl;
    
        // For randomized
        std::random_device rd;
        std::default_random_engine rng(rd());
    
        // Init matrix and params
        int num_nodes = 100;
        int max_clique_size = 10;
    
        // Parse command-line arguments
        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
    
            if (arg.rfind("--num_nodes=", 0) == 0) {  // Starts with "--num_nodes="
                num_nodes = std::atoi(arg.substr(12).c_str());  // Extract value after '='
            } 
            else if (arg.rfind("--max_clique_size=", 0) == 0) {  // Starts with "--max_clique_size="
                max_clique_size = std::atoi(arg.substr(18).c_str());  // Extract value after '='
            } 
            else {
                std::cerr << "Unknown argument: " << arg << "\n";
                return 1;
            }
        }
        const int k = 3;
    
        // Output parsed values
        // std::cout << "Number of nodes: " << num_nodes << "\n";
        // std::cout << "Max clique size: " << max_clique_size << "\n";
        // std::cout << "k: " << k << "\n";
    
    
        // Init graph
        GMC::KUniformHypergraph<int> graph(k);
        std::vector<int> nodes;
        for(int i=0;i<num_nodes;i++)
        {
            nodes.push_back(i);
            graph.add_node(i);
        }
    
        // Create max clique
        std::vector<int> max_clique; //= {0, 1, 2, 3, 4};
        std::shuffle(nodes.begin(), nodes.end(), std::default_random_engine(rng));
        for(int i=0;i<max_clique_size; i++)
        {
            max_clique.push_back(nodes[i]);
        }
    
        stable_sort(max_clique.begin(), max_clique.end());
    
        auto clique_edges = GMC::combinations_of_size(max_clique, k);
        for( auto e : clique_edges )
        {
            graph.add_edge(e);
        }
    
        // Add clique to matrix
        xt::xarray<double> M;
        if (k==2) {
            M = xt::zeros<double>({num_nodes, num_nodes});   
        } else if (k==3) {
            M = xt::zeros<double>({num_nodes, num_nodes, num_nodes});
        } else {
            std::cerr << "Invalid k: " << k << endl;
        }
        std::vector<std::vector<int>> idxs;
        generatePermutations(max_clique, k, idxs);
    
        // Add clique edges to matrix 
        for (const auto& idx : idxs) {
            if (k==2) {
                M(idx[0], idx[1]) = 1.0;
            } else if (k==3) {
                M(idx[0], idx[1], idx[2]) = 1.0;
            }
        }
        
        // Add random edges
        auto all_edges = GMC::combinations_of_size(nodes, k, true);
        std::shuffle(all_edges.begin(), all_edges.end(), std::default_random_engine(rng));
    
        double density = 0.1;
        std::vector< std::vector<int> > added_edges;
        for(int i=0; i < density*all_edges.size();i++)
        {
            added_edges.push_back(all_edges[i]);
    
            // Add random edges to matrix
            std::vector<std::vector<int>> idxs_rand_edges;
            generatePermutations(all_edges[i], k, idxs_rand_edges);
    
            for (const auto& idx : idxs_rand_edges) {
                if (k==2) {
                    M(idx[0], idx[1]) = 1.0;
                } else if (k==3) {
                    M(idx[0], idx[1], idx[2]) = 1.0;
                }
            }
        }
    
        graph.add_edges(added_edges);
    
        // Normalize M by dividing by the maximum value in M
        double max_value = xt::amax(M)();
        M /= max_value;
    
        // Set identity on the diagonal
        for (int i = 0; i < num_nodes; ++i) {
            if (k==2) {
                M(i, i) = 1.0;
            } else {
                M(i, i, i) = 1.0;
            }
        }
        cout << "Done making graph" << endl << endl;
        // cout << "M" << endl << M << endl;
        // exit(1);
    
    
        // //  ----------------CLIPPER test----------------
        // cout << "\nCLIPPER" << endl;
        // clipper::invariants::EuclideanDistance::Params iparams;
        // clipper::invariants::EuclideanDistancePtr invariant = std::make_shared<clipper::invariants::EuclideanDistance>(iparams);
        // clipper::Params params;
        // clipper::CLIPPER clipper(invariant, params);
        
        // // create a target/model point cloud of data
        // Eigen::Matrix3Xd model(3, 4);
        // model.col(0) << 0, 0, 0;
        // model.col(1) << 2, 0, 0;
        // model.col(2) << 0, 3, 0;
        // model.col(3) << 2, 2, 0;
    
        // // transform of data w.r.t model
        // Eigen::Affine3d T_MD;
        // T_MD = Eigen::AngleAxisd(M_PI/8, Eigen::Vector3d::UnitZ());
        // T_MD.translation() << 5, 3, 0;
    
        // // create source/data point cloud
        // Eigen::Matrix3Xd data = T_MD.inverse() * model;
    
        // // remove one point from the tgt (model) cloud---simulates a partial view
        // data.conservativeResize(3, 3);
    
        //     // an empty association set will be assumed to be all-to-all
        // clipper.scorePairwiseConsistency(model, data);
    
        // // find the densest clique of the previously constructed consistency graph
        // start = high_resolution_clock::now();
        // clipper.solve();
        // stop = high_resolution_clock::now();
        // duration = duration_cast<microseconds>(stop - start);
        // auto inliers = clipper.getSelectedAssociations();
        // // cout << "Clipper inliers " << inliers << endl; 
        // cout << "Clipper time sec " << duration.count()*1e-6 << endl << endl;
    
        // // ------------GKCM--------------
    
        // // // Find clique
        // // GMC::MaxCliqueSolverHeuristic<int> solver_heuristic;
        // // auto start = high_resolution_clock::now();
        // // auto clique_h = solver_heuristic.max_clique_heu(graph, true);
        // // auto stop = high_resolution_clock::now();
    
        // // GMC::print(std::cout, clique_h);
        // // std::cout << "\n";
        // // auto duration = duration_cast<microseconds>(stop - start);
        // // std::cout << "Time heuristic microsec " << duration.count() << std::endl;
    
        // // GMC::MaxCliqueSolverExact<int> solver_exact;
        // // start = high_resolution_clock::now();
        // // auto clique_e = solver_exact.max_clique(graph, true);
        // // stop = high_resolution_clock::now();
    
        // // GMC::print(std::cout, clique_e);
        // // std::cout << "\n";
        // // duration = duration_cast<microseconds>(stop - start);
        // // std::cout << "Time exact microsec " << duration.count() << std::endl;
    
        // // ------------GkCM Pt. 2 
        cout << "\nGkCM" << endl;
        // Find clique
        GMC::MaxCliqueSolverHeuristic<int> solver_heuristic;
        start = high_resolution_clock::now();
        auto clique_h = solver_heuristic.max_clique_heu(graph, true);
        stop = high_resolution_clock::now();
    
        // GMC::print(std::cout, clique_h);
        // std::cout << "\n";
        auto duration_h = duration_cast<microseconds>(stop - start);
        // std::cout << "Time heuristic sec " << duration_h.count() * 1e-6 << std::endl;
    
        // GMC::MaxCliqueSolverExact<int> solver_exact;
        // start = high_resolution_clock::now();
        // auto clique_e = solver_exact.max_clique(graph, true);
        // stop = high_resolution_clock::now();
    
        // GMC::print(std::cout, clique_e);
        // std::cout << "\n";
        // duration = duration_cast<microseconds>(stop - start);
        // std::cout << "Time exact sec " << duration.count() * 1e-6 << std::endl << std::endl;
    
        // // ----------------------------GkCLIPPER----------------------------
        // TODO: init properly
        // size_t k = M.dimension();                       // Tensor order
        double n_temp = pow(M.size(), 1.0/k);           // Weird casting issues when type size_t, use temp double
        size_t n = static_cast< float > (n_temp);
    
        // cout << "\nGkCLIPPER" << endl;
        GkCLIPPER gkclipper(n, k, M);
    
        // cout << "\nbegin solve" << endl;
        start = high_resolution_clock::now();
        gkclipper.solve();
        stop = high_resolution_clock::now();
        // cout << "end solve" << endl;
        duration = duration_cast<microseconds>(stop - start);
    
        int upper_bound = static_cast<int>(std::round(gkclipper.rho / std::tgamma(k + 1) + k)); //static_cast<int>(std::round(gkclipper.rho));//
        // cout << "\nFinal spectral radius: " << gkclipper.rho << endl;
        // cout << "Final clique number upper bound: " << gkclipper.rho / std::tgamma(k + 1) + k << ", " << upper_bound << endl;
        // cout << "Upper bound " << upper_bound << endl;
        // cout << "Final eigenvector: " << gkclipper.u << endl;
        cout << "Actual max clique: ";
        for(auto node : max_clique) {
            cout << node << " ";
        }
        cout << endl;
        auto my_clique = gkclipper.get_top_n_indices(gkclipper.u, max_clique_size);
        cout << "Densest clique: " << my_clique << endl;
    
        cout << "Heuristic max clique: ";
        GMC::print(std::cout, clique_h);
        std::cout << "\n";
        std::cout << "Time GkCLIPPER sec " << duration.count() * 1e-6 << std::endl;
        std::cout << "Time heuristic sec " << duration_h.count() * 1e-6 << std::endl;
        time_h += duration_h.count() * 1e-6;
        time_c += duration.count() * 1e-6 ;
    
        gkclipper.print_sorted_values(gkclipper.u);
    
        // cout << "\nCompare Values" << endl;
        // cout << "Actual Max Clique u Vals" << endl;
        // for(auto node : max_clique) {
        //     cout << gkclipper.u[node] << " ";
        // }
        // cout << "\nMy Clique u Vals" << endl;
        // for(auto node : my_clique) {
        //     cout << gkclipper.u[node] << " ";
        // }
        // cout << endl;
    
        bool correct_clique = true;
        for(int i=0; i < max_clique.size(); i++) {
            if(max_clique[i] != my_clique[i]) {
                correct_clique = false;
                break;
            }
        }
        if (correct_clique){
            cout << "\nClique is correct!" << endl;
            num_correct++;
        } else {
            cout << "\nWrong clique" << endl;
        }

    }
    cout << "Num correct: " << num_correct << endl;
    cout << "Total: " << num_runs << endl;

    cout << "\nAvg time CLIPPER: " << time_c / num_runs << endl;
    cout << "\nAvg time GkCM: " << time_h / num_runs << endl;

    // // // ------------Random Clique Generation--------------

    // // For randomized
    // std::random_device rd;
    // std::default_random_engine rng(rd());

    // std::cout << "Using gen_max_clique library" << std::endl;
    // // Example function or class usage from gmc.hpp
    // int k = 3;
    // int num_nodes = 10;
    // int max_clique_size = 5;

    // GMC::KUniformHypergraph<int> graph(k);

    // // Init graph nodes
    // std::vector<int> nodes;
    // for(int i=0;i<num_nodes;i++)
    // {
    //     nodes.push_back(i);
    //     graph.add_node(i);
    // }

    // auto M = xt::zeros<double>({num_nodes, num_nodes, num_nodes}); // gkclipper, init matrix with zeros
    // // cout << "M " << M << endl; 

    // // Create max clique
    // std::vector<int> max_clique;
    // std::shuffle(nodes.begin(), nodes.end(), std::default_random_engine(rng));
    // for(int i=0;i<max_clique_size; i++)
    // {
    //     max_clique.push_back(nodes[i]);
    // }
    // std::sort(max_clique.begin(), max_clique.end());

    // auto clique_edges = GMC::combinations_of_size(max_clique, k);
    // for( auto e : clique_edges )
    // {
    //     graph.add_edge(e);
    // }

    // // Add random edges
    // auto all_edges = GMC::combinations_of_size(nodes, k, true);
    // std::shuffle(all_edges.begin(), all_edges.end(), std::default_random_engine(rng));

    // double density = 0.2;
    // std::vector< std::vector<int> > added_edges;
    // for(int i=0; i < density*all_edges.size();i++)
    // {
    //     added_edges.push_back(all_edges[i]);
    // }

    // graph.add_edges(added_edges);

    // // Find clique
    // GMC::MaxCliqueSolverExact<int> solver;
    // start = high_resolution_clock::now();
    // auto clique = solver.max_clique(graph, true);
    // stop = high_resolution_clock::now();
    // duration = duration_cast<microseconds>(stop - start);
    // GMC::print(std::cout, clique);
    // std::cout << "\n";
    // cout << "Time clipper sec " << duration.count()*1e-6 << endl;

    return 0;
}
