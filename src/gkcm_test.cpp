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


int main() {
    // xt::random::seed(0);

    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    // // -------------9 node 4d example: Same sized clique, different weights-------------

    // // Initialize a 4D tensor M with zeros using xtensor
    // const int n_set = 9;
    // const int k_set = 4;
    // xt::xarray<double> M = xt::zeros<double>({n_set, n_set, n_set, n_set});

    // // Map to track occurrences of sorted indices
    // std::map<std::array<int, 4>, int> x_tracker;

    // // Generate permutations for [0, 1, 2, 3, 4] taken 4 at a time
    // std::vector<int> set1 = {0, 1, 2, 3, 4};
    // std::vector<std::vector<int>> idxs;
    // generatePermutations(set1, k_set, idxs);

    // // Make graph
    // GMC::KUniformHypergraph<int> graph(k_set);
    // graph.add_nodes({0, 1, 2, 3, 4, 5, 6, 7, 8});

    // for (const auto& idx : idxs) {
    //     std::array<int, 4> key = {idx[0], idx[1], idx[2], idx[3]};
    //     std::sort(key.begin(), key.end());

    //     if (x_tracker.find(key) != x_tracker.end()) {
    //         x_tracker[key]++;
    //     } else {
    //         x_tracker[key] = 0;
    //     }
    //     M(idx[0], idx[1], idx[2], idx[3]) = 0.5;
    //     graph.add_edge(idx);
    // }

    // // Generate permutations for [5, 6, 7, 8] taken 4 at a time
    // std::vector<int> set2 = {4, 5, 6, 7, 8};
    // idxs.clear();
    // generatePermutations(set2, k_set, idxs);

    // for (const auto& idx : idxs) {
    //     std::array<int, 4> key = {idx[0], idx[1], idx[2], idx[3]};
    //     std::sort(key.begin(), key.end());

    //     if (x_tracker.find(key) != x_tracker.end()) {
    //         x_tracker[key]++;
    //     } else {
    //         x_tracker[key] = 0;
    //     }

    //     M(idx[0], idx[1], idx[2], idx[3]) = 1.0;
    //     graph.add_edge(idx);
    // }

    // // Normalize M by dividing by the maximum value in M
    // double max_value = xt::amax(M)();
    // M /= max_value;

    // // Set identity on the diagonal
    // for (int i = 0; i < n_set; ++i) {
    //     M(i, i, i, i) = 1.0;
    // }
    

    // cout << "Done making M "<<  xt::view(M, 0, 1, xt::all(), xt::all()) << endl;

    // -------------Generalized Example-------------

    // For randomized
    std::random_device rd;
    std::default_random_engine rng(rd());

    // Init matrix and params
    const int num_nodes = 10;
    const int k = 2;
    int max_clique_size = 5;
    std::vector<int> clique_sizes = {5, 5};
    std::vector<double> weights = {1.0, 1.0};

    // Init graph
    GMC::KUniformHypergraph<int> graph(k);
    std::vector<int> nodes;
    for(int i=0;i<num_nodes;i++)
    {
        nodes.push_back(i);
        graph.add_node(i);
    }

    // Create max clique
    std::vector<int> max_clique;
    std::shuffle(nodes.begin(), nodes.end(), std::default_random_engine(rng));
    for(int i=0;i<max_clique_size; i++)
    {
        max_clique.push_back(nodes[i]);
    }

    cout << "Actual max clique: " << endl;
    for(auto node : max_clique) {
        cout << node << " ";
    }
    cout << endl;

    auto clique_edges = GMC::combinations_of_size(max_clique, k);
    for( auto e : clique_edges )
    {
        graph.add_edge(e);
    }

    // Add clique to matrix
    xt::xarray<double> M = xt::zeros<double>({num_nodes, num_nodes});
    std::map<std::array<int, k>, int> x_tracker;
    std::vector<std::vector<int>> idxs;
    generatePermutations(max_clique, k, idxs);

    // Add clique edges to matrix 
    for (const auto& idx : idxs) {
        std::array<int, k> key = {idx[0], idx[1]};
        std::sort(key.begin(), key.end());

        if (x_tracker.find(key) != x_tracker.end()) {
            x_tracker[key]++;
        } else {
            x_tracker[key] = 0;
        }
        M(idx[0], idx[1]) = 1.0;
    }
    
    // Add random edges
    auto all_edges = GMC::combinations_of_size(nodes, k, true);
    std::shuffle(all_edges.begin(), all_edges.end(), std::default_random_engine(rng));

    double density = 0.9;
    std::vector< std::vector<int> > added_edges;
    for(int i=0; i < density*all_edges.size();i++)
    {
        added_edges.push_back(all_edges[i]);

        // Add random edges to matrix
        std::vector<std::vector<int>> idxs_rand_edges;
        std::map<std::array<int, k>, int> x_tracker_rand_edges;
        generatePermutations(all_edges[i], k, idxs_rand_edges);

        for (const auto& idx : idxs_rand_edges) {
            std::array<int, k> key = {idx[0], idx[1]};
            std::sort(key.begin(), key.end());

            if (x_tracker_rand_edges.find(key) != x_tracker_rand_edges.end()) {
                x_tracker_rand_edges[key]++;
            } else {
                x_tracker_rand_edges[key] = 0;
            }
            M(idx[0], idx[1]) = 1.0;
        }
    }

    graph.add_edges(added_edges);



    // Normalize M by dividing by the maximum value in M
    double max_value = xt::amax(M)();
    M /= max_value;

    // Set identity on the diagonal
    for (int i = 0; i < num_nodes; ++i) {
        M(i, i) = 1.0;
    }

    cout << "M" << endl << M << endl;
    // exit(1);

    // // -------------9 node 4d example: Same sized cliques, different weights-------------
    // // Initialize a 4D tensor M with zeros using xtensor
    // const int n_set = 9;
    // xt::xarray<double> M = xt::zeros<double>({n_set, n_set, n_set, n_set});

    // // Map to track occurrences of sorted indices
    // std::map<std::array<int, 4>, int> x_tracker;

    // // Generate permutations for [0, 1, 2, 3, 4] taken 4 at a time
    // std::vector<int> set1 = {0, 1, 2, 3, 4};
    // std::vector<std::vector<int>> idxs;
    // generatePermutations(set1, 4, idxs);

    // for (const auto& idx : idxs) {
    //     std::array<int, 4> key = {idx[0], idx[1], idx[2], idx[3]};
    //     std::sort(key.begin(), key.end());

    //     if (x_tracker.find(key) != x_tracker.end()) {
    //         x_tracker[key]++;
    //     } else {
    //         x_tracker[key] = 0;
    //     }
    //     M(idx[0], idx[1], idx[2], idx[3]) = 0.5;
    // }

    // // Generate permutations for [5, 6, 7, 8] taken 4 at a time
    // std::vector<int> set2 = {4, 5, 6, 7, 8};
    // idxs.clear();
    // generatePermutations(set2, 4, idxs);

    // for (const auto& idx : idxs) {
    //     std::array<int, 4> key = {idx[0], idx[1], idx[2], idx[3]};
    //     std::sort(key.begin(), key.end());

    //     if (x_tracker.find(key) != x_tracker.end()) {
    //         x_tracker[key]++;
    //     } else {
    //         x_tracker[key] = 0;
    //     }

    //     M(idx[0], idx[1], idx[2], idx[3]) = 2.0;
    // }

    // // Normalize M by dividing by the maximum value in M
    // double max_value = xt::amax(M)();
    // M /= max_value;

    // // Set identity on the diagonal
    // for (int i = 0; i < n_set; ++i) {
    //     M(i, i, i, i) = 1.0;
    // }

    // // -------------7 node 3d example-------------
    // // GMC::KUniformHypergraph<int> graph(3);
    // // graph.add_nodes({0, 1, 2, 3, 4, 5, 6});
    // // graph.add_edge({0, 1, 4});
    // // graph.add_edge({0, 1, 3});
    // // graph.add_edge({0, 3, 4});
    // // graph.add_edge({1, 3, 4});
    // // graph.add_edge({0, 2, 4});
    // // graph.add_edge({4, 5, 6});
    // // graph.add_edge({3, 4, 6});

    // // -------------9 node 4d example-------------
    // // GMC::KUniformHypergraph<int> graph(4);
    // // graph.add_nodes({0, 1, 2, 3, 4, 5, 6, 7, 8});
    // // graph.add_edge({0, 1, 2, 3});
    // // graph.add_edge({0, 1, 2, 4});
    // // graph.add_edge({0, 1, 3, 4});
    // // graph.add_edge({0, 2, 3, 4});
    // // graph.add_edge({1, 2, 3, 4});
    // // graph.add_edge({5, 6, 7, 8});

    // // -------------12 node 2d example-------------
    // xt::xarray<double> M = {
    //     {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
    //     {0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     {0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
    //     {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0},
    //     {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
    //     {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
    //     {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
    //     {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
    //     {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
    //     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}
    // };
    // int num_nodes = 12;
    // GMC::KUniformHypergraph<int> graph(2);
    // graph.add_nodes({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});
    // int cell;
    // int col_start = 1;
    // for (int row = 0; row < num_nodes; row++) {
    //     for (int col = col_start; col < num_nodes; col++) {
    //         cell = M(row,col);
    //         if (cell == 1) {
    //             graph.add_edge({row, col});
    //         }
    //     };
    //     col_start++;
    // };

    // // Create a Fastor tensor from the xtensor using the template function
    // std::vector<double> M_values(M.data(), M.data() + M.size());
    // auto fastor_M = createFastorTensor<double, 12, 12>(M_values);

    // -------------7 node 3d example-------------
    // // xt::xarray<double> M = {
    // // {{1, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 1, 1, 0, 0},
    // //  {0, 0, 0, 0, 1, 0, 0},
    // //  {0, 1, 0, 0, 1, 0, 0},
    // //  {0, 1, 1, 1, 0, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0}},

    // // {{0, 0, 0, 1, 1, 0, 0},
    // //  {0, 1, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0},
    // //  {1, 0, 0, 0, 1, 0, 0},
    // //  {1, 0, 0, 1, 0, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0}},

    // // {{0, 0, 0, 0, 1, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 1, 0, 0, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0},
    // //  {1, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0}},

    // // {{0, 1, 0, 0, 1, 0, 0},
    // //  {1, 0, 0, 0, 1, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 1, 0, 0, 0},
    // //  {1, 1, 0, 0, 0, 0, 1},
    // //  {0, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 0, 1, 0, 0}},

    // // {{0, 1, 1, 1, 0, 0, 0},
    // //  {1, 0, 0, 1, 0, 0, 0},
    // //  {1, 0, 0, 0, 0, 0, 0},
    // //  {1, 1, 0, 0, 0, 0, 1},
    // //  {0, 0, 0, 0, 1, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 1},
    // //  {0, 0, 0, 1, 0, 1, 0}},

    // // {{0, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 1},
    // //  {0, 0, 0, 0, 0, 1, 0},
    // //  {0, 0, 0, 0, 1, 0, 0}},

    // // {{0, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 0},
    // //  {0, 0, 0, 0, 1, 0, 0},
    // //  {0, 0, 0, 1, 0, 1, 0},
    // //  {0, 0, 0, 0, 1, 0, 0},
    // //  {0, 0, 0, 0, 0, 0, 1}}
    // // };

    // -------------20 node 2d example-------------
    // // xt::xarray<double> M = {
    // //     {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2964, 0.0},
    // //     {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    // //     {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0016, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0747, 0.0},
    // //     {0.0, 0.0, 0.0, 1.0, 0.0, 0.0555, 0.2547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0102, 0.0, 0.7715, 0.0, 0.0, 0.0, 0.0},
    // //     {0.0, 0.0, 0.0, 0.0, 1.0, 0.0063, 0.0, 0.3846, 0.0, 0.0003, 0.0014, 0.0, 0.0, 0.0, 0.0, 0.0063, 0.0, 0.0, 0.0, 0.0},
    // //     {0.0, 0.0, 0.0, 0.0555, 0.0063, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9927, 0.0, 0.0, 0.9722, 0.0, 0.0, 0.0, 0.0},
    // //     {0.0, 0.0, 0.0, 0.2547, 0.0, 0.0, 1.0, 0.0, 0.0023, 0.0, 0.0, 0.8775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    // //     {0.0, 0.0, 0.0, 0.0, 0.3846, 0.0, 0.0, 1.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    // //     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0023, 0.0001, 1.0, 0.7914, 0.0, 0.0, 0.0, 0.0617, 0.0, 0.0, 0.9938, 0.0, 0.0, 0.0007},
    // //     {0.0, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0, 0.0, 0.7914, 1.0, 0.0, 0.0, 0.0001, 0.0091, 0.0, 0.2503, 0.0222, 0.0549, 0.0, 0.0},
    // //     {0.0, 0.0, 0.0, 0.0, 0.0014, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0008},
    // //     {0.0, 0.0, 0.0016, 0.0, 0.0, 0.0, 0.8775, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7007, 0.0},
    // //     {0.0, 0.0, 0.0, 0.0, 0.0, 0.9927, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 1.0, 0.0, 0.9978, 0.0, 0.0, 0.0, 0.0, 0.0},
    // //     {0.0, 0.0138, 0.0, 0.0102, 0.0, 0.0, 0.0, 0.0, 0.0617, 0.0091, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0},
    // //     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9978, 0.0, 1.0, 0.0012, 0.0, 0.0, 0.0, 0.0074},
    // //     {0.0, 0.0, 0.0, 0.7715, 0.0063, 0.9722, 0.0, 0.0, 0.0, 0.2503, 0.0, 0.0, 0.0, 0.0, 0.0012, 1.0, 0.0026, 0.0217, 0.0, 0.0},
    // //     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9938, 0.0222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0026, 1.0, 0.0, 0.0, 0.0},
    // //     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0549, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0217, 0.0, 1.0, 0.0007, 0.0},
    // //     {0.2964, 0.0, 0.0747, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0007, 1.0, 0.0},
    // //     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0007, 0.0, 0.0008, 0.0, 0.0, 0.0, 0.0074, 0.0, 0.0, 0.0, 0.0, 1.0}
    // // };

    // // ----------------Xtensor test----------------
    // xt::xarray<double> xtens = {{1,2,3},{4,5,6},{7,8,9}};
    // xt::xarray<double> xvec = {1, 2, 3};
    // start = high_resolution_clock::now();
    // xt::xarray<double> xresult= xt::linalg::tensordot(xtens, xvec, {0}, {0});
    // stop = high_resolution_clock::now();
    // duration = duration_cast<microseconds>(stop - start);

    // cout << "x tensor " << xresult << endl;
    // cout << "time sec " << duration.count()*1e-6 << endl << endl;

    // //  ----------------Eigen test----------------
    // Matrix<double, 3, 3> eigens = (Matrix<double, 3, 3>() << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished();
    // Vector3d eivec(1, 2, 3);

    // // Measure execution time
    // start = high_resolution_clock::now();
    // Vector3d eiresult = eigens.transpose() * eivec;
    // stop = high_resolution_clock::now();
    // duration = duration_cast<microseconds>(stop - start);

    // // Output results
    // cout << "Eigen tensor: " << eiresult.transpose() << endl;
    // cout << "time sec " << duration.count() * 1e-6 << endl << endl;;

    // //  ----------------Fastor test----------------
    // // Define the tensor and vector
    // Tensor<double, 3, 3> ftens = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    // Tensor<double, 3> fvec = {1, 2, 3};

    // // Start the timer
    // start = high_resolution_clock::now();

    // // Perform the tensor contraction using einsum
    // Tensor<double, 3> fresult = einsum<Fastor::Index<0, 1>, Fastor::Index<0>>(ftens, fvec);

    // // Stop the timer
    // stop = high_resolution_clock::now();
    // duration = duration_cast<microseconds>(stop - start);

    // // Output the result
    // cout << "Fastor tensor (einsum equivalent): ";
    // for (size_t i = 0; i < 3; ++i) {
    //     cout << fresult(i) << " ";
    // }
    // cout << endl;
    // cout << "time: " << duration.count() * 1e-6 << " seconds" << endl << endl;

    // //  ----------------Test Fastor templating----------------
    // try {
    //     // Example: Create a 2x3 tensor
    //     auto tensor = createFastorTensor<double, 2, 3>({1, 2, 3, 4, 5, 6});
    //     std::cout << "Created Tensor:\n" << tensor << std::endl;
    // } catch (const std::exception& ex) {
    //     std::cerr << "Error: " << ex.what() << std::endl;
    // }

    // std::cout << "\n";

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

    GMC::print(std::cout, clique_h);
    std::cout << "\n";
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time heuristic sec " << duration.count() * 1e-6 << std::endl;

    GMC::MaxCliqueSolverExact<int> solver_exact;
    start = high_resolution_clock::now();
    auto clique_e = solver_exact.max_clique(graph, true);
    stop = high_resolution_clock::now();

    GMC::print(std::cout, clique_e);
    std::cout << "\n";
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time exact sec " << duration.count() * 1e-6 << std::endl << std::endl;

    // // ----------------------------GkCLIPPER----------------------------
    // TODO: init properly
    // size_t k = M.dimension();                       // Tensor order
    double n_temp = pow(M.size(), 1.0/k);           // Weird casting issues when type size_t, use temp double
    size_t n = static_cast< float > (n_temp);

    cout << "\nGkCLIPPER" << endl;
    GkCLIPPER gkclipper(n, k, M);

    cout << "\nbegin solve" << endl;
    start = high_resolution_clock::now();
    gkclipper.solve();
    stop = high_resolution_clock::now();
    cout << "end solve" << endl;

    int upper_bound = static_cast<int>(std::round(gkclipper.rho));//static_cast<int>(std::round(gkclipper.rho / std::tgamma(k + 1) + k));
    cout << "\nFinal spectral radius: " << gkclipper.rho << endl;
    // cout << "Final clique number upper bound: " << gkclipper.rho / std::tgamma(k + 1) + k << ", " << upper_bound << endl;
    cout << "Upper bound " << upper_bound << endl;
    // cout << "Final eigenvector: " << gkclipper.u << endl;
    cout << "Densest clique: " << gkclipper.get_top_n_indices(gkclipper.u, upper_bound) << endl;
    cout << "done" << endl << endl;
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time GkCLIPPER sec " << duration.count() * 1e-6 << std::endl;

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
