#include "gen_max_clique/gmc.hpp"
#include <random>
#include <iostream>

#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <clipper/clipper.h>
#include <clipper/utils.h>

#include "GkCLIPPER.h"

// From Brendon's code src/gen_max_clique_tests/max_clique_exact_tests.cpp

int main() {

    auto start = high_resolution_clock::now();
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);

    // Xtensor test
    xt::xarray<double> xtens = {{1,2,3},{4,5,6},{7,8,9}};
    xt::xarray<double> xvec = {1, 2, 3};
    start = high_resolution_clock::now();
    xt::xarray<double> xresult= xt::linalg::tensordot(xtens, xvec, {0}, {0});
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    cout << "x tensor " << xresult << endl;
    cout << "time sec " << duration.count()*1e-6 << endl << endl;

    // Eigen test
    Matrix<double, 3, 3> eigens = (Matrix<double, 3, 3>() << 1, 2, 3, 4, 5, 6, 7, 8, 9).finished();
    Vector3d eivec(1, 2, 3);

    // Measure execution time
    start = high_resolution_clock::now();
    Vector3d eiresult = eigens.transpose() * eivec;
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    // Output results
    cout << "Eigen tensor: " << eiresult.transpose() << endl;
    cout << "time sec " << duration.count() * 1e-6 << endl << endl;;

    // Fastor test
    // Define the tensor and vector
    Tensor<double, 3, 3> ftens = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
    Tensor<double, 3> fvec = {1, 2, 3};

    // Start the timer
    start = high_resolution_clock::now();

    // Perform the tensor contraction using einsum
    Tensor<double, 3> fresult = einsum<Fastor::Index<0, 1>, Fastor::Index<0>>(ftens, fvec);

    // Stop the timer
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);

    // Output the result
    cout << "Fastor tensor (einsum equivalent): ";
    for (size_t i = 0; i < 3; ++i) {
        cout << fresult(i) << " ";
    }
    cout << endl;
    cout << "time: " << duration.count() * 1e-6 << " seconds" << endl << endl;

    // Test Fastor templating
    try {
        // Example: Create a 2x3 tensor
        auto tensor = createFastorTensor<double, 2, 3>({1, 2, 3, 4, 5, 6});
        std::cout << "Created Tensor:\n" << tensor << std::endl;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << std::endl;
    }

    std::cout << "\n";

    // CLIPPER test
    clipper::invariants::EuclideanDistance::Params iparams;
    clipper::invariants::EuclideanDistancePtr invariant = std::make_shared<clipper::invariants::EuclideanDistance>(iparams);
    clipper::Params params;
    clipper::CLIPPER clipper(invariant, params);
    
    // create a target/model point cloud of data
    Eigen::Matrix3Xd model(3, 4);
    model.col(0) << 0, 0, 0;
    model.col(1) << 2, 0, 0;
    model.col(2) << 0, 3, 0;
    model.col(3) << 2, 2, 0;

    // transform of data w.r.t model
    Eigen::Affine3d T_MD;
    T_MD = Eigen::AngleAxisd(M_PI/8, Eigen::Vector3d::UnitZ());
    T_MD.translation() << 5, 3, 0;

    // create source/data point cloud
    Eigen::Matrix3Xd data = T_MD.inverse() * model;

    // remove one point from the tgt (model) cloud---simulates a partial view
    data.conservativeResize(3, 3);

        // an empty association set will be assumed to be all-to-all
    clipper.scorePairwiseConsistency(model, data);

    // find the densest clique of the previously constructed consistency graph
    start = high_resolution_clock::now();
    clipper.solve();
    stop = high_resolution_clock::now();
    duration = duration_cast<microseconds>(stop - start);
    auto inliers = clipper.getSelectedAssociations();
    cout << "Clipper inliers " << inliers << endl; 
    cout << "Clipper time sec " << duration.count()*1e-6 << endl << endl;

    // ------------From known graph (not working GkCLIPPER)--------------
    // GMC::KUniformHypergraph<int> graph(4);
    // graph.add_nodes({0, 1, 2, 3, 4, 5, 6, 7, 8});
    // graph.add_edge({0, 1, 2, 3});
    // graph.add_edge({0, 1, 2, 4});
    // graph.add_edge({0, 1, 3, 4});
    // graph.add_edge({0, 2, 3, 4});
    // graph.add_edge({1, 2, 3, 4});

    // graph.add_edge({5, 6, 7, 8});

    // // Find clique
    // GMC::MaxCliqueSolverHeuristic<int> solver_heuristic;
    // auto start = high_resolution_clock::now();
    // auto clique_h = solver_heuristic.max_clique_heu(graph, true);
    // auto stop = high_resolution_clock::now();

    // GMC::print(std::cout, clique_h);
    // std::cout << "\n";
    // auto duration = duration_cast<microseconds>(stop - start);
    // std::cout << "Time heuristic microsec " << duration.count() << std::endl;

    // GMC::MaxCliqueSolverExact<int> solver_exact;
    // start = high_resolution_clock::now();
    // auto clique_e = solver_exact.max_clique(graph, true);
    // stop = high_resolution_clock::now();

    // GMC::print(std::cout, clique_e);
    // std::cout << "\n";
    // duration = duration_cast<microseconds>(stop - start);
    // std::cout << "Time exact microsec " << duration.count() << std::endl;

    // // ------------From known graph (not working GkCLIPPER)--------------
    // GMC::KUniformHypergraph<int> graph(3);

    // graph.add_nodes({0, 1, 2, 3, 4, 5, 6});
    // graph.add_edge({0, 1, 4});
    // graph.add_edge({0, 1, 3});
    // graph.add_edge({0, 3, 4});
    // graph.add_edge({1, 3, 4});
    // graph.add_edge({0, 2, 4});
    // graph.add_edge({4, 5, 6});
    // graph.add_edge({3, 4, 6});

    xt::xarray<double> M = {
        {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
        {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0},
        {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0},
        {0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0},
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}
    };

    // Create a Fastor tensor from the xtensor using the template function
    std::vector<double> M_values(M.data(), M.data() + M.size());
    auto fastor_M = createFastorTensor<double, 12, 12>(M_values);
    
    int num_nodes = 12;

    GMC::KUniformHypergraph<int> graph(2);
    graph.add_nodes({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11});

    int cell;
    int col_start = 1;
    for (int row = 0; row < num_nodes; row++) {
        for (int col = col_start; col < num_nodes; col++) {
            cell = M(row,col);
            if (cell == 1) {
                graph.add_edge({row, col});
            }
        };
        col_start++;
    };

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



    // xt::xarray<double> M = {
    // {{1, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 1, 1, 0, 0},
    //  {0, 0, 0, 0, 1, 0, 0},
    //  {0, 1, 0, 0, 1, 0, 0},
    //  {0, 1, 1, 1, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0}},

    // {{0, 0, 0, 1, 1, 0, 0},
    //  {0, 1, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0},
    //  {1, 0, 0, 0, 1, 0, 0},
    //  {1, 0, 0, 1, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0}},

    // {{0, 0, 0, 0, 1, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 1, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0},
    //  {1, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0}},

    // {{0, 1, 0, 0, 1, 0, 0},
    //  {1, 0, 0, 0, 1, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 1, 0, 0, 0},
    //  {1, 1, 0, 0, 0, 0, 1},
    //  {0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 1, 0, 0}},

    // {{0, 1, 1, 1, 0, 0, 0},
    //  {1, 0, 0, 1, 0, 0, 0},
    //  {1, 0, 0, 0, 0, 0, 0},
    //  {1, 1, 0, 0, 0, 0, 1},
    //  {0, 0, 0, 0, 1, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 1},
    //  {0, 0, 0, 1, 0, 1, 0}},

    // {{0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 1},
    //  {0, 0, 0, 0, 0, 1, 0},
    //  {0, 0, 0, 0, 1, 0, 0}},

    // {{0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 0},
    //  {0, 0, 0, 0, 1, 0, 0},
    //  {0, 0, 0, 1, 0, 1, 0},
    //  {0, 0, 0, 0, 1, 0, 0},
    //  {0, 0, 0, 0, 0, 0, 1}}
    // };

    // xt::xarray<double> M = {
    //     {1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2964, 0.0},
    //     {0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     {0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0016, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0747, 0.0},
    //     {0.0, 0.0, 0.0, 1.0, 0.0, 0.0555, 0.2547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0102, 0.0, 0.7715, 0.0, 0.0, 0.0, 0.0},
    //     {0.0, 0.0, 0.0, 0.0, 1.0, 0.0063, 0.0, 0.3846, 0.0, 0.0003, 0.0014, 0.0, 0.0, 0.0, 0.0, 0.0063, 0.0, 0.0, 0.0, 0.0},
    //     {0.0, 0.0, 0.0, 0.0555, 0.0063, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9927, 0.0, 0.0, 0.9722, 0.0, 0.0, 0.0, 0.0},
    //     {0.0, 0.0, 0.0, 0.2547, 0.0, 0.0, 1.0, 0.0, 0.0023, 0.0, 0.0, 0.8775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     {0.0, 0.0, 0.0, 0.0, 0.3846, 0.0, 0.0, 1.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0023, 0.0001, 1.0, 0.7914, 0.0, 0.0, 0.0, 0.0617, 0.0, 0.0, 0.9938, 0.0, 0.0, 0.0007},
    //     {0.0, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0, 0.0, 0.7914, 1.0, 0.0, 0.0, 0.0001, 0.0091, 0.0, 0.2503, 0.0222, 0.0549, 0.0, 0.0},
    //     {0.0, 0.0, 0.0, 0.0, 0.0014, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0008},
    //     {0.0, 0.0, 0.0016, 0.0, 0.0, 0.0, 0.8775, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7007, 0.0},
    //     {0.0, 0.0, 0.0, 0.0, 0.0, 0.9927, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 1.0, 0.0, 0.9978, 0.0, 0.0, 0.0, 0.0, 0.0},
    //     {0.0, 0.0138, 0.0, 0.0102, 0.0, 0.0, 0.0, 0.0, 0.0617, 0.0091, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0},
    //     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9978, 0.0, 1.0, 0.0012, 0.0, 0.0, 0.0, 0.0074},
    //     {0.0, 0.0, 0.0, 0.7715, 0.0063, 0.9722, 0.0, 0.0, 0.0, 0.2503, 0.0, 0.0, 0.0, 0.0, 0.0012, 1.0, 0.0026, 0.0217, 0.0, 0.0},
    //     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9938, 0.0222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0026, 1.0, 0.0, 0.0, 0.0},
    //     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0549, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0217, 0.0, 1.0, 0.0007, 0.0},
    //     {0.2964, 0.0, 0.0747, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0007, 1.0, 0.0},
    //     {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0007, 0.0, 0.0008, 0.0, 0.0, 0.0, 0.0074, 0.0, 0.0, 0.0, 0.0, 1.0}
    // };


    // TODO: init properly
    size_t k = M.dimension();                       // Tensor order
    double n_temp = pow(M.size(), 1.0/k);           // Weird casting issues when type size_t, use temp double
    size_t n = static_cast< float > (n_temp);

    GkCLIPPER gkclipper(n, k, M);

    // Test Fastor functions
    gkclipper.h_eigenvalue_rayleigh_Fastor();

    cout << "\nbegin solve" << endl;
    start = high_resolution_clock::now();
    gkclipper.solve();
    stop = high_resolution_clock::now();
    cout << "end solve" << endl;

    int upper_bound = static_cast<int>(std::round(gkclipper.rho / std::tgamma(k + 1) + k));
    cout << "\nFinal spectral radius: " << gkclipper.rho << endl;
    cout << "Final clique number upper bound: " << gkclipper.rho / std::tgamma(k + 1) + k << ", " << upper_bound << endl;
    cout << "Final eigenvector: " << gkclipper.u << endl;
    cout << "Densest clique: " << gkclipper.get_top_n_indices(gkclipper.u, upper_bound) << endl;
    cout << "done" << endl << endl;
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time GkCLIPPER sec " << duration.count() * 1e-6 << std::endl;

    // // ------------From random graph-------------
    // std::cout << "Using gen_max_clique library" << std::endl;
    // // Example function or class usage from gmc.hpp
    // GMC::KUniformHypergraph<int> graph(3);

    // int num_nodes = 10;
    // std::vector<int> nodes;
    // for(int i=0;i<num_nodes;i++)
    // {
    //     nodes.push_back(i);
    //     graph.add_node(i);
    // }

    // // Create max clique
    // int max_clique_size = 5;
    // std::vector<int> max_clique;
    // std::shuffle(nodes.begin(), nodes.end(), std::default_random_engine(100021));
    // for(int i=0;i<max_clique_size; i++)
    // {
    //     max_clique.push_back(nodes[i]);
    // }
    // std::sort(max_clique.begin(), max_clique.end());

    // auto clique_edges = GMC::combinations_of_size(max_clique, 3);
    // for( auto e : clique_edges )
    // {
    //     graph.add_edge(e);
    // }

    // // Add random edges
    // auto all_edges = GMC::combinations_of_size(nodes, 3, true);
    // std::shuffle(all_edges.begin(), all_edges.end(), std::default_random_engine(0));

    // double density = 0.2;
    // std::vector< std::vector<int> > added_edges;
    // for(int i=0; i < density*all_edges.size();i++)
    // {
    //     added_edges.push_back(all_edges[i]);
    // }

    // graph.add_edges(added_edges);

    // // Find clique
    // GMC::MaxCliqueSolverExact<int> solver;
    // auto clique = solver.max_clique(graph, true);
    // GMC::print(std::cout, clique);
    // std::cout << "\n";

    return 0;
}
