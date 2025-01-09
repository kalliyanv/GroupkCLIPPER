#include "gen_max_clique/gmc.hpp"
#include <random>
#include <iostream>


#include "GkCLIPPER.h"

using namespace std::chrono;



// From Brendon's code src/gen_max_clique_tests/max_clique_exact_tests.cpp

int main() {

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
    GMC::KUniformHypergraph<int> graph(3);

    graph.add_nodes({0, 1, 2, 3, 4, 5, 6});
    graph.add_edge({0, 1, 4});
    graph.add_edge({0, 1, 3});
    graph.add_edge({0, 3, 4});
    graph.add_edge({1, 3, 4});
    graph.add_edge({0, 2, 4});
    graph.add_edge({4, 5, 6});
    graph.add_edge({3, 4, 6});

    // Find clique
    GMC::MaxCliqueSolverHeuristic<int> solver_heuristic;
    auto start = high_resolution_clock::now();
    auto clique_h = solver_heuristic.max_clique_heu(graph, true);
    auto stop = high_resolution_clock::now();

    GMC::print(std::cout, clique_h);
    std::cout << "\n";
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time heuristic microsec " << duration.count() << std::endl;

    GMC::MaxCliqueSolverExact<int> solver_exact;
    start = high_resolution_clock::now();
    auto clique_e = solver_exact.max_clique(graph, true);
    stop = high_resolution_clock::now();

    GMC::print(std::cout, clique_e);
    std::cout << "\n";
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time exact microsec " << duration.count() << std::endl;

    // My Gk
    std::cout << "\n";

    xt::xarray<double> M = {
    {{1, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 1, 1, 0, 0},
     {0, 0, 0, 0, 1, 0, 0},
     {0, 1, 0, 0, 1, 0, 0},
     {0, 1, 1, 1, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0}},

    {{0, 0, 0, 1, 1, 0, 0},
     {0, 1, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {1, 0, 0, 0, 1, 0, 0},
     {1, 0, 0, 1, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0}},

    {{0, 0, 0, 0, 1, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 1, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {1, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0}},

    {{0, 1, 0, 0, 1, 0, 0},
     {1, 0, 0, 0, 1, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 1, 0, 0, 0},
     {1, 1, 0, 0, 0, 0, 1},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 1, 0, 0}},

    {{0, 1, 1, 1, 0, 0, 0},
     {1, 0, 0, 1, 0, 0, 0},
     {1, 0, 0, 0, 0, 0, 0},
     {1, 1, 0, 0, 0, 0, 1},
     {0, 0, 0, 0, 1, 0, 0},
     {0, 0, 0, 0, 0, 0, 1},
     {0, 0, 0, 1, 0, 1, 0}},

    {{0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 1},
     {0, 0, 0, 0, 0, 1, 0},
     {0, 0, 0, 0, 1, 0, 0}},

    {{0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 0, 0, 0},
     {0, 0, 0, 0, 1, 0, 0},
     {0, 0, 0, 1, 0, 1, 0},
     {0, 0, 0, 0, 1, 0, 0},
     {0, 0, 0, 0, 0, 0, 1}}
    };

    // TODO: init properly
    size_t k = M.dimension();                       // Tensor order
    double n_temp = pow(M.size(), 1.0/k);           // Weird casting issues when type size_t, use temp double
    size_t n = static_cast< float > (n_temp);

    GkCLIPPER clipper(n, k, M);
    cout << "\nbegin solve" << endl;
    start = high_resolution_clock::now();
    clipper.solve();
    stop = high_resolution_clock::now();
    cout << "end solve" << endl;

    int upper_bound = static_cast<int>(std::round(clipper.rho / std::tgamma(k + 1) + k));
    cout << "\nFinal spectral radius: " << clipper.rho << endl;
    cout << "Final clique number upper bound: " << clipper.rho / std::tgamma(k + 1) + k << ", " << upper_bound << endl;
    cout << "Final eigenvector: " << clipper.u << endl;
    cout << "Densest clique: " << clipper.get_top_n_indices(clipper.u, upper_bound) << endl;
    cout << "done" << endl << endl;
    duration = duration_cast<microseconds>(stop - start);
    std::cout << "Time CLIPPER microsec " << duration.count() << std::endl;

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
