#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <map>

using namespace std;
#include <Eigen/Dense>

using namespace Eigen;

// Function to compute the least-squares trilateration in 2D
Vector2d leastSquaresTrilateration2D(const MatrixXd& anchors, const VectorXd& distances) {
    int N = anchors.rows();
    if (N < 3) {
        throw std::invalid_argument("At least three anchors are required for a 2D position estimate.");
    }
    
    // Construct matrix A and vector b
    MatrixXd A(N - 1, 2);
    VectorXd b(N - 1);
    for (int i = 1; i < N; ++i) {
        A.row(i - 1) = 2 * (anchors.row(i) - anchors.row(0));
        b(i - 1) = distances(0) * distances(0) - distances(i) * distances(i)
                 - anchors.row(0).squaredNorm() + anchors.row(i).squaredNorm();
    }
    
    // Solve the least-squares problem Ax = b
    Vector2d position = A.colPivHouseholderQr().solve(b);
    return position;
}

// Function to compute the least-squares trilateration
Vector3d leastSquaresTrilateration3D(const MatrixXd& anchors, const VectorXd& distances) {
    int N = anchors.rows();
    if (N < 4) {
        throw std::invalid_argument("At least four anchors are required for a 3D position estimate.");
    }
    
    // Construct matrix A and vector b
    MatrixXd A(N - 1, 3);
    VectorXd b(N - 1);
    for (int i = 1; i < N; ++i) {
        A.row(i - 1) = 2 * (anchors.row(i) - anchors.row(0));
        b(i - 1) = distances(0) * distances(0) - distances(i) * distances(i)
                 - anchors.row(0).squaredNorm() + anchors.row(i).squaredNorm();
    }
    
    // Solve the least-squares problem Ax = b
    Vector3d position = A.colPivHouseholderQr().solve(b);
    return position;
}

// double h(vector<double> Xabcd, double Zabc) {

// }

// double consistency(double z_pred, double z_actual, vector<vector> Sigma) {
//     // Mahalanobis consistency metric
// }

int main() {
    // -------------Load data-------------

    string filename = "/home/kalliyanlay/Documents/BYU/research/CAAMS/goats-data/data/goats_14/goats_14_6_2002_15_20.csv";
    ifstream file(filename);


    if (!file.is_open()) {
        cerr << "Could not open the file." << endl;
        return 1;
    }

    string line;
    // Read the header line (optional)
    getline(file, line);

    // Determine the number of columns based on the header or first data row
    stringstream ss(line);
    string token;
    int num_cols = 0;
    while (getline(ss, token, ',')) {
        num_cols++;
    }
    cout << "num cols " << num_cols << endl;

   // Create vectors to store each column
    vector<vector<double>> columns(num_cols);
    
    // Rewind the file to the beginning to read data
    file.clear();
    file.seekg(0, ios::beg);
    
    // Skip the header line again
     getline(file, line);

    // Read data rows
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        int col_index = 0;

        while (getline(ss, value, ',')) {
           columns[col_index].push_back(stod(value));
            col_index++;
        }
    }
    
    file.close();
    
    std::map<std::string, int> col_names = {
        {"time", 0},
        {"insX", 1},
        {"insY", 2},
        {"insZ", 3},
        {"insR", 4},
        {"insP", 5},
        {"insY", 6}, 
        {"insVX", 7},
        {"insVY", 8},
        {"insVZ", 9},
        {"r1", 13},         // Range from each of 4 beacons
        {"r2", 14},
        {"r3", 15},
        {"r4", 16},
        {"fr1", 17},        // Filtered range
        {"fr2", 18},
        {"fr3", 19},
        {"fr4", 20}
    };

    // -------------Get all range data-------------
    // Get indices of inlier and outlier measurements
    string r = "r3";
    string fr = "fr3";
    vector<int> idxs_inliers;
    vector<int> idxs_outliers;
    // cout << col_names["r1"] << endl;
    int num_rows = columns[col_names[r]].size();
    for(int i = 0; i < num_rows; i++) {
        // cout << columns[col_names["r1"]][i] << " " << columns[col_names["fr1"]][i] << endl;
        if(columns[col_names[r]][i] != columns[col_names[fr]][i]) {
            idxs_outliers.push_back(i);
        } else {
            idxs_inliers.push_back(i);
        }
    }
    cout << "num inlier " <<  idxs_inliers.size() << endl;
    cout << "num outlier " << idxs_outliers.size() << endl;
    // // -------------Get subset of range data-------------
    int num_inliers = 300;
    int num_outliers = 40;
    
    // Randomly sort indices of each, then grab top n
    
    // -------------Build matrix using consistency metric-------------
    // Consistency metric
    double gamma = 3.0; // Threshold

    // Define known anchor positions (each row is an (x, y, z) coordinate)
    MatrixXd anchors(num_inliers, 3);
    VectorXd distances(num_inliers);
    for (int i = 0; i < num_inliers; i++) {
        anchors(i, 0) = columns[col_names["insX"]][idxs_inliers[i]];
        anchors(i, 1) = columns[col_names["insY"]][idxs_inliers[i]];
        anchors(i, 2) = columns[col_names["insZ"]][idxs_inliers[i]];
        distances(i) = columns[col_names[r]][idxs_inliers[i]];
    }
    
    Vector3d position = leastSquaresTrilateration3D(anchors, distances);
    std::cout << "Estimated position: " << position.transpose() << std::endl;
  
    
    // -------------Run GkCM on range data-------------
    
    // -------------Build factor graph-------------
    
    // -------------Run factor graph-------------
    
    // -------------Plot ground truth-------------

    return 0;
}

