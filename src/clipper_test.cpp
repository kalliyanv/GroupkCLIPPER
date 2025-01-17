#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <clipper/clipper.h>
#include <clipper/utils.h>

int main() {

    //
    // Algorithm setup
    //

    // instantiate the invariant function that will be used to score associations
    clipper::invariants::EuclideanDistance::Params iparams;
    clipper::invariants::EuclideanDistancePtr invariant =
                std::make_shared<clipper::invariants::EuclideanDistance>(iparams);

    clipper::Params params;
    clipper::CLIPPER clipper(invariant, params);

    //
    // Data setup
    //

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

    //
    // Identify data association
    //

    // an empty association set will be assumed to be all-to-all
    clipper.scorePairwiseConsistency(model, data);

    // find the densest clique of the previously constructed consistency graph
    clipper.solve();

    

    return 0;

}