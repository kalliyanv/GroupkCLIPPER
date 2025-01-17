import clipperpy  # Assuming CLIPPER Python bindings are installed
import numpy as np
import time
from scipy.spatial.transform import Rotation as R

A = np.array([[1, 2, 3],[4, 5, 6],[7, 8, 9]])
x = np.array([1, 2, 3])
start = time.time()
print(np.tensordot(A ,x, axes=([0], [0])))
print("time tensordot", (time.time() - start))
# exit()

# Algorithm setup
# Euclidean distance invariant only works for lidar pointclouds, bc distances should be the same
iparams = clipperpy.invariants.EuclideanDistanceParams()  # Create invariant params
invariant = clipperpy.invariants.EuclideanDistance(iparams)  # Instantiate the Euclidean Distance invariant
params = clipperpy.Params()  # Create default CLIPPER params
clipper_obj = clipperpy.CLIPPER(invariant, params)  # Instantiate the CLIPPER object

# Data setup
# Create a target/model point cloud using numpy
model = np.zeros((3, 4))
model[:, 0] = [0, 0, 0]
model[:, 1] = [2, 0, 0]
model[:, 2] = [0, 3, 0]
model[:, 3] = [2, 2, 0]

# Apply a transformation to the data point cloud (similar to C++ Eigen::Affine3d)
rotation = R.from_euler('z', np.pi / 8)  # Rotate by 22.5 degrees around Z-axis
T_MD = np.eye(4)  # 4x4 transformation matrix
T_MD[:3, :3] = rotation.as_matrix()  # Rotation part
T_MD[:3, 3] = [5, 3, 0]  # Translation part

# Apply inverse transformation to the model to get the data point cloud
data1 = np.dot(np.linalg.inv(T_MD), np.vstack((model, np.ones((1, 4)))))[:3, :]

# Remove one point from the data (partial view)
data1 = np.array(data1[:, :3])

# Identify data association

# TODO: Replace with score_groupk_consistency
clipper_obj.score_pairwise_consistency(model, data1, clipperpy.utils.create_all_to_all(np.shape(model)[1], np.shape(data1)[1]))

# TODO: Replace findDenseClique to generalized solver
# Find the densest clique
start = time.time()
clipper_obj.solve()
end = time.time()

print("\nAffinity Matrix")
print(clipper_obj.get_affinity_matrix())
# print("\nConstraint Matrix")
# print(clipper_obj.get_constraint_matrix())
# object_methods = [method_name for method_name in dir(clipper_obj)
#                   if callable(getattr(clipper_obj, method_name))]
# print(object_methods)

# Retrieve the inliers (selected associations) and verify correctness
# First col: index of a point in the model point cloud
# Second col: corresponding index of a point in the data point cloud
inliers = clipper_obj.get_selected_associations()
print(f"Inliers: {inliers}")
print("CLIPPER Time sec", end - start)
exit()

# # Optionally, add checks similar to assertions in C++
# assert inliers.shape[0] == 3, "Expected 3 inliers"
# for i in range(inliers.shape[0]):
#     assert inliers[i, 0] == inliers[i, 1], "Expected matching associations"


# ----------------------------------------------------------------------------------------
# M = np.array([
#     [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2964, 0.0],
#     [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0138, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0016, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0747, 0.0],
#     [0.0, 0.0, 0.0, 1.0, 0.0, 0.0555, 0.2547, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0102, 0.0, 0.7715, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 1.0, 0.0063, 0.0, 0.3846, 0.0, 0.0003, 0.0014, 0.0, 0.0, 0.0, 0.0, 0.0063, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0555, 0.0063, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9927, 0.0, 0.0, 0.9722, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.2547, 0.0, 0.0, 1.0, 0.0, 0.0023, 0.0, 0.0, 0.8775, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.3846, 0.0, 0.0, 1.0, 0.0001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0023, 0.0001, 1.0, 0.7914, 0.0, 0.0, 0.0, 0.0617, 0.0, 0.0, 0.9938, 0.0, 0.0, 0.0007],
#     [0.0, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0, 0.0, 0.7914, 1.0, 0.0, 0.0, 0.0001, 0.0091, 0.0, 0.2503, 0.0222, 0.0549, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0014, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0008],
#     [0.0, 0.0, 0.0016, 0.0, 0.0, 0.0, 0.8775, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7007, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.9927, 0.0, 0.0, 0.0, 0.0001, 0.0, 0.0, 1.0, 0.0, 0.9978, 0.0, 0.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0138, 0.0, 0.0102, 0.0, 0.0, 0.0, 0.0, 0.0617, 0.0091, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9978, 0.0, 1.0, 0.0012, 0.0, 0.0, 0.0, 0.0074],
#     [0.0, 0.0, 0.0, 0.7715, 0.0063, 0.9722, 0.0, 0.0, 0.0, 0.2503, 0.0, 0.0, 0.0, 0.0, 0.0012, 1.0, 0.0026, 0.0217, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9938, 0.0222, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0026, 1.0, 0.0, 0.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0549, 0.0, 0.0, 0.0, 0.0003, 0.0, 0.0217, 0.0, 1.0, 0.0007, 0.0],
#     [0.2964, 0.0, 0.0747, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.7007, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0007, 1.0, 0.0],
#     [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0007, 0.0, 0.0008, 0.0, 0.0, 0.0, 0.0074, 0.0, 0.0, 0.0, 0.0, 1.0]
# ])

M = np.array([[1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
 [0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
 [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
 [0., 1., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
 [1., 0., 0., 0., 1., 0., 0., 0., 1., 1., 0., 0.],
 [0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
 [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
 [0., 0., 0., 0., 0., 1., 0., 1., 0., 0., 0., 0.],
 [1., 0., 0., 0., 1., 0., 0., 0., 1., 0., 0., 0.],
 [0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
 [0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 1., 0.],
 [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])

C = np.ones_like(M)
C = np.where(M > 0, C, M)
# print(C)

# Algorithm setup
# Euclidean distance invariant only works for lidar pointclouds, bc distances should be the same
iparams = clipperpy.invariants.EuclideanDistanceParams()  # Create invariant params
invariant = clipperpy.invariants.EuclideanDistance(iparams)  # Instantiate the Euclidean Distance invariant
params = clipperpy.Params()  # Create default CLIPPER params
clipper_obj = clipperpy.CLIPPER(invariant, params)  # Instantiate the CLIPPER object

# object_methods = [method_name for method_name in dir(clipper_obj)
#                   if callable(getattr(clipper_obj, method_name))]

# import time

# clipper_obj.set_matrix_data(M, C)
# start = time.time()
# clipper_obj.solve()
# end = time.time()
# print("CLIPPER Time", end - start)

# inliers = clipper_obj.get_selected_associations()
# print(f"Inliers: {inliers}")

# object_methods = [method_name for method_name in dir(clipper_obj)
#                   if callable(getattr(clipper_obj, method_name))]
# print(object_methods)