import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import combinations, permutations

np.set_printoptions(suppress=True)
np.random.seed(15)

# NOTE: Collinear poses can cause trilateration to fail because there will be more than one 
# possible solution for the landmark position
def straight_path(length):
    x = np.linspace(0, length, length + 1)
    y = np.zeros_like(x)
    return x, y

def circular_path(length):
    angles = np.linspace(0, 2 * np.pi, length, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    return x, y

def random_manhattan_path(length):
    x, y = [0], [0]
    for _ in range(length):
        # Define the possible choices as a NumPy array
        choices = np.array([(1, 0), (-1, 0), (0, 1), (0, -1)])

        # Randomly select one of the choices
        direction = choices[np.random.choice(len(choices))]
        x.append(x[-1] + direction[0])
        y.append(y[-1] + direction[1])
    return x, y

def make_landmarks(min, max, num_lm):
    landmarks = np.random.uniform(min, max, size=(num_lm, 2))
    return landmarks

def plot_path(x, y, title):
    # plt.figure(figsize=(6, 6))
    plt.grid()
    plt.plot(x, y, 'bo')
    plt.plot(x[0], y[0], 'ro', label='start')
    plt.plot(x[-1], y[-1], 'kx', label='end')
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim((-10, 10))    
    plt.ylim((-10, 10))    
    # plt.axis('equal')


def calculate_distance(point1, point2):
    """
    Calculate the Euclidean distance between two points.

    Args:
        point1 (tuple): Coordinates of the first point (x1, y1).
        point2 (tuple): Coordinates of the second point (x2, y2).

    Returns:
        float: The distance between the two points.
    """
    x1, y1 = point1
    x2, y2 = point2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def trilaterate_landmark(poses, ranges):
    """
    Estimate the position of a landmark given 3 robot poses and corresponding ranges.
    
    Args:
        poses (list of tuples): List of robot poses [(x1, y1), (x2, y2), (x3, y3)].
        ranges (list of floats): List of measured ranges [r1, r2, r3].

    Returns:
        tuple: Estimated position of the landmark (x, y).
    """
    # Extract robot poses and range measurements
    x1, y1 = poses[0]
    x2, y2 = poses[1]
    x3, y3 = poses[2]
    r1, r2, r3 = ranges

    # Formulate the equations for trilateration
    A = np.array([
        [2 * (x2 - x1), 2 * (y2 - y1)],
        [2 * (x3 - x1), 2 * (y3 - y1)],
    ])
    
    b = np.array([
        r1**2 - r2**2 - x1**2 + x2**2 - y1**2 + y2**2,
        r1**2 - r3**2 - x1**2 + x3**2 - y1**2 + y3**2
    ])

    # Solve the linear system to estimate the landmark position
    landmark_position = np.linalg.solve(A, b)
    return np.array(landmark_position)

def mahalanobis_distance(x, mean, cov):
    """
    Compute the Mahalanobis distance between a vector and a distribution.
    d^2 = (x - μ)^T * Σ^-1 * (x - μ)
    
    Args:
        x (numpy array): The data point (vector) to measure.
        mean (numpy array): The mean of the distribution.
        cov (numpy array): The covariance matrix of the distribution.
    
    Returns:
        float: The Mahalanobis distance.
    """
    # diff = x - mean
    # cov_inv = np.linalg.inv(cov)
    # distance = np.sqrt(np.dot(np.dot(diff.T, cov_inv), diff))

    distance = abs(x - mean) / np.sqrt(cov)
    return distance

def triangulate_range(poses_abcd, r_abc):

    p_d = poses_abcd[3]

    return np.linalg.norm(trilaterate_landmark(poses_abcd[0:3], r_abc) - p_d)

def range_consistency(poses_abcd, r_abcd, sigma_sq):
    '''
    gamma is the consistency threshold
    '''
    # For now, use Mahalanobis distance
    # TODO use a consistency metric that takes nongaussian range measurements into accout

    r_d = r_abcd[3]
    gamma = 3 # units of standard deviation

    # 1 if measurement is consistent, 0 if not consistent
    # TODO are x and mu correct in this case?
    pred_range = triangulate_range(poses_abcd, r_abcd[0:3])
    dist = mahalanobis_distance(pred_range, r_d, sigma_sq)
    # print("mahal dist calc in function", dist)
    if dist <= gamma:
        return 1
    else:
        return 0

def plot_all(path_choice, path_length, landmarks):
    # Make path
    if path_choice == 1:
        x, y = straight_path(path_length)
        plot_path(x, y, "Straight Path")
    elif path_choice == 2:
        x, y = circular_path(path_length)
        plot_path(x, y, "Circular Path")
    elif path_choice == 3:
        x, y = random_manhattan_path(path_length)
        plot_path(x, y, "Random Manhattan Path")
    else:
        print("Invalid choice!")

    plt.plot(landmarks[:,0], landmarks[:,1], "k^")

def get_M():

    plt.figure(0)

    choice = 2
    length = 5
    num_landmarks = 1

    # Make path
    if choice == 1:
        x, y = straight_path(length)
        plot_path(x, y, "Straight Path")
    elif choice == 2:
        x, y = circular_path(length)
        plot_path(x, y, "Circular Path")
    elif choice == 3:
        x, y = random_manhattan_path(length)
        plot_path(x, y, "Random Manhattan Path")
    else:
        print("Invalid choice!")

    # Set pose indices
    pose_indices = np.arange(len(x))

    # Make landmarks
    landmarks = make_landmarks(-5, 5, num_landmarks)

    # Get ranges
    sigma_sq = 0.01                    # Variance
    sigma = math.sqrt(sigma_sq)     # Standard deviation
    outlier_set = set()
    prob_outlier = 0.2
    dists_actual_all = {}
    dists_noisy_all = {}
    for lm_coord in landmarks:
        dists = {}
        dists_noisy = {}
        for x_coord, y_coord in zip(x,y):
            robot_coord = np.array([x_coord, y_coord])
            dist = calculate_distance(robot_coord, lm_coord)

            # Randomly add outlier ranges
            if np.random.rand() < prob_outlier:
                outlier_set.add(tuple(robot_coord))
                dist_noisy = dist + np.random.uniform(sigma*4, sigma*6) # 99% inliers should be within 3 sigma
                print("")
                print("OUTLIER ADDED")
                print("curr coord", robot_coord)
                print("dist actual", dist)
                print("dist noisy", dist_noisy, " (outlier)")
                print("mahal dist", mahalanobis_distance(np.array([dist_noisy]), np.array([dist]), sigma_sq))

            # Corrupt with gaussian noise
            else:
                dist_noisy = dist #+ np.random.normal(0, sigma)
                print("")
                print("NOT AN OUTLIER")
                print("curr coord", robot_coord)
                print("dist actual", dist)
                print("dist noisy", dist_noisy)
                print("mahal dist", mahalanobis_distance(np.array([dist_noisy]), np.array([dist]), sigma_sq))

            dists[tuple(robot_coord)] = dist
            dists_noisy[tuple(robot_coord)] = dist_noisy
        dists_actual_all[tuple(lm_coord)] = dists
        dists_noisy_all[tuple(lm_coord)] = dists_noisy

    
    print("Num outliers: ", len(outlier_set))

    # Allocate memory for affinity tensor. TODO use consistency, or scaled consistency?
    M = np.zeros((length, length, length, length))
    
    # Get all combinations or 4 measurements TODO over entire path or only window of measurements? Incrementally add measurements?
    for lm_coord in dists_noisy_all:
        print("\nlandmark pose actual", lm_coord)
        pose_combos = list(combinations(dists_noisy_all[lm_coord], 4))
        idx_combos = list(combinations(pose_indices, 4))

        # Iterate through each combo of 4
        for pose_combo, idx_combo in zip(pose_combos, idx_combos):

            pose_perms = list(permutations(pose_combo, 4))
            idx_perms = list(permutations(idx_combo, 4))
            
            for pose_perm, idx_perm in zip(pose_perms, idx_perms):
                print("")

                # Calculate consistency of one point against the other three. TODO: Check each of the four?
                poses = np.array(pose_perm)
                ranges_actual = np.array([dists_actual_all[lm_coord][pt] for pt in pose_perm])
                ranges_noisy = np.array([dists_noisy_all[lm_coord][pt] for pt in pose_perm])
                ranges_test = ranges_noisy #np.array([ranges_actual[0], ranges_actual[1], ranges_actual[2], ranges_noisy[3]])
            
                consistency = range_consistency(poses, ranges_test, sigma_sq)
                pred_range = triangulate_range(poses, ranges_test[0:3])
                mahal_dist = mahalanobis_distance(pred_range, ranges_test[3], sigma_sq)
                score = math.exp(-(1/2)*mahal_dist**2 / sigma_sq)

                # Check if outlier present
                outlier_present = 0
                for i in pose_perm:
                    if i in outlier_set:
                        outlier_present = 1
                if outlier_present:
                    print("Outlier meas present")
                else:
                    print("Not outlier meas")

                # Results
                print(idx_perm)
                print("consistency", consistency)
                print("score", score)
                print("est landmark", trilaterate_landmark(poses, ranges_test[0:3]), "   actual landmark", lm_coord)
                print("pred range", pred_range, "  noisy range", ranges_noisy[3], "   actual range", ranges_actual[3])
                print("mahal dist", mahal_dist)

                # Update affinity matrix
                M[idx_perm[0], idx_perm[1], idx_perm[2], idx_perm[3]] = score

    #             # Plot intersecting circles
    #             plot_all(choice, length, landmarks)
    #             ax = plt.gca()
    #             for i in range(3):
    #                 plt.plot(poses[i,0], poses[i,1], 'rx')
    #                 circle = plt.Circle((poses[i,0], poses[i,1]), ranges_test[i], fill=False)
    #                 ax.add_patch(circle)
    #                 plt.draw()

    #             # Range we are checking consisteny of 
    #             circle = plt.Circle((poses[i,0], poses[i,1]), ranges_test[3], fill=False, color='green')
    #             ax.add_patch(circle)
    #             plt.draw()

    #             # Predicted landmark loc
    #             est_landmark = trilaterate_landmark(poses, ranges_test[0:3])
    #             plt.plot(est_landmark[0], est_landmark[1], 'r^')

    #             ax.set_aspect('equal')
    #             plt.grid()
    #             plt.waitforbuttonpress()
    #             plt.clf()
    #             plt.grid()

    #             if not plt.fignum_exists(0):
    #                 plt.close()
    #                 exit()



    # plt.xlim(0, 1)
    # print("Affinity Matrix")
    # print(M)
    # plt.show()

    # Add identity
    for i in range(length):
        M[i, i, i, i] = 1

    # TODO FIX THIS INCORRECT INDEX, MESSY FIX FOR NOW
    for perm in permutations([0, 2, 4], 3):
        M[perm[0], perm[1], perm[2], 1] = 0
    # exit()
    return M

if __name__ == "__main__":
    print(get_M())
