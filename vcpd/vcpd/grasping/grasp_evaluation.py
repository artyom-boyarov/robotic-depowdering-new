import numpy as np
import scipy.linalg

class GraspGeometry:
    com: np.ndarray
    contact1rot: np.ndarray
    contact2rot: np.ndarray
    contact1pos: np.ndarray
    contact2pos: np.ndarray
    base_pos: np.ndarray
    found: bool


def hat(v: np.ndarray):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

# Returns grasp matrix G
def get_G(grasp_geometry: GraspGeometry):
    P1 = np.zeros((6, 6))
    P1[0:3, 0:3] = np.eye(3)
    P1[3:6, 0:3] = hat(grasp_geometry.contact1pos - grasp_geometry.com)
    P1[3:6, 3:6] = np.eye(3)

    Rhat1 = scipy.linalg.block_diag(grasp_geometry.contact1rot, grasp_geometry.contact1rot)
    G_tilda_1 = ((Rhat1.T) @ (P1.T)).T

    P2 = np.zeros((6, 6))
    P2[0:3, 0:3] = np.eye(3)
    P2[3:6, 0:3] = hat(grasp_geometry.contact2pos - grasp_geometry.com)
    P2[3:6, 3:6] = np.eye(3)

    Rhat2 = scipy.linalg.block_diag(grasp_geometry.contact2rot, grasp_geometry.contact2rot)
    G_tilda_2 = ((Rhat2.T) @ (P2.T)).T

    G_tilda = np.hstack((G_tilda_1, G_tilda_2))

    # Soft finger model for H matrix
    H1 = np.zeros((4, 6))
    H1[0:3, 0:3] = np.eye(3)
    H1[3, 3] = 1

    H2 = H1

    H = scipy.linalg.block_diag(H1, H2)

    G = (H @ (G_tilda.T)).T
    return G

    


# Finds minimum singular value of grasp matrix G
# The larger this is the less like the grasp to fall into a  singular configuration
def min_singular_value_G(grasp_geometry: GraspGeometry):
    G = get_G(grasp_geometry)
    singular_values = scipy.linalg.svdvals(G)
    return np.min(singular_values)

# Finds volume of ellipsoid in wrench space
# The larger the value the better the grasp
def vol_ellipsoid_in_wrench(grasp_geometry: GraspGeometry):
    G = get_G(grasp_geometry)
    return np.sqrt(scipy.linalg.det(G @ (G.T)))

def null_space_G(grasp_geometry: GraspGeometry):
    G = get_G(grasp_geometry)
    return scipy.linalg.null_space(G)


def grasp_force(grasp_geometry: GraspGeometry):
    G_null_space = null_space_G(grasp_geometry)
    return abs(G_null_space[0])


def is_force_closure(grasp_geometry: GraspGeometry):
    G = get_G(grasp_geometry)
    G_null_space = scipy.linalg.null_space(G)
    NV = 6
    return G_null_space.size > 0 and (np.linalg.matrix_rank(G) == NV)

def get_minimum_force_friction(finger_normal: np.ndarray, friction_coeff: float, object_mass: float) -> float:
    angle = abs(np.dot(finger_normal, np.array([0,0,1])))

    w = 9.81 * np.cos(angle) * object_mass

    return w / (2*friction_coeff)

if __name__ == "__main__":
    min_sing_val = min_singular_value_G(
        np.array([0,0,0]), 
        np.array([[0,1,0], [0,0,1], [1,0,0]]),
        np.array([0,-2,0]).T,
        np.array([[1,0,0], [0,0,-1], [0,1,0]]),
        np.array([0,2,0]).T
    )