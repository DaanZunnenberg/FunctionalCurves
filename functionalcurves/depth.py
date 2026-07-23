import numpy as np
import scipy
from ctypes import *
# from depth.multivariate.Depth_approximation import depth_approximation
# from depth.multivariate import halfspace
import sys, os, glob
import platform
# from depth.multivariate.import_CDLL import libExact,libApprox
from typing import Any, Callable
from tqdm import tqdm
from statsmodels.tsa.vector_ar.var_model import VARProcess

import warnings
warnings.filterwarnings(action = 'ignore')

import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 12)

__VERSION__: str = '1.1.1'
__UPDATES__: str = """
    12/02/2025:
    NOTE: ...
"""

halfspace: Callable = lambda _ = 0, *args, **kwargs : NotImplementedError()

class Depth(object):
    def __init__(self, X: Any, dates: Any = None) -> None:
        self.X = X
        self.dates = dates
 
    def halfspace(self, *args, **kwargs) -> list: return halfspace(*args, **kwargs)
import numpy as np

unit_vector: Callable = lambda *V, **kwargs : [np.float64(v / np.linalg.norm(v)) for v in V] 

def rad(v1, v2, standardize: bool = False, norm: bool = True, handle_complex: bool = True):
    """Calculate the directed angle(s) between points on a unit circle in radians.
    
    v1: (N, 2) array of N points or a single (2,) point
    v2: (N, 2) array of N points or a single (2,) point
    Returns: Angles in [0, 2π] as an (N,) array or a single float.
    """
    
    if norm: v1, v2 = unit_vector(v1, v2, handle_complex = handle_complex)
    elif handle_complex: v1, v2 = np.float64(v1), np.float64(v2)
    # Convert inputs to numpy arrays
    v1, v2 = np.array(v1), np.array(v2)
    
    # Compute dot product and determinant (cross product in 2D)
    dot_product = np.sum(v1 * v2, axis=-1)  # Handles both (N,2) and (2,)
    det = v1[..., 0] * v2[..., 1] - v1[..., 1] * v2[..., 0]  # Cross product determinant

    # Use atan2 to get angles in the range [-π, π]
    theta = np.arctan2(det, dot_product)

    # Convert negative angles to [0, 2π]
    try: theta[theta < 0] += 2 * np.pi
    except: theta = (theta < 0) * (2 * np.pi) + theta

    return theta - (standardize * np.pi / 2)

def rad_dec(v1, v2 = np.array([1,0]), norm: bool = True, **kwargs) -> np.float64:
    """Calculates the angle between vectors"""
    raise DeprecationWarning
    unit_vector: Callable = lambda *V, **kwargs : [v / np.linalg.norm(v) for v in V] 
    if norm: v1, v2 = unit_vector(v1, v2)
    angle: np.float64 = np.arctan2(np.cross(v1, v2), np.dot(v1, v2))
    try:
        if angle < 0: return 2 * np.pi + angle
        else: return angle
    except ValueError as e:
        return np.array([(u < 0) * (2 * np.pi) + u for u in angle])

def rad_wu(*args, **kwargs): return rad(*args, v2 = -np.array([1,0]), **kwargs)

def GaussianDepth(*args, 
                  mean: Any = [0,0], 
                  cov: Any = [[1,0], [0,1]], 
                  X0: Any = [0,1],
                  **kwargs) -> tuple:
    """
    Calculates and returns the minimal direction and halfspace depth at point X0 of a bivariate normal with mean:mean and cov:cov
    """
    mean = np.array(mean)
    cov = np.matrix(cov)
    X0 = np.array(X0)
    
    X0_normalised: Any = np.float64((X0 - mean) @ scipy.linalg.sqrtm(np.linalg.inv(cov)))
    U0_normalised: Any = np.float64(X0_normalised / np.linalg.norm(X0_normalised))
    U0_denormalised: Any = np.float64(scipy.linalg.sqrtm(cov) @ U0_normalised)

    minimal_direction: np.ndarray = rad([1,0], U0_denormalised)
    minimal_vector: np.ndarray = U0_denormalised / np.linalg.norm(U0_denormalised)

    univariate_mean: float = minimal_vector @ mean
    univariate_var: float = (minimal_vector @ cov @ minimal_vector.T).item()
    """
    H(x,u):={y:u'(y-x)>=0}. Let Y=u'X, then y~N(u'mu,u' Sigma u) and P(H(X,u))
    """
    Y = scipy.stats.norm(loc = univariate_mean, scale = univariate_var ** .5)

    minimal_depth = 1 - Y.cdf((minimal_vector @ X0 - univariate_mean) / np.sqrt(univariate_var))
    
    return minimal_depth, minimal_direction, minimal_vector

# GaussianDepth()

# mean = [1,2]
# cov = [[4,2], [-3, 5]]
# X0 = [.7,-2]

# X0_translated = np.float64((X0 - mean) @ scipy.linalg.sqrtm(np.linalg.inv(cov)))

# x, y = np.mgrid[-2:2:.01, -4:2:.01]
# pos = np.dstack((x, y))
# rv = scipy.stats.multivariate_normal([0,0], [[1,0], [0,1]])
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# ax2.contourf(x, y, rv.pdf(pos), levels = 25)
# ax2.scatter(X0_translated[0], X0_translated[1], color = 'red', marker = 'x')
# ax2.arrow(X0_translated[0], X0_translated[1], X0_translated[0], X0_translated[1], head_width=0.1, head_length=0.1, width = 0.02, color = 'red', label = 'True', alpha = .8)

# X0_translated = np.float64((X0 - mean) @ scipy.linalg.sqrtm(np.linalg.inv(cov)))

# U0_norm = np.float64(scipy.linalg.sqrtm(cov) @ (X0_translated / np.linalg.norm(X0_translated)))

# x, y = np.mgrid[-5:8:.01, -5:8:.01]
# pos = np.dstack((x, y))
# rv = scipy.stats.multivariate_normal(mean, cov)
# fig2 = plt.figure()
# ax2 = fig2.add_subplot(111)
# ax2.contourf(x, y, rv.pdf(pos), levels = 25)
# ax2.scatter(X0[0], X0[1], color = 'red', marker = 'x')
# ax2.arrow(X0[0], X0[1], U0_norm[0], U0_norm[1], head_width=0.1, head_length=0.1, width = 0.02, color = 'red', label = 'True', alpha = .8)



class Estimator(object):
    """
    Empirical estimation of Tukey's depth and direction
    """
    def __init__(self,
                 X: Any,
                 X0: tuple = (0,0), 
                 skip: int = False, 
                 norm_weights: Any = np.array([1,1]),
                 method: str = 'point_wise',
                 step: float = 0.0,
                 num: int = 10 ** 3
                ) -> None:
        self.X = X
        self.X0 = X0
        self.skip = skip
        self.norm_weights = norm_weights
        self.method = method
        self.num = num
        self.step = step
        self.Rot_mapper: Callable = lambda angle: np.matrix([[np.cos(angle), -np.sin(angle)],[np.sin(angle),np.cos(angle)]])

    def main(self) -> list[float]:
        depth_array: list[float] = []
        direction_array: list[float] = []

        if self.skip: raise NotImplementedError()
        else:
            # Center each observation relative to X0
            X_shift = self.X - self.X0

            # Project each point on the unit circle centered at X0
            X_proj = X_shift / np.linalg.norm(X_shift, axis = 1).T[:, np.newaxis]

            if self.method == 'deg': 
                # Array of relative angles
                if self.step: angles = np.arange(-np.pi, np.pi, np.step)
                else:
                    angles: np.array = np.linspace(-np.pi, np.pi, num = self.num)
                for index, angle in enumerate(angles):
                    """
                    For each theta in (-pi, pi], calculate the depth with the directional vector u(theta)
                    """
                    # Get the unit vector for the angle
                    x0, y0 = np.sqrt(1 - (np.sin(angle) ** 2) ) * np.sign(np.cos(angle)), np.sin(angle)
                    unit_vector_i: np.ndarray = np.array([x0, y0])

                    # Calculate the relative angle
                    angles_rel: np.array = rad(X_proj, unit_vector_i)

                    # Count the angles that are within pi/2 from Xi and devide by the sample size to obtain the estimated depth with direction to Xi
                    depth_array.append(np.where(np.abs(angles_rel - np.pi) >= np.pi / 2, 1, 0).mean())

                    # The directional array is simply the direction of Xi relative to (1,0)
                    direction_array.append(rad(np.array([[1,0]]), unit_vector_i)[0])

            elif self.method == 'point_wise':
                for index, sample in enumerate(X_proj):
                    """
                    For each sample Xi in X, calculate the angles theta(Xi, X) relative to each other obervation
                    """

                    # Array of relative angles
                    angles: np.array = rad(X_proj, sample)

                    # Count the angles that are within pi/2 from Xi and devide by the sample size to obtain the estimated depth with direction to Xi
                    depth_array.append(np.where(np.abs(angles - np.pi) >= np.pi / 2, 1, 0).mean())

                    # The directional array is simply the direction of Xi relative to (1,0)
                    direction_array.append(rad(np.array([[1,0]]), sample)[0])
            
            else:
                raise ValueError()
        
        # Take the minimal depth
        self.minimal_depth: float = min(depth_array)

        # Find the direction associated to the minimal depth
        self.minimal_direction: float = direction_array[depth_array.index(self.minimal_depth)]

        # Return the minimal depth, minimal direction, angles, depths and directions
        return self.minimal_depth, self.minimal_direction, angles, depth_array, direction_array
    
    def Angle_to_vetor(self, angle, normalise: bool = True, **kwargs) -> np.ndarray:
        if self.X0[1] < 0: angle = angle
        if not normalise: return np.dot(self.Rot_mapper(angle, **kwargs), np.array([1,0]))
        else: 
            estimated_minimal_vector = np.dot(self.Rot_mapper(angle, **kwargs), np.array([1,0]))
            return estimated_minimal_vector / np.linalg.norm(estimated_minimal_vector)

def TDE(X: Any, 
        X0: tuple = (0,0), 
        skip: int = False, 
        norm_weights: Any = np.array([1,1])
        ) -> tuple:
    """
    Estimates the Tukey depth using couting
    """
    raise DeprecationWarning()
    if isinstance(type(X0), tuple): X0 = np.array(X0)

    # Define the depth_array and direction_array to store the depth resp. direction of each point (direction) relative to X0
    depth_array: list[float] = []
    direction_array: list[float] = []

    if skip: raise NotImplementedError()
    else:
        # Center each observation relative to X0
        X_shift = X - X0

        # Project each point on the unit circle centered at X0
        X_proj = X_shift / np.linalg.norm(X_shift, axis = 1).T[:, np.newaxis]
        
        for index, sample in enumerate(X_proj):
            """
            For each sample Xi in X, calculate the angles theta(Xi, X) relative to each other obervation
            """

            # Array of relative angles
            angles: np.array = angle_between_points(X_proj, sample)

            # Count the angles that are within pi/2 from Xi and devide by the sample size to obtain the estimated depth with direction to Xi
            depth_array.append(np.where(np.abs(angles) <= np.pi / 2, 1, 0).mean())

            # The directional array is simply the direction of Xi relative to (1,0)
            direction_array.append(angle_between_points(np.array([[1,0]]), sample)[0])
    
    # Take the minimal depth
    TD: float = min(depth_array)

    # Find the direction associated to the minimal depth
    DR: float = direction_array[depth_array.index(TD)]

    return TD, DR, angles, depth_array, direction_array

class Analytic_Depth(object):
    """
    Returns the analytical halfspace depth and direction based on the VAR parameters.
    """
    def __init__(self, 
                X: np.matrix = None, 
                A0: np.matrix = np.zeros((2,1)), 
                A1: np.matrix = None, 
                S0: np.matrix = np.eye(2), 
                DL: tuple[tuple[float, ...], tuple[float, ...]] = (),
                X0: tuple[float,float] = np.array([0,0]),
                norm: bool = False,
                config: dict = None) -> None:
        self.X = X
        self.A0 = A0
        self.A1 = A1
        self.S0 = S0
        self.X0 = X0
        self.config = config
        self.norm = norm
        self.__Construct_normal
        self.__Scale
    
    @property
    def __norm(self) -> None:
        if self.norm:
            self.X0 = ...

    @property
    def __Construct_normal(self, **kwargs) -> None:
        """
        Mean and variance of a stationary order one vector autoregressive process
        """
        p1, p2 = self.A1.shape
        q1, q2 = (p1 ** 2, 1)
        if self.X.shape[1] > 2: 
            raise ValueError()
        else: 
            self.mean: np.ndarray = np.linalg.inv(np.eye(p1) - self.A1) @ self.A0
            self.variance = (np.linalg.inv(np.eye(q1) - np.kron(self.A1,self.A1)) @ self.S0.reshape((q1, q2))).reshape(p1,p2)
    
    @property
    def __Scale(self) -> None:
        """
        Gives the directional vector of X0 along with the angle
        """
        if np.sum(self.X0 ** 2) == 0: 
            try: raise ArgumentError('(0,0) prohibited')
            except ArgumentError as e: 
                print(e)
                self.dir = None

        # Normalised point with respect to the underlying stationary measure
        self.X0_normalised = (self.X0 - self.mean.T[0]) @ scipy.linalg.sqrtm(np.linalg.inv(self.variance))
        
        # Directional vector equals normalised X0 
        self.dir_vect_norm: np.ndarray = self.X0_normalised  / np.linalg.norm(self.X0_normalised)
        self.dir_norm = rad([1,0], self.dir_vect_norm)
        
        # Directional vector on the ellipse 
        self.X0_denorm = self.dir_vect_norm @ scipy.linalg.sqrtm(self.variance)
        self.X0_denorm_directional_vector = self.dir_vect_norm @ scipy.linalg.sqrtm(np.linalg.inv(self.variance))
        self.X0_denorm_direction_norm = rad([1,0], self.X0_denorm_directional_vector)

        # self.dir_vect = scipy.linalg.sqrtm(self.variance) @ (self.dir_vect_norm + self.mean.T[0])

        # # Angle between ellipse directional vector and (1,0)
        # self.dir = angle_between_points(np.array([[1,0]]), self.dir_vect)[0]

        # Project back onto the ellipse
        self.dir_ellipse = self.X0_denorm
        self.dir = self.X0_denorm_direction_norm
        self.dir_vect_ellipse = self.X0_denorm_directional_vector
        self.__filter_complex

    @property
    def __filter_complex(self) -> None:
        for name, val in vars(self).items():
            try: 
                if np.iscomplex(val).any(): self.__setattr__(name, np.float64(val))
            except: 
                if np.iscomplex(val): self.__setattr__(name, np.float64(val))

    def TD_analytic(self) -> ...:
        """
        Gives the exact Tukey depth if the data has an unconditional bivariate Gaussian law. 
        """
        if self.X.shape[1] > 2: 
            raise ValueError()
        else: 
            mean: np.ndarray = self.mean
            variance: np.ndarray = self.variance
        
        # Call X0 and the associated minimal directional vector at X0
        X0: np.ndarray = self.X0
        minimal_direction: np.ndarray =  self.dir_vect_ellipse / np.linalg.norm(self.dir_vect_ellipse)
        
        # The parameters of the normal variable u'X.
        univariate_mean: float = minimal_direction @ mean
        univariate_var: float = minimal_direction @ variance @ minimal_direction.T
        """
        H(x,u):={y:u'(y-x)>=0}. Let Y=u'X, then y~N(u'mu,u' Sigma u) and P(H(X,u))
        """

        Y = scipy.stats.norm(loc = univariate_mean, scale = univariate_var ** .5)

        Tukey_depth_true = 1 - Y.cdf((minimal_direction @ X0 - univariate_mean) / np.sqrt(univariate_var))
        
        return Tukey_depth_true, minimal_direction


if __name__ == '__main__':
    X0 = np.array((0, 2))           # Depth point
    est_dir, est_dep, test = [], [], [] # Estimated lists
    est_dir2, est_dep2, test = [], [], [] # Estimated lists
    N_samples, jump = 8_000, 200         # Simulation parameters

    AR_coef: np.ndarray = np.array([[[0.95, 0.02], [0.55, 0.25]]])  # VAR(1) parameters
    VAR = VARProcess(coefs = AR_coef, coefs_exog = None, sigma_u = np.eye(2))
    X = VAR.simulate_var(steps = N_samples)
    samples = range(jump, N_samples, jump)

    True_depth = Analytic_Depth(X = X, A1 = AR_coef[0], S0 = np.eye(2), X0 = X0)
    mean, variance = True_depth.mean, True_depth.variance

    true_directional_vector = unit_vector(True_depth.X0_denorm_directional_vector)[0]
    true_directional_angle = True_depth.dir
    true_depth, true_direction = True_depth.TD_analytic()

    for n in tqdm(samples):
        XN = X[:n]
        depth_estimator = Estimator(X = XN, X0 = X0, method = 'deg')
        depth_estimator2 = Estimator(X = XN, X0 = X0, method = 'point_wise')

        estimated_depth, estimated_direction, relative_angles, relative_depths, relative_directions = depth_estimator.main()
        estimated_depth2, estimated_direction2, relative_angles2, relative_depths2, relative_directions2 = depth_estimator2.main()

        est_dir.append(estimated_direction)
        est_dep.append(estimated_depth)

        est_dir2.append(estimated_direction2)
        est_dep2.append(estimated_depth2)

    # Graph statistics
    fig, axs = plt.subplots(3, 1, layout='constrained')
    axs[0].plot(samples, est_dep, lw = 1, c = 'C0', marker = 'x', label = 'Method: deg')
    axs[0].plot(samples, est_dep2, lw = 1, c = 'C1', marker = 'x', label = 'Method: point_wise')
    axs[0].axhline(true_depth, color = 'black', linestyle = 'dashed', linewidth = 2, label = 'True Tukey depth')
    axs[0].set_xlabel('Sample size')
    axs[0].set_ylabel('Depth')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(samples, est_dir, lw = 1, c = 'C0', marker = 'x', label = 'Method: deg')
    axs[1].plot(samples, est_dir2, lw = 1, c = 'C1', marker = 'x', label = 'Method: point_wise')
    axs[1].axhline(true_directional_angle, color = 'black', linestyle = 'dashed', linewidth = 2, label = 'True minimal angle')
    axs[1].set_xlabel('Sample size')
    axs[1].set_ylabel('Direction')
    axs[1].grid(True)
    axs[1].legend()
    # axs[1].set_ylim([true_TP - np.pi / 8, true_TP + np.pi / 8])

    axs[2].plot(X[:,0], lw = 1, c = 'C0', marker = 'x')
    axs[2].plot(X[:,1], lw = 1, c = 'C1', marker = 'x')
    axs[2].set_xlabel('Index')
    axs[2].set_ylabel('X')
    axs[2].grid(True)

    plt.show()

    

    # Graph minimal direction
    X_shifted = XN - X0
    X_projected = X_shifted / np.linalg.norm(X_shifted, axis = 1).T[:, np.newaxis]

    estimated_minimal_vector = depth_estimator.Angle_to_vetor(estimated_direction, normalise = True)
    estimated_minimal_vector2 = depth_estimator2.Angle_to_vetor(estimated_direction2, normalise = True)

    plt.scatter(XN[:,0], XN[:,1], c = relative_depths2)
    plt.scatter(X0[0], X0[1], color = 'red', s = 100)
    plt.scatter(mean[0][0], mean[1][0], color = 'red', s = 100, marker = 'x')
    # plt.arrow(X0[0], X0[1], 7 * a.item(0), 7 * a.item(1), head_width=0.8, head_length=0.8, width = 0.2, color = 'red', label = 'True')
    plt.arrow(X0[0], X0[1], 7 * true_directional_vector[0], 7 * true_directional_vector[1], head_width=0.8, head_length=0.8, width = 0.2, color = 'black', label = 'True', alpha = .5)
    plt.arrow(X0[0], X0[1], 7 * estimated_minimal_vector.item(0), 6 * estimated_minimal_vector.item(1), head_width=0.8, head_length=0.8, width = 0.2, color = 'C0', label = "Estimated (deg)", alpha = 1)
    plt.arrow(X0[0], X0[1], 7 * estimated_minimal_vector2.item(0), 5 * estimated_minimal_vector2.item(1), head_width=0.8, head_length=0.8, width = 0.2, color = 'C1', label = "Estimated (point_wise)", alpha = 1)
    plt.legend()
    plt.show()


    # dir_est = est_dir[-1]
    # plt.scatter(X_projected[:,0], X_projected[:,1], c = relative_depths2)
    # plt.show()

    # est_dir_float = np.sign(X0[1]) * est_dir[-1]
    # rot_est: np.matrix = np.matrix([[np.cos(est_dir_float), -np.sin(est_dir_float)],[np.sin(est_dir_float),np.cos(est_dir_float)]])
    # a_est = np.dot(rot_est, np.array([1,0]))

    # # Normalize the data
    # X_norm =  (X - mean.T) @ scipy.linalg.sqrtm(np.linalg.inv(variance))

    # # Normalize the point of interest
    # X0_norm = (X0 - mean.T[0]) @ scipy.linalg.sqrtm(np.linalg.inv(variance))

    # # Calculate and normalize the directional vector
    # X0_norm_dir_vec = X0_norm / np.linalg.norm(X0_norm)
    # X0_norm_dir = rad_wu(p1 = np.array([X0_norm_dir_vec]))

    # # Transform the directional vector back onto the ellipse
    # X0_denorm = X0_norm_dir_vec @ scipy.linalg.sqrtm(variance)
    # X0_denorm_dir_vec = X0_norm_dir_vec @ scipy.linalg.sqrtm(np.linalg.inv(variance))



    # #np.where(np.abs(angles) <= np.pi / 2, angles, 1)
    # plt.scatter(XN[:,0], XN[:,1], c = relative_angles2)
    # plt.scatter(X0[0], X0[1], color = 'red', s = 100)
    # plt.scatter(mean[0][0], mean[1][0], color = 'red', s = 100, marker = 'x')
    # # plt.arrow(X0[0], X0[1], 7 * a.item(0), 7 * a.item(1), head_width=0.8, head_length=0.8, width = 0.2, color = 'red', label = 'True')
    # plt.arrow(X0[0], X0[1], 7 * X0_denorm_dir_vec[0], 7 * X0_denorm_dir_vec[1], head_width=0.8, head_length=0.8, width = 0.2, color = 'red', label = 'True', alpha = .5)
    # plt.arrow(X0[0], X0[1], 7 * a_est.item(0), 7 * a_est.item(1), head_width=0.8, head_length=0.8, width = 0.2, color = 'blue', label = "Est", alpha = .5)
    # plt.legend()
    # plt.show()





    

    # TD = Analytic_Depth(X = X, A1 = AR_coef[0], S0 = np.eye(2), X0 = X0)

    # variance = TD.variance
    # mean = TD.mean
    # dir = TD.dir

    # # X = (X - mean.T) / np.linalg.norm(X - mean.T)

    # X_norm =  (X - mean.T) @ scipy.linalg.sqrtm(np.linalg.inv(variance))
    # X0_norm = (X0 - mean.T[0]) @ scipy.linalg.sqrtm(np.linalg.inv(variance))
    # X0_norm_dir_vec = X0_norm / np.linalg.norm(X0_norm)
    # X0_norm_dir = rad_wu(p1 = np.array([X0_norm_dir_vec]))

    # X0_denorm = X0_norm_dir_vec @ scipy.linalg.sqrtm(variance)

    # X0_denorm_dir_vec = X0_norm_dir_vec @ scipy.linalg.sqrtm(np.linalg.inv(variance))

    # plt.scatter(X[:,0], X[:,1], alpha = .5)
    # plt.scatter(X0[0], X0[1], color = 'red', s = 100)
    # plt.scatter(X0_denorm[0], X0_denorm[1], color = 'red', s = 100, marker = 'X')
    # plt.scatter(X0_denorm_dir_vec[0], X0_denorm_dir_vec[1], color = 'red', s = 100, marker = 'X')
    # plt.scatter(mean[0],mean[1], c = 'red', marker = 'x')
    # plt.show()

    # plt.scatter(X_norm[:,0], X_norm[:,1], alpha = .5)
    # plt.scatter(X0_norm[0], X0_norm[1], color = 'red', s = 100)
    # plt.scatter(X0_norm_dir_vec[0], X0_norm_dir_vec[1], color = 'red', s = 100, marker = 'X')
    # plt.scatter(0,0, c = 'red', marker = 'x')
    # plt.show()


    # # X0_scaled = np.linalg.inv(variance) @ X0.T
    # # true_TP = TD.dir
    # true_depth, true_direction, true_angle = TD.TD_analytic()

    # for n in tqdm(samples):
    #     XN = X[:n]

    #     dep_list, dir_list, angles, depth_array, direction_array = TDE(X = XN, X0 = X0)

    #     est_dir.append(dir_list)
    #     est_dep.append(dep_list)



    # fig, axs = plt.subplots(3, 1, layout='constrained')
    # axs[0].plot(samples, est_dep, lw = 1, c = 'C0', marker = 'x')
    # # axs[0].axhline(true_depth, color = 'C1', linestyle = 'dashed', linewidth = 2)
    # axs[0].set_xlabel('Sample size')
    # axs[0].set_ylabel('Depth')
    # axs[0].grid(True)

    # axs[1].plot(samples, est_dir, lw = 1, c = 'C0', marker = 'x')
    # # axs[1].axhline(true_depth, color = 'C1', linestyle = 'dashed', linewidth = 2)
    # axs[1].set_xlabel('Sample size')
    # axs[1].set_ylabel('Direction')
    # axs[1].grid(True)
    # # axs[1].set_ylim([true_TP - np.pi / 8, true_TP + np.pi / 8])

    # axs[2].plot(X[:,0], lw = 1, c = 'C0', marker = 'x')
    # axs[2].plot(X[:,1], lw = 1, c = 'C1', marker = 'x')
    # axs[2].set_xlabel('Index')
    # axs[2].set_ylabel('X')
    # axs[2].grid(True)

    # plt.show()


    # X_shift = XN - X0
    # X_proj = X_shift / np.pow( (X_shift*X_shift) @ np.array([1,1]), .5).T[:, np.newaxis]
    
    # dir_est = est_dir[-1]
    # plt.scatter(X_proj[:,0], X_proj[:,1], c = relative_depths2)
    # plt.show()

    # est_dir_float = np.sign(X0[1]) * est_dir[-1]
    # rot_est: np.matrix = np.matrix([[np.cos(est_dir_float), -np.sin(est_dir_float)],[np.sin(est_dir_float),np.cos(est_dir_float)]])
    # a_est = np.dot(rot_est, np.array([1,0]))

    # #np.where(np.abs(angles) <= np.pi / 2, angles, 1)
    # plt.scatter(XN[:,0], XN[:,1], c = relative_angles2)
    # plt.scatter(X0[0], X0[1], color = 'red', s = 100)
    # plt.scatter(mean[0][0], mean[1][0], color = 'red', s = 100, marker = 'x')
    # # plt.arrow(X0[0], X0[1], 7 * a.item(0), 7 * a.item(1), head_width=0.8, head_length=0.8, width = 0.2, color = 'red', label = 'True')
    # plt.arrow(X0[0], X0[1], 7 * X0_denorm_dir_vec[0], 7 * X0_denorm_dir_vec[1], head_width=0.8, head_length=0.8, width = 0.2, color = 'red', label = 'True', alpha = .5)
    # plt.arrow(X0[0], X0[1], 7 * a_est.item(0), 7 * a_est.item(1), head_width=0.8, head_length=0.8, width = 0.2, color = 'blue', label = "Est", alpha = .0)
    # plt.legend()
    # plt.show()
