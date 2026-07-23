import numpy as np
from typing import Callable, Any
import scipy
from tqdm import tqdm

def transition_markov(X, p, e, /, **kwargs): return X - X / (1 + (np.abs(X) ** p)) + e
def transition_diff(X, p, e, /, **kwargs): return kwargs.get('rho', .9) * X + e / (1+ np.abs(X) ** p)

class MixingMarkovModel(object):

    def __init__(self, 
                 mixing_rate: np.float64 = 1.0,
                 transition_function: Callable = transition_markov,
                 ) -> None:
        self.mixing_rate = mixing_rate
        self.transition_functional = transition_function
        self.X = []

    @staticmethod
    def __reshape(X) -> np.ndarray: return np.vstack(X)
         
    def simulate(self, 
                 innov: str = 'gaussian',
                 innov_config: dict = {},
                 n_samples: int = 100,
                 **kwargs
                 ):
        if not isinstance(innov, str):
            ...

        else:
            if innov == 'gaussian':
                    innov_array: np.array = np.random.normal(size = (n_samples, 2), **innov_config)
                    self.X.append(np.array([0,0]))
                    for inn in range(n_samples - 1):
                        self.X.append(
                            self.transition_functional(self.X[-1], self.mixing_rate, innov_array[inn], **kwargs)
                            )
                        
                    return self.__reshape(self.X)
            else: raise NotImplementedError(innov + ' not implemented yet')

class MixingLinearModel(object):

    def __init__(self, 
                 mixing_rate: np.float64 = 4.0,
                 transition_function: Callable = transition_markov,
                 **kwargs,
                 ) -> None:
        self.mixing_rate = mixing_rate
        self.transition_functional = transition_function
        self.X = []
        self.weights = self.Weights
        self.zeta0 = self.zeta

    @staticmethod
    def __reshape(X) -> np.ndarray: return np.vstack(X)

    def __error(self, est: float = 0, k: int = 1) -> float:
        r"""
        Gives an adjustment to account for the difference between the theoretical distribution \zeta(2p)\mathcal{L}(\xi_0) and the empirical distribution \sum_{k=1}^nk^{-p}}\mathcal{L}(\xi_0)
        """
        if not est: 
            true_0 = self.zeta0
            true_k = self.zeta_k(k = k)
            return true_0 / true_k
        else: 
            true_0 = self.zeta0
            true_k = self.zeta_k(k = k)
            return true_0 / true_k

    @property
    def Weights(self) -> Any: 
        """
        Gives an (n,2) array with the weights. When first called, an error fires and @Self.Weights returns [].
        """
        try: return np.array([self.weights, self.weights]).T
        except: return []
    
    @property
    def zeta(self) -> float: return scipy.special.zeta(self.mixing_rate, 1)
    def zeta_k(self, k: int = 1) -> float: return self.zeta0 - scipy.special.zeta(self.mixing_rate, k)
    
    def MarkovFunction(self, innov_array: Any, control_size: bool = True, control_error: bool = True) -> Any: 
        n_samples: int = len(innov_array)
        if control_error: 
            ce: float = self.__error(est = False, k = n_samples)
        else:
            ce: float = 1.0
        if control_size:
            return ce * np.sum(self.Weights[:n_samples] * innov_array, axis = 0)
        else: return []
         
    def simulate(self, 
                 innov: str = 'gaussian',
                 innov_config: dict = {'mean': [0,0], 'cov':[[1,0],[0,1]]},
                 n_samples: int = 1000,
                 error: np.float64 = .1 ** 8,
                 verbose: int = 1,
                 **kwargs
                 ):
        self.innov_config = innov_config

        # The approximate samlpes needed to have an error in $n^{-p}$ smaller than error.
        n_samples_error = int(n_samples + (1 / error) ** (1 / self.mixing_rate))
        n_samples_diff = n_samples_error - n_samples
        self.weights = 1 / np.array(range(1, n_samples_error + 1, 1)) ** (self.mixing_rate)
        
        if verbose == 1: print(f'n_samples required for error {error} :: {n_samples_error}')
        if not isinstance(innov, str):
            ...             

        else:
            if innov == 'gaussian':
                    """
                    Gaussian innovations are p-integrable for any p>0.
                    """
                    
                    # Generate a multivariate normal sample
                    innov_array: np.array = scipy.stats.multivariate_normal(**innov_config).rvs(size = n_samples_error)
                    self.innov_array = innov_array
                    
                    for inn in tqdm(range(0, n_samples + 1), disable = (verbose == 0)):
                        self.X.append(
                            self.MarkovFunction(innov_array = innov_array[-inn - n_samples_diff:], control_error=True)
                        )

                    return self.__reshape(self.X)
            else: raise NotImplementedError(innov + ' not implemented yet')

    def distribution(self) -> Any:
        self.mean = self.zeta0 * np.array(self.innov_config.get('mean', [0,0]))
        self.cov = scipy.special.zeta(2 * self.mixing_rate) * np.matrix(self.innov_config.get('cov', [[1,0], [0,1]]))
        return self.mean, self.cov
    
class CovHC_(object):

    def __init__(self, X: Any = [], HCX: str = 'HC1') -> None:
        self.X = X
        self.HCX = HCX

    def __call__(self, *args: Any, **kwds: Any) -> Any: return ...



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from functionalcurves.depth import *

    Markov_chain = MixingLinearModel(mixing_rate = 1.1, transition_function=transition_diff)

    true_directional_vector = np.array([0,1])
    mean = np.array([0,0])

    X0 = np.array((0, 1.5))           # Depth point
    est_dir, est_dep, test = [], [], [] # Estimated lists
    est_dir2, est_dep2, test = [], [], [] # Estimated lists
    N_samples, jump = 1_000, 50         # Simulation parameters

    X = Markov_chain.simulate(n_samples = N_samples, error = .1 ** 6)
    samples = range(jump, N_samples, jump)

    rel_depth1, rel_depth2 = [], []

    for n in tqdm(samples):
        XN = X[:n]
        depth_estimator = Estimator(X = XN, X0 = X0, method = 'deg')
        depth_estimator2 = Estimator(X = XN, X0 = X0, method = 'point_wise')

        estimated_depth, estimated_direction, relative_angles, relative_depths, relative_directions = depth_estimator.main()
        estimated_depth2, estimated_direction2, relative_angles2, relative_depths2, relative_directions2 = depth_estimator2.main()

        rel_depth1.append(relative_depths)
        rel_depth2.append(relative_depths2)

        est_dir.append(estimated_direction)
        est_dep.append(estimated_depth)

        est_dir2.append(estimated_direction2)
        est_dep2.append(estimated_depth2)

    # Graph statistics
    fig, axs = plt.subplots(3, 1, layout='constrained')
    axs[0].plot(samples, est_dep, lw = 1, c = 'C0', marker = 'x', label = 'Method: deg')
    axs[0].plot(samples, est_dep2, lw = 1, c = 'C1', marker = 'x', label = 'Method: point_wise')
    axs[0].set_xlabel('Sample size')
    axs[0].set_ylabel('Depth')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(samples, est_dir, lw = 1, c = 'C0', marker = 'x', label = 'Method: deg')
    axs[1].plot(samples, est_dir2, lw = 1, c = 'C1', marker = 'x', label = 'Method: point_wise')
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
    plt.scatter(mean[0], mean[1], color = 'red', s = 100, marker = 'x')
    plt.arrow(X0[0], X0[1], 2 * true_directional_vector[0], 2 * true_directional_vector[1], head_width=0.2, head_length=0.2, width = 0.05, color = 'black', label = 'True', alpha = .5)
    plt.arrow(X0[0], X0[1], 2 * estimated_minimal_vector.item(0), 1.8 * estimated_minimal_vector.item(1), head_width=0.2, head_length=0.2, width = 0.05, color = 'C0', label = "Estimated (deg)", alpha = 1)
    plt.arrow(X0[0], X0[1], 2 * estimated_minimal_vector2.item(0), 1.6 * estimated_minimal_vector2.item(1), head_width=0.2, head_length=0.2, width = 0.05, color = 'C1', label = "Estimated (point_wise)", alpha = 1)
    plt.legend()
    plt.show()

    import numpy as np
    import matplotlib.pyplot as plt
    import time
    from IPython.display import display, clear_output

    true_directional_angle = np.float64(np.pi / 2)
    diff1_main = np.abs(est_dir-true_directional_angle)
    diff2_main = np.abs(est_dir2-true_directional_angle)

    # ub = max(np.abs(max(diff1_main)), np.abs(max(diff2_main)))
    # lb = min(min(diff1_main), min(diff2_main)) / 2
    # for idx, n in enumerate(samples):
    #     XN = X[:n]

    #     X_shifted = XN - X0
    #     X_projected = X_shifted / np.linalg.norm(X_shifted, axis = 1).T[:, np.newaxis]

    #     estimated_minimal_vector = depth_estimator.Angle_to_vetor(est_dir[idx], normalise = True)
    #     estimated_minimal_vector2 = depth_estimator2.Angle_to_vetor(est_dir2[idx], normalise = True)

    #     diff1 = est_dir[:idx]-true_directional_angle
    #     diff2 = est_dir2[:idx]-true_directional_angle

    #     # Clear the previous plot
    #     clear_output(wait=True)
        
    #     # Create a new figure and plot
    #     fig, ax = plt.subplots(3,1, figsize = (12,12))
    #     # ax[0].set_xlim(-1.2 * max(np.abs(X[:,0])), 1.2 * max(np.abs(X[:,0])))
    #     # ax[0].set_ylim(-1.2 * max(np.abs(X[:,1])), 1.2 * max(np.abs(X[:,1])))

    #     ax[0].scatter(XN[:,0], XN[:,1], c = rel_depth2[idx])
    #     ax[0].scatter(X0[0], X0[1], color = 'red', s = 100)
    #     ax[0].scatter(mean[0], mean[1], color = 'red', s = 100, marker = 'x')
    #     # plt.arrow(X0[0], X0[1], 7 * a.item(0), 7 * a.item(1), head_width=0.8, head_length=0.8, width = 0.2, color = 'red', label = 'True')
    #     ax[0].arrow(X0[0], X0[1], 2 * true_directional_vector[0], 2 * true_directional_vector[1], head_width=0.2, head_length=0.2, width = 0.05, color = 'black', label = 'True', alpha = .5)
    #     ax[0].arrow(X0[0], X0[1], 2 * estimated_minimal_vector.item(0), 1.8 * estimated_minimal_vector.item(1), head_width=0.2, head_length=0.2, width = 0.05, color = 'C0', label = "Estimated (deg)", alpha = 1)
    #     ax[0].arrow(X0[0], X0[1], 2 * estimated_minimal_vector2.item(0), 1.6 * estimated_minimal_vector2.item(1), head_width=0.2, head_length=0.2, width = 0.05, color = 'C1', label = "Estimated (point_wise)", alpha = 1)
    #     ax[0].legend()
    #     ax[0].set_title(f'Sample size : {n}')

    #     ax[1].set_xlim(0, N_samples)
    #     # ax[1].set_ylim(lb, ub)
    #     ax[1].plot(samples[:idx], (np.array(samples[:idx]) ** (1/3)) *  np.abs(diff1), lw = 1, c = 'C0', marker = 'x', label = 'Method: deg')
    #     ax[1].plot(samples[:idx], (np.array(samples[:idx]) ** (1/3)) *  np.abs(diff2), lw = 1, c = 'C1', marker = 'x', label = 'Method: point_wise')
    #     ax[1].set_xlabel('Sample size')
    #     ax[1].set_ylabel('Direction')
    #     ax[1].grid(True)
    #     ax[1].legend()
    #     ax[1].set_yscale('linear')

    #     ax[2].set_xlim(0, N_samples)
    #     ax[2].plot(XN[:,0], lw = 1, c = 'C0', marker = 'x')
    #     ax[2].plot(XN[:,1], lw = 1, c = 'C1', marker = 'x')
    #     ax[2].set_xlabel('Index')
    #     ax[2].set_ylabel('X')
    #     ax[2].grid(True)
    #     plt.show()

    #     # Pause for 0.5 seconds
    #     time.sleep(0.01)


