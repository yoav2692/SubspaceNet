import torch
import torch.nn as nn

from src.utils import *
from src.system_model import SystemModel
import matplotlib.pyplot as plt


class SubspaceMethod(nn.Module):
    """

    """

    def __init__(self, system_model: SystemModel):
        super(SubspaceMethod, self).__init__()
        self.system_model = system_model
        self.eigen_threshold_val = 0.3
        self.eigen_threshold = nn.Parameter(torch.tensor(self.eigen_threshold_val), requires_grad=False)
        self.eigen_distribution_regularization_flag = False
        self.ev_dist_diff_weight = 1/100
        self.create_eigen_distribution()
        self.PLOT_EV = False
        if self.PLOT_EV:
            self.colorCounter = 0
            plt.figure(f"EV Distribution for {self.eigen_threshold_val}")

    def subspace_separation(self,
                            covariance: torch.Tensor,
                            number_of_sources: torch.tensor = None,
                            eigen_regularization: bool = True) \
            -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.tensor):
        """

        Args:
            covariance:
            number_of_sources:
            eigen_regularization:

        Returns:
            the signal ana noise subspaces, both as torch.Tensor().
        """
        eigenvalues, eigenvectors = torch.linalg.eig(covariance)
        sorted_idx = torch.argsort(torch.real(eigenvalues), descending=True)
        sorted_eigvectors = torch.gather(eigenvectors, 2,
                                         sorted_idx.unsqueeze(-1).expand(-1, -1, covariance.shape[-1]).transpose(1, 2))
        # number of sources estimation
        real_sorted_eigenvals = torch.gather(torch.real(eigenvalues), 1, sorted_idx)
        normalized_eigen = real_sorted_eigenvals / real_sorted_eigenvals[:, 0][:, None]
        # torch.mean(normalized_eigen,dim=1)
        # normalized_eigen[ii]>torch.mean(normalized_eigen,dim=1)[ii]
        # torch.mean(normalized_eigen[0][normalized_eigen[0]>torch.mean(normalized_eigen,dim=1)[0]])
        # torch.mean(normalized_eigen[ii][normalized_eigen[ii]<torch.mean(normalized_eigen,dim=1)[ii]])

        source_estimation = torch.linalg.norm(
            nn.functional.relu(
                normalized_eigen - self.eigen_threshold * torch.ones_like(normalized_eigen)),
            dim=1, ord=0)
        if number_of_sources is None:
            warnings.warn("Number of sources is not defined, using the number of sources estimation.")
            signal_subspace = sorted_eigvectors[:, :, :source_estimation]
            noise_subspace = sorted_eigvectors[:, :, source_estimation:]
        else:
            signal_subspace = sorted_eigvectors[:, :, :number_of_sources]
            noise_subspace = sorted_eigvectors[:, :, number_of_sources:]
        if self.PLOT_EV:
            ## use your debug console 
            # self.PLOT_EV
            # plt.figure() or plt.figure().clear
            plt.stem(normalized_eigen[0].detach().numpy(),  bottom=self.eigen_threshold_val , markerfmt=f'C{self.colorCounter}o')
            self.colorCounter += 1
            self.colorCounter %= 10

        if eigen_regularization:
            l_eig = self.eigen_regularization(normalized_eigen, number_of_sources)
            if self.eigen_distribution_regularization_flag:
                ev_dist_diff = self.eigen_distribution_regularization(normalized_eigen, number_of_sources)
                l_eig += ev_dist_diff * self.ev_dist_diff_weight
        else:
            l_eig = None

        return signal_subspace.to(device), noise_subspace.to(device), source_estimation, l_eig

    def eigen_regularization(self, normalized_eigenvalues: torch.Tensor, number_of_sources: int):
        """

        Args:
            normalized_eigenvalues:
            number_of_sources:

        Returns:

        """
        l_eig = (normalized_eigenvalues[:, number_of_sources - 1] - self.eigen_threshold) * \
                (normalized_eigenvalues[:, number_of_sources] - self.eigen_threshold)
        l_eig = torch.sum(l_eig)
        return l_eig

    def eigen_distribution_regularization(self, normalized_eigenvalues: torch.Tensor, number_of_sources: int):
        ev_dist_diff = torch.linalg.norm(
            normalized_eigenvalues - self.eigen_distribution,
            dim=1, ord=2)
        ev_dist_diff = sum(ev_dist_diff)
        return ev_dist_diff * self.ev_dist_diff_weight

    # def create_eigen_distribution(self, number_of_sources: int, descisiveness: int):
    def create_eigen_distribution(self):
        K = int( 3 * self.system_model.sensors_array.last_sensor_loc // 2)
        number_of_sources = 8
        Nv = self.system_model.sensors_array.last_sensor_loc
        # offset = 0 #number_of_sources
        # descisiveness = 0.5
        # sigmoid = nn.Sigmoid()
        # output = 1 - sigmoid( offset + descisiveness * torch.arange(-K,K))
        # eigen_distribution = output[torch.arange(0,output.shape[0],int(output.shape[0]/self.system_model.sensors_array.last_sensor_loc))]
        eigen_distribution = [1 - i**2/(Nv**2)  if i < number_of_sources else (Nv - 1 - i)/Nv for i in range(self.system_model.sensors_array.last_sensor_loc) ]
        self.eigen_distribution = torch.tensor(eigen_distribution)
        return eigen_distribution

    def pre_processing(self, x: torch.Tensor, mode: str = "sample"):
        if mode == "sample":
            Rx = self.__sample_covariance(x)
        elif mode == "sps":
            Rx = self.__spatial_smoothing_covariance(x)
        else:
            raise ValueError(
                f"SubspaceMethod.pre_processing: method {mode} is not recognized for covariance calculation.")

        return Rx

    def __sample_covariance(self, x: torch.Tensor):
        """
        Calculates the sample covariance matrix.

        Args:
        -----
            X (np.ndarray): Input samples matrix.

        Returns:
        --------
            covariance_mat (np.ndarray): Covariance matrix.
        """
        if x.dim() == 2:
            x = x[None, :, :]
        batch_size, sensor_number, samples_number = x.shape
        Rx = torch.einsum("bmt, btl -> bml", x, torch.conj(x).transpose(1, 2)) / samples_number
        return Rx

    def __spatial_smoothing_covariance(self, x: torch.Tensor):
        """
        Calculates the covariance matrix using spatial smoothing technique.

        Args:
        -----
            X (np.ndarray): Input samples matrix.

        Returns:
        --------
            covariance_mat (np.ndarray): Covariance matrix.
        """
        if x.dim() == 2:
            x = x[None, :, :]
        batch_size, sensor_number, samples_number = x.shape
        # Define the sub-arrays size
        sub_array_size = sensor_number // 2 + 1
        # Define the number of sub-arrays
        number_of_sub_arrays = sensor_number - sub_array_size + 1
        # Initialize covariance matrix
        Rx_smoothed = torch.zeros(batch_size, sub_array_size, sub_array_size, dtype=torch.complex128, device=device)

        for j in range(number_of_sub_arrays):
            # Run over all sub-arrays
            x_sub = x[:, j:j + sub_array_size, :]
            # Calculate sample covariance matrix for each sub-array
            sub_covariance = torch.einsum("bmt, btl -> bml", x_sub, torch.conj(x_sub).transpose(1, 2)) / samples_number
            # Aggregate sub-arrays covariances
            Rx_smoothed += sub_covariance.to(device) / number_of_sub_arrays
        # Divide overall matrix by the number of sources
        return Rx_smoothed
