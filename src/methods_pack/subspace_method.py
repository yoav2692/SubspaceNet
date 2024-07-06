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
        self.eigen_threshold_val = 0.5
        self.eigen_threshold = nn.Parameter(torch.tensor(self.eigen_threshold_val), requires_grad=False)
        self.PLOT_EV = False
        if self.PLOT_EV:
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
        else:
            pass

        if self.PLOT_EV:
            plt.stem(normalized_eigen[0].detach().numpy(),  bottom=self.eigen_threshold_val)

        if eigen_regularization:
            l_eig = self.eigen_regularization(normalized_eigen, number_of_sources)
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
        # eigen_regularization = nn.functional.elu(eigen_regularization, alpha=1.0)
        return l_eig

    def pre_processing(self, x: torch.Tensor, mode: str = "sample", expension_tensor: torch.Tensor = None):
        if mode == "sample":
            Rx = self.__sample_covariance(x)
        elif mode == "sps":
            Rx = self.__spatial_smoothing_covariance(x)
        else:
            raise ValueError(
                f"SubspaceMethod.pre_processing: method {mode} is not recognized for covariance calculation.")
        if expension_tensor is not None:
            Rx = expend_correlation_matrix(Rx,expension_tensor)
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
