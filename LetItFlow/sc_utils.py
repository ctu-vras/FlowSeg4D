# Functions in this file are from: https://github.com/ZhiChen902/SC2-PCR
# Implementation taken from: https://github.com/kavisha725/MBNSF/


import torch


def power_iteration(M, num_iterations=10):
    """
    Calculate the leading eigenvector using power iteration algorithm
    Input:
        - M:      [bs, num_corr, num_corr] the compatibility matrix
    Output:
        - solution: [bs, num_corr] leading eigenvector
    """
    leading_eig = torch.ones_like(M[:, :, 0:1])
    leading_eig_last = leading_eig
    for _ in range(num_iterations):
        leading_eig = torch.bmm(M, leading_eig)  # vector sum row and column-wise
        leading_eig = leading_eig / (
            torch.norm(leading_eig, dim=1, keepdim=True) + 1e-6
        )  # average displacement
        if torch.allclose(leading_eig, leading_eig_last):
            break
        leading_eig_last = leading_eig
    leading_eig = leading_eig.squeeze(-1)
    return leading_eig


def spatial_consistency_score(M, leading_eig):
    """
    Calculate the spatial consistency score based on spectral analysis.
    Input:
        - M:          [bs, num_corr, num_corr] the compatibility matrix
        - leading_eig [bs, num_corr]           the leading eigenvector of matrix M
    Output:
        - sc_score
    """
    sc_score = leading_eig[:, None, :] @ M @ leading_eig[:, :, None]
    sc_score = sc_score.squeeze(-1) / M.shape[1]
    return sc_score
