import torch
import torch.nn.functional as F
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from pytorch3d.ops.knn import knn_points
from sklearn.neighbors import NearestNeighbors
from einops import rearrange


def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_softmax(logits, temperature = 1.):
    dtype, size = logits.dtype, logits.shape[-1]

    assert temperature > 0

    scaled_logits = logits / temperature

    # gumbel sampling and derive one hot

    noised_logits = scaled_logits + gumbel_noise(scaled_logits)

    indices = noised_logits.argmax(dim = -1)

    hard_one_hot = F.one_hot(indices, size).type(dtype)

    # get soft for gradients

    soft = scaled_logits.softmax(dim = -1)

    # straight through

    hard_one_hot = hard_one_hot + soft - soft.detach()

    # return indices and one hot

    return hard_one_hot, indices



def sklearn_knn(pts1, pts2, K=1):
    '''
    KNN using sklearn to avoid pytorch3d dependency
    pts1: torch.Tensor (bs, N, 3) - source points
    pts2: torch.Tensor (bs, M, 3) - target points
    out: torch.Tensor (bs, N, K) - indices of the nearest neighbors
    '''

    NNclass = [NearestNeighbors(n_neighbors=K, algorithm='ball_tree').fit(pts2[i].detach().cpu().numpy()) for i in range(pts2.shape[0])]
    
    knn_list = [NNclass[i].kneighbors(pts1[i].detach().cpu().numpy()) for i in range(pts1.shape[0])]
    
    dist = torch.stack([torch.tensor(knn[0]) for knn in knn_list])
    indices = torch.stack([torch.tensor(knn[1]) for knn in knn_list])

    return dist, indices    


def scatter(src=None, index=None, dim=0, out=None, L=None, dim_size=None, reduce='sum', fill_value=0):
    ''' 
    Scatter with reduce operations
    Inputs:
        src: torch.Tensor (BS, N, 3) - source tensor, dimension 3 is the feature dimension and can be arbitrary
        index: torch.Tensor (BS, N) - index tensor, max (L)
        dim: int - dimension to scatter
    out: torch.Tensor (BS, L, 3) - output tensor

    '''
    if L is None:
        out_tensor = torch.zeros((src.shape[0], index.max() + 1 , src.shape[-1]))
    else:
        out_tensor = torch.zeros((src.shape[0], L, src.shape[-1]))

    
    out_tensor.scatter_reduce_(dim, index.unsqueeze(-1).repeat(1,1,src.shape[-1]), src, reduce=reduce)

    return out_tensor


def instance_cross_covariance(pts1, warped_pts1, class_ids, L=None, weighted=None):
    
    ''' 
    Calculates the cross covariance matrix between two point clouds based on the class ids
    and splits the covariance matrices according to classes
    Inputs:
        pts1: torch.Tensor (bs, N, 3)
        warped_pts1: torch.Tensor (bs, N, 3)
        class_ids: torch.Tensor (bs, N) (with Integers up to L)
    Outputs:
        H: torch.Tensor (bs, L, 3, 3) Covariance matrices
    '''
    
    id_means = scatter(pts1, class_ids, dim=1, reduce='mean')  
    warped_means = scatter(warped_pts1, class_ids, dim=1, reduce='mean')

    shifted_pts1 = pts1 - id_means.gather(1, class_ids.unsqueeze(-1).repeat(1,1,3))     # this can be multiplied by the weights
    shifted_warped_pts1 = warped_pts1 - warped_means.gather(1, class_ids.unsqueeze(-1).repeat(1,1,3))  # this can be multiplied by the weights

    # Scattered Covariance (checked)
    cov_xy = scatter(src=shifted_pts1[:, :, 0:1] * shifted_warped_pts1[:, :, 1:2], index=class_ids, dim=1, L=L, reduce='mean')
    cov_xz = scatter(src=shifted_pts1[:, :, 0:1] * shifted_warped_pts1[:, :, 2:3], index=class_ids, dim=1, L=L, reduce='mean')
    cov_yz = scatter(src=shifted_pts1[:, :, 1:2] * shifted_warped_pts1[:, :, 2:3], index=class_ids, dim=1, L=L, reduce='mean')
    var_xyz = scatter(src=shifted_pts1 * shifted_warped_pts1, index=class_ids, dim=1, L=L, reduce='mean')

    H = torch.stack((torch.cat((var_xyz[:, :, 0:1], cov_xy, cov_xz), dim=-1),
        torch.cat((cov_xy, var_xyz[:, :, 1:2], cov_yz), dim=-1),
        torch.cat((cov_xz, cov_yz, var_xyz[:, :, 2:3]), dim=-1)), dim=-2)

    return H


def fit_svd_motion(pts1, warped_pts1, id_mask):
    ''' 
    Fits the motion using SVD decomposition in a batch manner
    Inputs:
        pts1: torch.Tensor (bs, N, 3)
        warped_pts1: torch.Tensor (bs, N, 3) - correspondences or warped flow
        id_mask: torch.Tensor (bs, N, L)
    Outputs:
        T: torch.Tensor (bs, L, 3, 4) Transformation matrices
    '''
    
    class_ids = torch.argmax(id_mask, dim=-1)

    bs = pts1.shape[0]
    l = id_mask.shape[-1]
    
    id_means = scatter(src=pts1, index=class_ids, L=l, dim=1, reduce='mean')
    
    warped_means = scatter(src=warped_pts1, index=class_ids, L=l, dim=1, reduce='mean')

    H = instance_cross_covariance(pts1, warped_pts1, class_ids, L=l)

    U, S, V = torch.linalg.svd(H.view(-1, 3, 3))

    R = torch.bmm(V, U.transpose(1, 2))
    
    # Correct reflection matrix to rotation matrix
    det = torch.det(R)
    diag = torch.ones_like(H.view(-1,3,3)[..., 0], requires_grad=False)
    diag[:, 2] = det
    R = V.bmm(torch.diag_embed(diag).bmm(U.transpose(1, 2)))
    
    t = warped_means.view(bs * l, 3).unsqueeze(-1) - torch.bmm(R, rearrange(id_means, 'b l c -> (b l) c').unsqueeze(-1))
    
    # Merge translation and rotation
    T = torch.cat((R, t), dim=2)
    T = T.view(bs, l, 3, 4)

    fill = torch.zeros((bs, l, 1, 4), device=pts1.device)
    fill[:, :, -1, 3] = 1

    T = torch.cat((T, fill), dim=2)

    return T

def transform_pts(pts1, T, id_mask):
    ''' 
    Transforms the point cloud using the transformation matrices and class ids
    Inputs:
        pts1: torch.Tensor (bs, N, 3)
        T: torch.Tensor (bs, L, 4, 4) Transformation matrices
        id_mask: torch.Tensor (bs, N, L)
    Outputs:
        transformed_pts: torch.Tensor (bs, N, 3) 

    '''
    bs = pts1.shape[0]    
    N = pts1.shape[1]
    l = id_mask.shape[-1]
    class_ids = torch.argmax(id_mask, dim=-1)

    # Create batch indices for broadcasting
    batch_indices = torch.arange(bs, device=pts1.device).view(bs, 1).repeat(1, N).view(-1)

    # Rigid transformation applied to points
    broadcasted_T = T[batch_indices, class_ids.view(-1)].view(bs, N, 4, 4) 
    transformed_pts = broadcasted_T.view(bs * N, 4, 4)[:, :3, :3] @ pts1.view(bs * N, 3, 1) + broadcasted_T.view(bs * N, 4, 4)[:, :3, 3:4]
    transformed_pts = transformed_pts.view(bs, N, 3)

    return transformed_pts

def object_aware_icp(pts1, pts2, id_mask, truncated_dist=0.2, max_iteration=50, tolerance=0.001, verbose=False, visualize_path=None):
    
    '''
    Iterative Closest Point with object-mask and paralelized for batch
    Inputs: pts1: torch.Tensor (bs, N, 3) - source points
            pts2: torch.Tensor (bs, M, 3) - target points
            id_mask: torch.Tensor (bs, N, L) - object mask
            truncated_dist: float - distance threshold for ICP
            max_iteration: int - maximum number of iterations
            tolerance: float - tolerance for convergence
            verbose: bool - print progress
            visualize_path: str - path to save visualization

    Outputs: final_T: torch.Tensor (bs, L, 4, 4) - final transformation matrices    
             src: torch.Tensor (bs, N, 3) - transformed source points
             id_mask: torch.Tensor (bs, N, L) - updated object mask

    '''
    
    hot_encoding, indices = gumbel_softmax(id_mask, temperature=1)
    
    # Make differentiable points cloud based on the mask
    src = (pts1.unsqueeze(-1) * hot_encoding.unsqueeze(-2)).sum(dim=-1)

    for i in range(max_iteration):
        
        dist, indices, _ = knn_points(src, pts2, K=1)
        # id_mask = id_mask.gather(1, indices.expand(-1, -1, id_mask.size(-1)))
        
        # Commented line is to extract next frame mask from correspondences
        # nn_masks = id_mask.gather(1, indices.expand(-1, -1, id_mask.size(-1)))

        correspondeces = pts2.gather(1, indices.expand(-1, -1, src.size(-1)))

        T = fit_svd_motion(src, correspondeces, id_mask)

        # update source points
        src = transform_pts(src, T, id_mask)

        mean_error = torch.mean(torch.norm(correspondeces - src, dim=-1))

        if mean_error < tolerance:
            break
        
        if verbose:
            tqdm.write(f"Iteration {i+1}, Mean Error: {mean_error.item():.6f}, Classes: {torch.unique(id_mask.argmax(dim=-1), return_counts=True)[1]}")

        if visualize_path is not None:
            plt.close()
            for j in range(src.shape[1]):
                plt.plot([src[-1, j, 0].detach(), correspondeces[-1, j, 0].detach()],
                 [src[-1, j, 1].detach(), correspondeces[-1, j, 1].detach()], 'k-', lw=0.5)

            plt.scatter(pts2[-1, :, 0], pts2[-1, :, 1], c='r', marker='x')
            plt.scatter(src[-1, :, 0].detach(), src[-1, :, 1].detach(), c=id_mask[-1].argmax(dim=-1), cmap='viridis')
            plt.xlim(-1, 3)
            plt.ylim(-1, 3)
            plt.savefig(f'{visualize_path}/{i:04d}.png')

    final_T = fit_svd_motion(pts1, src, id_mask)

    return final_T, src, id_mask

if __name__ == '__main__':
    
    # Create toy data
    torch.manual_seed(0)
    bs = 3  # number of point clouds in sequence
    N = 15  # number of points
    L = 3   # number of classes
    pts1 = torch.rand(bs, N, 3)

    # The points are made from sequence
    pts2 = pts1[1:].clone()    # remove last batch
    pts1 = pts1[:-1]    # remove last batch

    # Toy example to test the object mask
    pts2 = pts1.clone()
    pts2[:, :, 0] += 1.5
    pts2[:, :, -1] = 0 
    pts1[:, :, -1] = 0

    # Initialize instance probabilities, will come from the model
    id_mask = torch.rand(pts1.shape[0], pts1.shape[1], L).requires_grad_(True)

    # Solution to multi-class ICP with backpropagation
    final_T, transformed_pts, new_ids = object_aware_icp(pts1, pts2, id_mask, max_iteration=20, verbose=True, visualize_path='samples/')
    
    # Final transformed points and distance loss
    dist, indices_nn, _ = knn_points(transformed_pts, pts2, K=1)

    # multiply id_mask with dist, can be done without gumbel softmax by indexing argmax of the mask
    hot_encoding, indices = gumbel_softmax(id_mask, temperature=1)

    # Dynamic loss decrease probability of the instance, if distance is still high, resulting in a change of category
    # Detach is simple version for better numerical stability. Gradients flow through transformation though.
    dynamic_loss = (hot_encoding * dist.detach()).mean()