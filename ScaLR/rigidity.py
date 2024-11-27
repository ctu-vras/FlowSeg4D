import torch
from torch_scatter import scatter

def rigid_transformation(pc1, flow, id_mask):
    """
    Applies rigid transformation to a point cloud.

    Args:
        pc1 (torch.Tensor): Input point cloud tensor.
        flow (torch.Tensor): Flow tensor.
        id_mask (torch.Tensor): ID mask tensor.

    Returns:
        torch.Tensor: Transformed point cloud tensor.
    """
    # TODO also check for valid batches

    # OGC masking the transformed point cloud to propaged gradients to the probabilities, point cloud is superposition of prob ratios
    warped_pc1 = pc1 + flow 

    id_means = scatter(pc1[0], id_mask[0], dim=0, reduce='mean')    # TODO - this will be weightes as well
    warped_means = scatter(warped_pc1[0], id_mask[0], dim=0, reduce='mean')

    shifted_pc1 = pc1[0] - id_means[id_mask[0]]     # this will be multiplied by the weights
    shifted_warped_pc1 = warped_pc1[0] - warped_means[id_mask[0]] # this will be multiplied by the weights

    # cov_ids = scatter(cov_residuals, id_mask[0], dim=0, reduce='mean')

    # Scattered Covariance (check for valid batches)
    cov_matrices = scatter(shifted_pc1[:, :, None] * shifted_warped_pc1[:, None, :], id_mask[0], dim=0, reduce='mean')

    # TODO - divide covariance matrices by weights
    # TODO - deal with reflection

    # kabsch
    U, S, V = torch.svd(cov_matrices)   
    # Rotation matrix
    R = V @ U.transpose(1, 2)
    # Translation vector
    t = warped_means.unsqueeze(-1) - torch.bmm(R, id_means.unsqueeze(-1))

    # transformed_t = torch.bmm(id_means[:, None, :], R) # shift to local object origin center and also put global origin to first center

    T = torch.cat((R, t), dim=2)

    fill = torch.zeros((R.shape[0], 1, 4), device=pc1.device)
    fill += torch.tensor([0, 0, 0, 1], device=pc1.device).float()
    T = torch.cat((T, fill), dim=1)

    return T
    
    # print("HERE CHECK, There is a mistake")
    # rigid_pc = (torch.bmm(pc1, rot_mats[:,:3,:3].transpose(1,2)) + translations[:, :].unsqueeze(-1)).squeeze(-1)
    # different

    # rigid_flow = (rigid_pc - pc1[0]).unsqueeze(0)

    # rigid_loss = (rigid_flow.detach() - flow).norm(dim=-1).mean()

    return pc_transformed


def reconstruct_rigid_flow(pc1, id_trans, id_mask):
    #  Reconstuction of rigid flow from Kabsch
    rot_mats = id_trans[:, :3, :3][id_mask[0]]
    translations = id_trans[:, :3, 3][id_mask[0]]

    # print(rot_mats.shape, translations.shape, pc1.shape)
    pc_transformed = torch.bmm(rot_mats, pc1[0].unsqueeze(-1)).squeeze(-1) + translations

    rigid_flow = (pc_transformed - pc1[0]).unsqueeze(0)

    return rigid_flow


def rigid_loss(pc1, flow, id_mask):
    # Reconstruction of rigid flow from Kabsch
    T = rigid_transformation(pc1, flow, id_mask)
    
    # Reconstruction of rigid flow from Kabsch
    rigid_flow = reconstruct_rigid_flow(pc1, T, id_mask)

    rigid_loss = (rigid_flow.detach() - flow).norm(dim=-1).mean()

    return rigid_loss 

# def Unifinished_scatter_svd():
#     from torch_scatter import scatter
#     # for noise cluster, do not do it. Reassign later by KNN
#     # print(clusters.max())
#     flow = torch.zeros((1, current_pc1.shape[1], 3), device=current_pc1.device)

#     dist, nn_ind, _ = knn_points(current_pc1 + flow, pc2, K=1)

#     correspondences = pc2[0, nn_ind[0,:,0]].unsqueeze(0)

#     weights = torch.ones(correspondences.shape[1], device=correspondences.device)   # tmp, later reassigned with visibility



#     # knn
#     # TODO also check for valid batches

#     # OGC masking the transformed point cloud to propaged gradients to the probabilities, point cloud is superposition of prob ratios
#     # warped_pc1 = pc1 + flow 
#     weighted_means = scatter(weights, pc1_ids[0], dim=0, reduce='mean')
#     total_weight = scatter(weights, pc1_ids[0], dim=0, reduce='sum')
#     # Weighted means
#     source_means = scatter(current_pc1[0] * weights[:,None], pc1_ids[0], dim=0, reduce='mean')    # TODO - this will be weightes as well
#     source_means = source_means / weighted_means[:, None]

#     correspondences_means = scatter(correspondences[0] * weights[:, None], pc1_ids[0], dim=0, reduce='mean')
#     correspondences_means = correspondences_means / weighted_means[:, None]

#     # Mean-centered points sets
#     Xc = current_pc1[0] - source_means[pc1_ids[0]]
#     Yc = correspondences[0] - correspondences_means[pc1_ids[0]]

#     # Apply weights to the points
#     Xc *= weights[:, None]
#     Yc *= weights[:, None]

#     # todo total weight with respect to means?
#     # each cov matrix should be divided by the total weight of the cluster

#     cov_matrices = scatter(Xc[:, :, None] * Yc[:, None, :], pc1_ids[0], dim=0, reduce='mean')
#     cov_matrices = cov_matrices / total_weight[:, None, None]

#     U, S, V = torch.svd(cov_matrices)   

#     # # Rotation matrix
#     R = V @ U.transpose(1, 2)


#     # print(R)
#     # # Translation vector
#     t = correspondences_means.unsqueeze(-1) - torch.bmm(R, source_means.unsqueeze(-1))  # get compunents

#     # cov_matrices
#     # # Scattered Covariance (check for valid batches)

#     # TODO - valid batches
#     # # TODO - deal with reflection

#     T = torch.cat((R, t), dim=2)

#     fill = torch.zeros((R.shape[0], 1, 4), device=pc1.device)
#     fill += torch.tensor([0, 0, 0, 1], device=pc1.device).float()
#     T = torch.cat((T, fill), dim=1)

#     # T[-10]