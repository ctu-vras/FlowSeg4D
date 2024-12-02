import torch

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

from pytorch3d.ops.knn import knn_points
import cv2
import os

pts1 = np.load('samples/pts1.npy')  # N x 5 (intensity, x, y, z, radius)
ground_mask1 = pts1[:, 3] > - 1.3
pts1 = pts1[ground_mask1]
color = plt.cm.jet(pts1[:, 3])

# plt.scatter(pts1[:, 1], pts1[:, 2], c=color, s=0.1, cmap='jet')
# plt.savefig('test_dbscan.png', dpi=300)
# plt.close()


# pts1 = pts1[ground_mask1]

# pts2 = np.load('samples/pts2.npy')
# ground_mask2 = pts2[:, 0] > +0.
# pts2 = pts2[ground_mask2]

# pred1 = np.load('samples/pred1.npy')[ground_mask1]
# pred2 = np.load('samples/pred2.npy')[ground_mask2]
# norm_pca_features = np.load('samples/norm_pca_features.npy')[ground_mask1]

db_clusters = DBSCAN(eps=0.6, min_samples=2).fit_predict(pts1[:,1:4])

plt.scatter(pts1[:,1], pts1[:,2], c=db_clusters, s=0.03, cmap='jet')

for i in range(0, db_clusters.max()):
    plt.text(pts1[db_clusters==i,1].min(), pts1[db_clusters==i,2].min(), str(i), fontsize=5)

object_mask = db_clusters == 188
plt.scatter(pts1[object_mask,1], pts1[object_mask,2], c='r', s=0.1)
# plt.show()
plt.savefig('test_dbscan.png', dpi=300)

# fit min box
# breakpoint()

# create toy data
N = 10
M = 12
L = 4
pts1 = torch.rand(N, 3)
pts2 = torch.rand(M, 3)
pts2[:, 0] += 1.0

id_mask = torch.rand(N, L, requires_grad=False)
# id_mask[:,1] += 10
id_mask.requires_grad_(True)

# there is a intermediate operation to get box_reg
box_reg = torch.rand(L, 7, requires_grad=True)  # x, y, z, w, h, l, theta


def construct_transformation_matrix(box_reg): 
    ''' Perserves gradient(?) and batch dimension '''
    x, y, z, theta = box_reg[:, 0], box_reg[:, 1], box_reg[:, 2], box_reg[:, 6]
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    
    transformation_matrix = torch.stack([
        torch.stack([cos_theta, -sin_theta, zeros, x], dim=-1),
        torch.stack([sin_theta, cos_theta, zeros, y], dim=-1),
        torch.stack([zeros, zeros, ones, z], dim=-1),
        torch.stack([zeros, zeros, zeros, ones], dim=-1)
    ], dim=-2)
    
    return transformation_matrix

transformation_matrix = construct_transformation_matrix(box_reg)



optimizer = torch.optim.Adam([id_mask, box_reg], lr=0.3)

for i in range(20):
    
    # Multiplying the point cloud with probabilities
    # This will allow gradient to be backpropagated through the point cloud into class probabilities
    softmaxed_id_mask = torch.nn.functional.log_softmax(id_mask, dim=1)
    out = softmaxed_id_mask.unsqueeze(-1) * pts1.unsqueeze(1)
    recon_pts1 = out.mean(1)

    # Problem je v softmaxu! log_softmax funguje

    dist, nn_ind, _ = knn_points(recon_pts1.unsqueeze(0), pts2.unsqueeze(0), K=1)

    # dist = (pts2[nn_ind[...,0]] - recon_pts1[None]).norm(dim=-1) 
    # loss = (recon_pts1 - pts2[nn_ind[0]]).norm(dim=1)
    # instance_pred = id_mask.argmax(dim=1)
    # loss = dist[0] * id_mask.softmax(dim=-1)
    # loss = loss.mean()
    

    loss = dist.mean()
    loss.backward()
    print('Loss:', loss.item(), id_mask.argmax(dim=1).detach().numpy())
    

    optimizer.step()
    optimizer.zero_grad()

    # box_reg -= 0.1 * box_reg.grad
    # box_reg.grad.zero_()

    plt.close()
    plt.axis('equal')
    plt.grid(True)
    plt.xlim(-2, 1)
    plt.ylim(-2, 1)
    # plt.plot(pts1[:, 0], pts1[:, 1], 'bo')

    plt.scatter(recon_pts1[:, 0].detach(), recon_pts1[:, 1].detach(), s=10, c=id_mask.argmax(dim=1).detach().numpy(), cmap='jet')
    plt.colorbar()
    
    # plt.plot(recon_pts1[:, 0].detach(), recon_pts1[:, 1].detach(), 'go')
    # plt.plot(pts2[:, 0], pts2[:, 1], 'ro')
    plt.savefig(f'samples/demo_inst/{i:04d}.png')

image_folder = 'samples/demo_inst'
video_name = 'assets/instance_dynamic_loss.avi'

images = [img for img in sorted(os.listdir(image_folder)) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'DIVX'), 3, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()