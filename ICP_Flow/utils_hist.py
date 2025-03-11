import warnings

import torch

from ICP_Flow.hist_cuda.hist import hist
from ICP_Flow.utils_helper import nearest_neighbor_batch

warnings.filterwarnings('ignore')

# tok=1 already works decently. topk=5 is for ablation study and works slightly better
def topk_nms(x, k=5, kernel_size=11):
    b, _, _, _ = x.shape
    x = x.unsqueeze(1)
    xp = torch.nn.functional.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2)
    mask = (x == xp).float().clamp(min=0.0)
    xp = x * mask
    votes, idxs = torch.topk(xp.view(b, -1), dim=1, k=k)
    del xp, mask
    return votes, idxs.long()

# ### sometimes memeory overflows when time duration increases, because of too many clusters
# ### in case memory overflows, decrease the batch size to fit your GPUs if necessary
def estimate_init_pose(args, src, dst):
    transformations = []
    assert len(src) == len(dst)
    n = len(src)//args.chunk_size if len(src) % args.chunk_size ==0 else len(src)//args.chunk_size +1
    for k in range(0, n):
        transformation = estimate_init_pose_batch(args, 
                                                  src[k*args.chunk_size: (k+1)*args.chunk_size], 
                                                  dst[k*args.chunk_size: (k+1)*args.chunk_size])
        transformations.append(transformation)
    transformations = torch.vstack(transformations)
    return transformations

def estimate_init_pose_batch(args, src, dst):
    pcd1 = src[:, :, 0:3]
    pcd2 = dst[:, :, 0:3]
    mask1 = src[:, : , -1] > 0.0
    mask2 = dst[:, : , -1] > 0.0

    ###########################################################################################
    eps = 1e-8
    # https://pytorch.org/docs/stable/generated/torch.arange.html#torch-arange
    bins_x = torch.arange(-args.translation_frame, args.translation_frame+args.thres_dist-eps, args.thres_dist)
    bins_y = torch.arange(-args.translation_frame, args.translation_frame+args.thres_dist-eps, args.thres_dist)
    bins_z = torch.arange(-args.thres_dist, args.thres_dist+args.thres_dist-eps, args.thres_dist)

    # bug there: when batch size is large!
    t_hist = hist(dst, src, 
                  bins_x.min(), bins_y.min(), bins_z.min(),
                  bins_x.max(), bins_y.max(), bins_z.max(),
                  len(bins_x), len(bins_y), len(bins_z))
    b, h, w, d = t_hist.shape
    ###########################################################################################

    _, t_argmaxs = topk_nms(t_hist)
    t_dynamic = torch.stack([ bins_x[t_argmaxs//d//w%h], bins_y[t_argmaxs//d%w], bins_z[t_argmaxs%d] ], dim=-1) + args.thres_dist//2
    del t_hist, bins_x, bins_y, bins_z

    n = pcd1.shape[1]
    t_both = torch.cat([t_dynamic, t_dynamic.new_zeros(b, 1, 3)], dim=1)
    k = t_both.shape[1]

    pcd1_ = pcd1[:, None, :, :] + t_both[:, :, None, :]
    pcd2_ = pcd2[:, None, :, :].expand(-1, k, -1, -1)

    _, errors = nearest_neighbor_batch(pcd1_.reshape(b*k, n, 3), pcd2_.reshape(b*k, n, 3))
    _, errors_inv = nearest_neighbor_batch(pcd2_.reshape(b*k, n, 3), pcd1_.reshape(b*k, n, 3))

    # using errors
    errors = (errors.view(b, k, n) * mask1[:, None, :]).sum(dim=-1) / mask1[:, None, :].sum(dim=-1)
    errors_inv = (errors_inv.view(b, k, n) * mask2[:, None, :]).sum(dim=-1) / mask2[:, None, :].sum(dim=-1)
    errors = torch.minimum(errors, errors_inv)
    _, idx = errors.min(dim=-1)
    t_best = t_both[torch.arange(0, b, device=idx.device), idx, :]
    del pcd1_, pcd2_, errors

    transformation = torch.eye(4)[None].repeat(b, 1, 1)
    transformation[:, 0:3, -1] = t_best
    return transformation

