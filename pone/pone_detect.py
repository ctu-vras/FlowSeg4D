import numpy as np
import  glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.path import Path
import open3d as o3d
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import argparse

import pandas as pd
import time
from pone_utils import MinimumBoundingBox, estimateMinimumAreaBox



def write_obj(vertices, output_obj_path='output.obj'):
    with open(output_obj_path, 'w') as obj_file:
        for vertex in vertices:
            if vertex.shape[0] == 3:
                obj_file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]}\n')
            elif vertex.shape[0] == 6:
                obj_file.write(f'v {vertex[0]} {vertex[1]} {vertex[2]} {vertex[3]} {vertex[4]} {vertex[5]}\n')
            else:
                raise ValueError('Vertices must be 3D or 6D (with color)')



def transfer_box_to_segmentation(scan, box_corners):
    
    
    def points_in_box(scan, box_corners):
        path = Path(box_corners)
        inside = path.contains_points(scan[:,:2])
        
        return inside

    segmentation_mask = np.zeros(scan.shape[0], dtype=bool)

    for corners in box_corners:
        inside = points_in_box(scan, corners)
        segmentation_mask = segmentation_mask | inside

    return segmentation_mask

def load_sequence(start_frame, end_frame, sequence):
    files = sorted(glob.glob(sequence + '/*.npz'))

    trans_list = []
    scan_list = []
    global_scan_list = []
    box_corners_list = []
    timestamp_list = []
    segmentation_mask_list = []

    for t_idx, file in enumerate(files[start_frame:end_frame]):
        f = np.load(file, allow_pickle=True)

        scan = f['scan']
        scan = scan[scan[:, 2] > GROUND_REMOVE]
        scan = scan[scan[:, 2] < HEIGHT_REMOVE]

        seg_mask = np.zeros(scan.shape[0], dtype=bool)
        seg_mask = transfer_box_to_segmentation(scan, f['box_corners']) # shift pose?

        transformation = f['transformation']
        local_box = f['box_corners']        

        # breakpoint()
        global_box = [transformation[:2,:2] @ box.T +  transformation[:2,3:4] for box in local_box]
        segmentation_mask_list.append(seg_mask)
        trans_list.append(f['transformation'])
        scan_list.append(scan)
        box_corners_list.append(global_box)
        timestamp_list.append(f['timestamp'])

        

        scan[:, 3] = 1
        global_scan = f['transformation'] @ scan.T 
        global_scan = global_scan.T
        global_scan[:, 3] = t_idx + start_frame
        global_scan_list.append(global_scan)

    # rewrite for time
    

    return np.concatenate(global_scan_list, axis=0), \
           np.concatenate(segmentation_mask_list, axis=0), \
           box_corners_list

    
def ICP4dSolver(time_pts_list, instance_pts, available_times, threshold=2.5, per_frame_move_thresh=0.05):
    
    trans_init = np.eye(4)
    boxes = []
    
    trans_list = []
    
    mask1 = instance_pts[:, -1] == available_times[0]
    
    for t_idx, t in enumerate(sorted(available_times)):
        
        
        target_pts = instance_pts[mask1, :3]
        box_t, corner_pts_t = estimateMinimumAreaBox(target_pts)
        target_pts = target_pts - box_t[:3]
        
        mask2 = instance_pts[:, -1] == t
        
        source_pts = instance_pts[mask2, :3]
        box_s, corner_pts_s = estimateMinimumAreaBox(source_pts)
        source_pts = source_pts - box_s[:3]    


        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_pts)

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(target_pts)
        

        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        

        trans = reg_p2p.transformation
        trans_init = trans 

        traj_trans = trans.copy()    # inverse

        trans_list.append(traj_trans)
        boxes.append(box_s)

    
    # Trajectory
    trajectory = []
    for i in range(len(time_pts_list)):
        trajectory_p = np.linalg.inv(trans_list[i])[:3, -1] + boxes[i][:3]
        trajectory.append(trajectory_p)

    trajectory = np.stack(trajectory)

    position_difference = np.linalg.norm(trajectory[0] - trajectory[-1], axis=0)

    dynamic = position_difference > per_frame_move_thresh * len(trajectory)

    return trajectory, dynamic, trans_list, boxes, 



def main(global_scan, segmentation, gt_box_list, EPS, MAX_ADJACENT_TIMES, Z_SCALE, MIN_NBR_PTS, MIN_INCLUDED_TIMES, MAX_SIZE, visualize_path=None, metric_path=None, store_visuals=None):
    
    pts = global_scan.copy()
    pts[:, 3] = pts[:, 3] * EPS / (MAX_ADJACENT_TIMES) 
    pts[:, 2] *= Z_SCALE

    # voxel_size = (0.2, 0.2, 0.2, EPS / (MAX_ADJACENT_TIMES) / 2 )  # Adjust the voxel size as EPS
    voxel_size = (0.2, 0.2, 0.2)  # Adjust the voxel size as EPS

    coords = (pts[:, :3] / voxel_size).astype(int)  # here I lose granularity, it should have been reverse indexed through grid
    downsampled_points = np.unique(coords, axis=0)  
    downsampled_points = downsampled_points * voxel_size
    


    db_time = time.time()
    cluster_ids = DBSCAN(eps=EPS, min_samples=1, algorithm='ball_tree').fit_predict(downsampled_points)
    
    # Perform KNN search to map downsampled points back to original points
    knn = NearestNeighbors(n_neighbors=1).fit(downsampled_points)
    distances, indices = knn.kneighbors(pts[:,:3])

    # Assign cluster IDs to original points based on nearest downsampled points
    cluster_ids = cluster_ids[indices.flatten()]
    end_db_time = time.time()
    
    cluster_ids += 1


    dynamic_mask = np.zeros(cluster_ids.shape[0], dtype=bool)
    logic_mask = np.zeros(cluster_ids.shape[0], dtype=bool)
    trajectories = {}
    boxes_dict = {}
    dynamic_id = {}

    print(f'Clustering time: {end_db_time - db_time:.2f} seconds')
    for i in tqdm(range(cluster_ids.max()), desc='Per-instance 4D ICP'):
        if i == 0: continue # noise

        i_mask = cluster_ids == i
        instance_pts = pts[i_mask]

        if len(instance_pts) < MIN_NBR_PTS:
            continue
        
        if len(np.unique(instance_pts[:,-1])) < MIN_INCLUDED_TIMES:
            continue

        available_times = sorted(np.unique(instance_pts[:,-1]))
        
        mask1 = instance_pts[:, -1] == available_times[0]   # HERE!   

        time_pts_list = [instance_pts[:, :4][instance_pts[:, -1] == t] for t in available_times]
        
        
        break_due_to_size = False

        for idx in range(len(time_pts_list)):
            min_size = time_pts_list[idx].min(axis=0)
            max_size = time_pts_list[idx].max(axis=0)
            if np.abs(max_size[0] - min_size[0]) > MAX_SIZE or np.abs(max_size[1] - min_size[1]) > MAX_SIZE:
                break_due_to_size = True
        
        if break_due_to_size:
            continue

        try:
            
            trajectory, dynamic, trans_list, boxes = ICP4dSolver(time_pts_list, instance_pts, available_times, threshold=3.5, per_frame_move_thresh=0.1)
            dynamic_mask[i_mask] = dynamic
            dynamic_id[i] = dynamic
            trajectories[i] = trajectory
            boxes_dict[i] = boxes
            logic_mask[i_mask] = True
        
        except:
            # Sometimes, there is too few points or skipped frames by occlussion.
            # These problems are not handled in this code.
            continue

    if store_visuals is not None or visualize_path is not None:
        instance_dynamic_mask = cluster_ids * dynamic_mask
        # Transform int ids into colors
        colors = plt.cm.get_cmap('tab20', cluster_ids.max() + 1)
        rgb_colors = colors(instance_dynamic_mask)
        label_colors = plt.cm.get_cmap('jet', int(segmentation.max() + 1))(segmentation.astype('int'))
        dbscan_colors = plt.cm.get_cmap('tab20', cluster_ids.max() + 1)(cluster_ids)

    if visualize_path is not None:
        # Plotting
        plt.close()
        plt.figure(dpi=100, figsize=(10,10))
        plt.plot(pts[~dynamic_mask, 0], pts[~dynamic_mask, 1], 'b.', markersize=.3)
        plt.plot(global_scan[segmentation, 0], global_scan[segmentation, 1], 'r.', markersize=0.1)
        # boxes
        for t_box in gt_box_list:
            for obj_box in t_box:
                plt.plot(obj_box.T[[0,1,2,3,0], 0], obj_box.T[[0,1,2,3,0], 1], 'g-', linewidth=0.1)


        for i in trajectories.keys():
            if dynamic_id[i]:
                pts_i = pts[cluster_ids == i]
                plt.plot(pts_i[:, 0], pts_i[:, 1], '.', markersize=.6)
            

        for i in trajectories.keys():
            # Plot only dynamic trajectories
            if dynamic_id[i]:
                plt.plot(trajectories[i][:,0], trajectories[i][:,1], marker='+', color='k', markersize=3, linestyle='-', linewidth=1.0)
        
        plt.tight_layout()
        plt.axis('equal')
        plt.title("PONE - Sequence")
        plt.savefig(visualize_path, dpi=200)


    TP = np.logical_and(dynamic_mask, segmentation).sum()
    FP = np.logical_and(dynamic_mask, ~segmentation).sum()
    FN = np.logical_and(~dynamic_mask, segmentation).sum()

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    IOU = TP / (TP + FP + FN)

    data = {
        'Metric': ['TP', 'FP', 'FN', 'Precision', 'Recall', 'F1', 'IoU'],
        'Value': [TP, FP, FN, precision, recall, F1, IOU]
    }

    df = pd.DataFrame(data)
    df = df.reset_index(drop=True)

    
    if metric_path is not None:
        # Save metrics to text file
        df.to_csv(metric_path, index=False)
    
    else:
        print(df.T)
        
    
    if store_visuals is not None:
        write_obj(np.concatenate((pts[:,:3], rgb_colors[:,:3]),axis=1), f'{store_visuals}/output_prediction.obj')
        write_obj(np.concatenate((pts[:,:3], dbscan_colors[:,:3]),axis=1), f'{store_visuals}/output_cluster.obj')
        write_obj(np.concatenate((global_scan[:,:3], label_colors[:,:3]),axis=1), f'{store_visuals}/output_label.obj')

    return trajectories, dynamic_id, dynamic_mask




if __name__ == '__main__':
    

    parser = argparse.ArgumentParser()
    parser.add_argument('--job-chunk', type=str, default=0)
    parser.add_argument('--frame-range', type=str, default=2)
    parser.add_argument('--EPS', type=float, default=0.4)
    parser.add_argument('--MAX_ADJACENT_TIMES', type=int, default=20)
    parser.add_argument('--Z_SCALE', type=float, default=0.3)
    parser.add_argument('--MIN_NBR_PTS', type=int, default=20)
    parser.add_argument('--MIN_INCLUDED_TIMES', type=int, default=2)
    parser.add_argument('--MAX_SIZE', type=int, default=8)

    parser.add_argument('--visualize_path', type=str, default=None)
    parser.add_argument('--metric_path', type=str, default=None)
    parser.add_argument('--store_visuals', type=str, default=None)
    args = parser.parse_args()

    GROUND_REMOVE = 0.5
    HEIGHT_REMOVE = 5.5

    DATA_DIR = '/mnt/personal/vacekpa2/data/PONE/zima/'
    
    all_frames = glob.glob(DATA_DIR + '/sequences/*/*.npz')
    init_frame = int(args.job_chunk) * int(args.frame_range)
    
    first_file = all_frames[init_frame]
    sequence = os.path.dirname(first_file)
    start_frame = int(os.path.basename(first_file).split('.')[0])
    end_frame = start_frame + int(args.frame_range)

    try:
        global_scan, segmentation, gt_box_list = load_sequence(start_frame, end_frame, sequence)

    except:
        print('Number of frames is exceeded')
        exit()
    
    trajectories, dynamic_id, dynamic_mask = main(global_scan, segmentation, gt_box_list,
         args.EPS, args.MAX_ADJACENT_TIMES, args.Z_SCALE, args.MIN_NBR_PTS, args.MIN_INCLUDED_TIMES, args.MAX_SIZE,
         args.visualize_path, args.metric_path, args.store_visuals)

    

    if True:
        dyn_trajs = {k: v for k, v in trajectories.items() if dynamic_id[k]}

        def extrapolate_trajectory(trajectory, num_points=2):
            if len(trajectory) < 2:
                raise ValueError("Trajectory must contain at least two points for extrapolation")

            # Calculate the direction vectors for extrapolation
            direction_forward = trajectory[-1] - trajectory[-2]
            direction_backward = trajectory[0] - trajectory[1]

            # Extrapolate forward
            forward_points = [trajectory[-1] + direction_forward * (i + 1) for i in range(num_points)]

            # Extrapolate backward
            backward_points = [trajectory[0] + direction_backward * (i + 1) for i in range(num_points)]

            # Combine the original trajectory with the extrapolated points
            extrapolated_trajectory = np.vstack((backward_points[::-1], trajectory, forward_points))

            return extrapolated_trajectory

        fig, ax = plt.subplots(2)

        for k, v in dyn_trajs.items():
            ax[0].plot(v[:, 0], v[:, 1], label=f"Object {k}", linewidth=1)
            ax[0].text(v[0, 0], v[0, 1], f"Object {k}", fontsize=8, color='black')


        for key in dyn_trajs:
            if len(dyn_trajs[key]) > 2:
                dyn_trajs[key] = extrapolate_trajectory(dyn_trajs[key], num_points=2)

        for k, v in dyn_trajs.items():
            ax[1].plot(v[:, 0], v[:, 1], label=f"Object {k}", linewidth=1)
            ax[1].text(v[0, 0], v[0, 1], f"Object {k}", fontsize=8, color='black')

        ax[1].grid(True)
        ax[0].grid(True)
        plt.savefig('extrapolated_trajectory.png', dpi=200)
