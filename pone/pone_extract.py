import numpy as np
import  glob
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def get_corners(obj):
    """
    Convert offset representation of the box to XY corners.
    The corners are ordered counter-clockwise starting from rear right.
        3---------2\
        |         | 〉
        0---------1/
    """
    x, phi, bb = obj['x'][[0, 1]], obj['x'][2], obj['bb']
    v = np.array([np.cos(phi), np.sin(phi)])
    w = np.array([-v[1], v[0]])
    
    return np.array([bb[0] * v + bb[1] * w + x,
                        bb[2] * v + bb[1] * w + x,
                        bb[2] * v + bb[3] * w + x,
                        bb[0] * v + bb[3] * w + x])
    

if __name__ == '__main__':
    
    DATA_DIR = '/mnt/personal/vacekpa2/data/PONE/zima/'
    ANNOTAION_DIR = DATA_DIR + '/annotations/'
    RAW_DATA_DIR = DATA_DIR + '/raw/'


    anno_keys = ['odom_list', 'vehicle_list', 'lanes']
    file_keys = ['odom_list', 'scan_list']

    for file in sorted(glob.glob(RAW_DATA_DIR + '/*.npz')):
        
        f = np.load(file, allow_pickle=True)
        name_anno = file.split('/')[-1][5:][:-9] + '.npz'
        
        one_annotation = np.load(ANNOTAION_DIR + name_anno, allow_pickle=True)

        # scanpoints, timestamps and odometry:    
        scan_list, odom_list = f['scan_list'], f['odom_list']

        odom_list['t']  # UTC timestamps in seconds
        odom_list['v'], odom_list['yaw_rate']  # velocity [mps] and yaw rate [radps]
        odom_list['transformation']  # transformation matrices from ego to world coordinates. World coordinates are defined by position in t=odom_list['t'][0]

        os.makedirs(DATA_DIR + f'/sequences/{name_anno[:-4]}', exist_ok=True)
        # scan_list is list of scans
        for frame_idx in tqdm(range(len(scan_list))):
            data_dict = {}

            scan = scan_list[frame_idx]
            scan['x']  # 2D array of XY [m]
            scan['z'], scan['i']  # elevation [m] and intensity normalized to [0, 1]


            # vehicle annotation:
            vehicle_list = one_annotation['vehicle_list']

            # vehicle_list is list of object tracks, annotation range is lateral (-20, 20), longitudinal (0, 200)
            # vehicle_list[0] is the first object track
                
            # for a given scan frame, the corresponding vehicles can be retrieved based on timestamp
            scan, t_frame = scan_list[frame_idx], odom_list['t'][frame_idx]
            obj_frame = [get_corners(obj[obj['t']==[t_frame]][0]) for obj in vehicle_list if (obj['t']==[t_frame]).any()]
            

            data_dict['scan'] = np.stack((scan['x'][:,0], scan['x'][:,1], scan['z'], scan['i']), axis=1)
            data_dict['box_corners'] = obj_frame
            data_dict['timestamp'] = t_frame
            data_dict['transformation'] = one_annotation['odom_list'][frame_idx]['transformation']

            np.savez(DATA_DIR + f'/sequences/{name_anno[:-4]}/{frame_idx:04d}.npz', **data_dict)

