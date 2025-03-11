import numpy as np
import random
import glob
import warnings
import parmap
import os
import itertools
import shutil
import argparse
import copy
import sys
import yaml
import json
import torch
import open3d as o3d
import seaborn as sns
import multiprocessing
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objs as go
from utils_visualization import visualize_pcd, visualize_pcd_multiple, visualize_pcd_plotly, draw_registration_result
from utils_match import match_pcds
from utils_helper import transform_points
from utils_cluster import cluster_pcd
from timeit import default_timer as timer
warnings.filterwarnings('ignore')
np.set_printoptions(threshold=sys.maxsize)

def track(args, point_src, point_dst, label_src, label_dst):
    pairs, transformations = match_pcds(args, point_src, point_dst, label_src, label_dst) 
    if args.if_verbose:
        print(f'match_pcds pairs: {torch.round(pairs[:, 0:2], decimals=2)}')
    return pairs, transformations
