from utils.visualization import visualize_scene

if __name__ == "__main__":
    pcd_dir = "/home/vlkjan6/Documents/diplomka/dataset/PONE/val/"
    labels_dir = "/home/vlkjan6/Documents/diplomka/dataset/PONE/val-preds/"

    visualize_scene(pcd_dir, labels_dir)
