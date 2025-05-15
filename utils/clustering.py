import torch
import hdbscan
import numpy as np
from sklearn.cluster import DBSCAN

from alpine import Alpine


class Clusterer:
    def __init__(self, config):
        self.config = config
        self.clusterer = None
        if config["clustering"]["clustering_method"] == "alpine":
            BBOX = (
                config["alpine"]["BBOX_WEB"]
                if config["alpine"]["bbox_source"] == "web"
                else config["alpine"]["BBOX_DATASET"]
            )
            self.clusterer = Alpine(
                config["fore_classes"],
                BBOX,
                k=config["alpine"]["neighbours"],
                margin=config["alpine"]["margin"],
            )
        elif config["clustering"]["clustering_method"] == "hdbscan":
            self.clusterer = hdbscan.HDBSCAN(
                algorithm="best",
                approx_min_span_tree=True,
                gen_min_span_tree=True,
                metric="euclidean",
                min_cluster_size=config["clustering"]["min_cluster_size"],
                min_samples=None,
            )
        elif config["clustering"]["clustering_method"] == "dbscan":
            self.clusterer = DBSCAN(
                eps=config["clustering"]["epsilon"],
                min_samples=config["clustering"]["min_cluster_size"],
            )
        else:
            raise ValueError(
                f"Unsupported clustering method: {self.config['clustering_method']}"
            )

    def get_semantic_clustering(self, points: torch.Tensor) -> torch.Tensor:
        """
        Perform semantic clustering on input points using DBSCAN.

        Args:
            points (torch.Tensor): Tensor of shape (N, D) where last column contains class labels.
            config (dict): Dictionary with clustering parameters

        Returns:
            torch.Tensor: Cluster labels of shape (N,).
        """
        points_np = points.cpu().numpy()
        labels = np.full(points.shape[0], -1, dtype=np.int64)

        if self.config["clustering"]["clustering_method"] == "alpine":
            labels = self.clusterer.fit_predict(points_np[:, :3], points_np[:, -1]) - 1
        else:
            class_ids, class_counts = np.unique(points_np[:, -1], return_counts=True)
            valid_classes = class_ids[
                class_counts >= self.config["clustering"]["min_cluster_size"]
            ]

            cluster_id = 0
            for class_id in valid_classes:
                mask = points_np[:, -1] == class_id

                class_labels = self.clusterer.fit_predict(points_np[mask, :3])

                # merge labels, outliers are labeled as -1
                updated_labels = class_labels + cluster_id
                updated_labels[class_labels == -1] = -1
                labels[mask] = updated_labels

                unique_labels = np.unique(class_labels)
                cluster_id += (
                    (len(unique_labels) - 1)
                    if -1 in unique_labels
                    else len(unique_labels)
                )

        # keep only the top clusters
        labels = torch.tensor(labels, dtype=torch.int64, device=points.device)

        lbls, counts = torch.unique(labels, return_counts=True)
        valid_mask = lbls != -1
        cluster_info = torch.stack((lbls[valid_mask], counts[valid_mask]), dim=1)
        cluster_info = cluster_info[torch.argsort(cluster_info[:, 1], descending=True)]

        clusters_labels = cluster_info[:self.config["clustering"]["num_clusters"], 0]
        labels[~torch.isin(labels, clusters_labels)] = -1

        return labels
