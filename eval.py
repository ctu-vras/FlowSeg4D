import numpy as np

OFFSET = 2 ** 32

class EvalPQ4D:
    def __init__(self, num_classes, ignore=None, offset=OFFSET, min_points=30):
        self.num_classes = num_classes
        ignore = ignore or []
        self.ignore = np.array(ignore, dtype=np.int32)
        self.include = np.array([n for n in range(num_classes) if n not in ignore], dtype=np.int32)
        self.offset = offset
        self.min_points = min_points
        self.eps = 1e-15

        self.reset()

    ### MISCELLANEOUS
    def reset(self):
        self.conf_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int32)

        self.sequences = []
        self.preds = {}
        self.gts = {}
        self.intersects = {}

        self.pan_aq = np.zeros(self.num_classes, dtype=np.float32)

    def update_dict(self, stat_dict, unique_ids, unique_counts):
        for idx, count in zip(unique_ids, unique_counts):
            if idx == 1:
                continue
            if idx in stat_dict:
                stat_dict[idx] += count
            else:
                stat_dict[idx] = count

    def update(self, seq, pred_sem, pred_inst, gt_sem, gt_inst):
        self.update_iou(pred_sem, gt_sem)
        self.update_pan(seq, pred_sem, pred_inst, gt_sem, gt_inst)

    ### IoU
    def update_iou(self, pred_sem, gt_sem):
        idxs = np.stack([pred_sem, gt_sem], axis=0)

        np.add.at(self.conf_matrix, tuple(idxs), 1)

    def get_iou_vals(self):
        conf_matrix = self.conf_matrix.copy()
        conf_matrix[:, self.ignore] = 0

        tp = conf_matrix.diagonal()
        fp = conf_matrix.sum(axis=1) - tp
        fn = conf_matrix.sum(axis=0) - tp

        return tp, fp, fn

    def get_iou(self):
        tp, fp, fn = self.get_iou_vals()

        intersection = tp
        union = tp + fp + fn

        num_classes = np.count_nonzero(union)

        union = np.maximum(union, self.eps)
        iou = intersection / union
        iou_mean = np.sum(iou) / num_classes

        prec = tp / np.maximum(tp + fp, self.eps)
        iou_p = np.sum(prec) / num_classes
        recall = tp / np.maximum(tp + fn, self.eps)
        iou_r = np.sum(recall) / num_classes

        return iou, iou_mean, iou_p, iou_r

    ### Panoptic
    def update_pan(self, seq, pred_sem, pred_inst, gt_sem, gt_inst):
        if seq not in self.sequences:
            self.sequences.append(seq)
            self.preds[seq] = {}
            self.gts[seq] = [{} for _ in range(self.num_classes)]
            self.intersects[seq] = [{} for _ in range(self.num_classes)]

        pred_inst += 1
        gt_inst += 1

        for class_id in self.ignore:
            gt_mask = gt_sem != class_id

            pred_sem = pred_sem[gt_mask]
            pred_inst = pred_inst[gt_mask]
            gt_sem = gt_sem[gt_mask]
            gt_inst = gt_inst[gt_mask]

        cl_preds = self.preds[seq]
        for class_id in self.include:
            cl_gts = self.gts[seq][class_id]
            cl_intersects = self.intersects[seq][class_id]

            pred_inst_in_cl = pred_inst * (pred_sem == class_id)
            gt_inst_in_cl = gt_inst * (gt_sem == class_id)

            unique_gt, counts_gt = np.unique(gt_inst_in_cl[gt_inst_in_cl > 0], return_counts=True)
            self.update_dict(cl_gts, unique_gt[counts_gt > self.min_points], counts_gt[counts_gt > self.min_points])

            mask = np.zeros_like(gt_inst_in_cl)
            for valid_id in unique_gt[counts_gt > self.min_points]:
                mask = np.logical_or(mask, gt_inst_in_cl == valid_id)
            gt_inst_in_cl = gt_inst_in_cl * mask

            unique_pred, counts_pred = np.unique(pred_inst_in_cl[pred_inst_in_cl > 0], return_counts=True)
            self.update_dict(cl_preds, unique_pred[counts_pred > self.min_points], counts_pred[counts_pred > self.min_points])

            valid_combos = np.logical_and(pred_inst > 0, gt_inst_in_cl > 0)
            offset_combos = pred_inst[valid_combos] + self.offset * gt_inst_in_cl[valid_combos]
            unique_combos, counts_combos = np.unique(offset_combos, return_counts=True)
            self.update_dict(cl_intersects, unique_combos, counts_combos)

    ### PQ4D
    def compute(self):
        precs = []
        recalls = []
        num_tubes = [0] * self.num_classes

        for seq in self.sequences:
            cl_preds = self.preds[seq]
            for class_id in self.include:
                cl_gts = self.gts[seq][class_id]
                cl_intersects = self.intersects[seq][class_id]
                outer_sum_iou = 0.0
                num_tubes[class_id] += len(cl_gts)

                for gt_id, gt_size in cl_gts.items():
                    inner_sum_iou = 0.0
                    for pr_id, pr_size in cl_preds.items():
                        TPA_key = pr_id + self.offset * gt_id
                        if TPA_key in cl_intersects:
                            TPA = cl_intersects[TPA_key]
                            prec = TPA / float(pr_size)
                            recall = TPA / float(gt_size)
                            inner_sum_iou += TPA * (TPA / (gt_size + pr_size - TPA))
                            precs.append(prec)
                            recalls.append(recall)
                    outer_sum_iou += float(inner_sum_iou) / float(gt_size)
                self.pan_aq[class_id] += outer_sum_iou

        AQ_overall = np.sum(self.pan_aq) / np.sum(num_tubes)
        AQ = self.pan_aq / np.maximum(num_tubes, self.eps)

        iou, iou_mean, iou_p, iou_r = self.get_iou()

        AQ_p = np.mean(precs)
        AQ_r = np.mean(recalls)

        PQ4D = np.sqrt(AQ_overall * iou_mean)

        return PQ4D, AQ_overall, AQ, AQ_p, AQ_r, iou, iou_mean, iou_p, iou_r


### TEST
if __name__ == "__main__":
    classes = 3  # ignore, car, truck
    cl_strings = ["ignore", "car", "truck"]
    ignore = [0]  # only ignore ignore class

    sem_gt = np.zeros(20, dtype=np.int32)
    sem_gt[5:10] = 1
    sem_gt[10:] = 2

    inst_gt = np.zeros(20, dtype=np.int32)
    inst_gt[5:10] = 1
    inst_gt[10:] = 1
    inst_gt[15:] = 2

    #we have 3 instance 1 car, 2 truck as gt
    sem_pred = np.zeros(20, dtype=np.int32)
    sem_pred[5:10] = 1
    sem_pred[10:15] = 2
    sem_pred[15:] = 1

    inst_pred = np.zeros(20, dtype=np.int32)
    inst_pred[5:10] = 1
    inst_pred[10:] = 2

    # evaluator
    class_evaluator = EvalPQ4D(3, ignore, OFFSET, 1)
    class_evaluator.update(1, sem_pred, inst_pred, sem_gt, inst_gt)
    PQ4D, AQ_ovr, AQ, AQ_p, AQ_r, iou, iou_mean, iou_p, iou_r = class_evaluator.compute()
    np.testing.assert_equal(PQ4D, np.sqrt(1.0/3))
    np.testing.assert_equal(AQ_ovr, 2.0/3)
    np.testing.assert_equal(AQ, [0, 1.0, 0.5])
    np.testing.assert_equal(AQ_p, 2.0/3)
    np.testing.assert_equal(AQ_r, 1.0)
    np.testing.assert_equal(iou, [0, 0.5, 0.5])
    np.testing.assert_equal(iou_mean, 0.5)