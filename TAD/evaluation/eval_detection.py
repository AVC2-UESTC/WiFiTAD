# import json
# import numpy as np
# import pandas as pd
# from joblib import Parallel, delayed

# from .utils_eval import get_blocked_videos
# from .utils_eval import interpolated_prec_rec
# from .utils_eval import segment_iou

# import warnings
# warnings.filterwarnings("ignore", message="numpy.dtype size changed")
# warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# class ANETdetection(object):
#     GROUND_TRUTH_FIELDS = ['database']
#     PREDICTION_FIELDS = ['results', 'version', 'external_data']

#     def __init__(self, ground_truth_filename=None, prediction_filename=None,
#                  ground_truth_fields=GROUND_TRUTH_FIELDS,
#                  prediction_fields=PREDICTION_FIELDS,
#                  tiou_thresholds=np.linspace(0.5, 0.95, 10), 
#                  subset='validation', verbose=False, 
#                  check_status=False):
#         if not ground_truth_filename:
#             raise IOError('Please input a valid ground truth file.')
#         if not prediction_filename:
#             raise IOError('Please input a valid prediction file.')
#         self.subset = subset
#         self.tiou_thresholds = tiou_thresholds
#         self.verbose = verbose
#         self.gt_fields = ground_truth_fields
#         self.pred_fields = prediction_fields
#         self.ap = None
#         self.check_status = check_status

#         if self.check_status:
#             self.blocked_videos = get_blocked_videos()
#         else:
#             self.blocked_videos = list()

#         self.ground_truth, self.activity_index, self.video_lst = self._import_ground_truth(ground_truth_filename)
#         self.prediction = self._import_prediction(prediction_filename)

#         if self.verbose:
#             print('[INIT] Loaded annotations from {} subset.'.format(subset))
#             nr_gt = len(self.ground_truth)
#             print('\tNumber of ground truth instances: {}'.format(nr_gt))
#             nr_pred = len(self.prediction)
#             print('\tNumber of predictions: {}'.format(nr_pred))
#             print('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

#     def _import_ground_truth(self, ground_truth_filename):
#         """Reads ground truth file, checks if it is well formatted, and returns
#            the ground truth instances and the activity classes.
#         """
#         with open(ground_truth_filename, 'r') as fobj:
#             data = json.load(fobj)
#         if not all([field in data.keys() for field in self.gt_fields]):
#             raise IOError('Please input a valid ground truth file.')

#         activity_index, cidx = {}, 0
#         video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
#         for videoid, v in data['database'].items():
#             if self.subset != v['subset']:
#                 continue
#             if videoid in self.blocked_videos:
#                 continue
#             for ann in v['annotations']:
#                 if ann['label'] not in activity_index:
#                     activity_index[ann['label']] = cidx
#                     cidx += 1
#                 video_lst.append(videoid)
#                 t_start_lst.append(float(ann['segment'][0]))
#                 t_end_lst.append(float(ann['segment'][1]))
#                 label_lst.append(activity_index[ann['label']])

#         ground_truth = pd.DataFrame({'video-id': video_lst,
#                                      't-start': t_start_lst,
#                                      't-end': t_end_lst,
#                                      'label': label_lst})
#         if self.verbose:
#             print(activity_index)
#         return ground_truth, activity_index, video_lst

#     def _import_prediction(self, prediction_filename):
#         """Reads prediction file, checks if it is well formatted, and returns
#            the prediction instances.
#         """
#         with open(prediction_filename, 'r') as fobj:
#             data = json.load(fobj)
#         if not all([field in data.keys() for field in self.pred_fields]):
#             raise IOError('Please input a valid prediction file.')

#         video_lst, t_start_lst, t_end_lst = [], [], []
#         label_lst, score_lst = [], []
#         for videoid, v in data['results'].items():
#             if videoid in self.blocked_videos:
#                 continue
#             if videoid not in self.video_lst:
#                 continue
#             for result in v:
#                 if result['label'] not in self.activity_index:
#                     continue
#                 label = self.activity_index[result['label']]
#                 video_lst.append(videoid)
#                 t_start_lst.append(float(result['segment'][0]))
#                 t_end_lst.append(float(result['segment'][1]))
#                 label_lst.append(label)
#                 score_lst.append(result['score'])
#         prediction = pd.DataFrame({'video-id': video_lst,
#                                    't-start': t_start_lst,
#                                    't-end': t_end_lst,
#                                    'label': label_lst,
#                                    'score': score_lst})
#         return prediction

#     def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
#         """Get all predictions of the given label. Return empty DataFrame if there
#         are no predictions with the given label.
#         """
#         try:
#             return prediction_by_label.get_group(cidx).reset_index(drop=True)
#         except:
#             if self.verbose:
#                 print('Warning: No predictions of label \'%s\' were provided.' % label_name)
#             return pd.DataFrame()

#     def wrapper_compute_average_precision(self):
#         """Computes average precision for each class in the subset."""
#         ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

#         ground_truth_by_label = self.ground_truth.groupby('label')
#         prediction_by_label = self.prediction.groupby('label')

#         results = Parallel(n_jobs=len(self.activity_index))(
#             delayed(compute_average_precision_detection)(
#                 ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
#                 prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
#                 label_name=label_name,
#                 tiou_thresholds=self.tiou_thresholds,
#             ) for label_name, cidx in self.activity_index.items())

#         for i, cidx in enumerate(self.activity_index.values()):
#             ap[:, cidx] = results[i]

#         return ap
    
#     def compute_accuracy(self, ground_truth, prediction, tiou_thresholds):
#         """计算多个 tIoU 阈值下的定位准确率、分类准确率和召回率。"""
#         if prediction.empty:
#             return [0.0] * len(tiou_thresholds), [0.0] * len(tiou_thresholds), [0.0] * len(tiou_thresholds)

#         ground_truth_gbvn = ground_truth.groupby('video-id')

#         n_gt = float(len(ground_truth))
#         npred = float(len(prediction))

#         correct_localizations = [0] * len(tiou_thresholds)
#         correct_classifications = [0] * len(tiou_thresholds)
#         total_iou_successful = [0.0] * len(tiou_thresholds)  # To store total IoU of successful localizations

#         gt_matched = np.zeros((len(tiou_thresholds), len(ground_truth)), dtype=bool)

#         prediction_values = prediction[['video-id', 't-start', 't-end', 'label']].values

#         video_offsets = {}
#         offset = 0
#         for video_id in ground_truth['video-id'].unique():
#             video_offsets[video_id] = offset
#             offset += len(ground_truth_gbvn.get_group(video_id))

#         for video_id, t_start_pred, t_end_pred, label_pred in prediction_values:
#             try:
#                 ground_truth_videoid = ground_truth_gbvn.get_group(video_id)
#             except KeyError:
#                 continue

#             gt_values = ground_truth_videoid[['t-start', 't-end', 'label']].values
#             base_idx = video_offsets[video_id]

#             for tidx, tiou_threshold in enumerate(tiou_thresholds):
#                 max_iou = 0
#                 max_idx = -1

#                 # Calculate IoU for each ground truth segment
#                 for idx, (t_start_gt, t_end_gt, label_gt) in enumerate(gt_values):
#                     iou = segment_iou(
#                         np.array([t_start_pred, t_end_pred]),
#                         np.array([[t_start_gt, t_end_gt]])
#                     )[0]
#                     if iou > max_iou:
#                         max_iou = iou
#                         max_idx = idx

#                 # Check if the max IoU is above the threshold
#                 correct_localizations[tidx] += 1
#                 if max_iou >= tiou_threshold:
#                     correct_classifications[tidx] += 1
#                     total_iou_successful[tidx] += max_iou  # Add IoU to total for successful localization

#                     # if label_pred == gt_values[max_idx, 2]:
#                     #     correct_classifications[tidx] += 1

#                     if max_iou >= tiou_threshold and not gt_matched[tidx, base_idx + max_idx]:
#                         if label_pred == gt_values[max_idx, 2]:
#                             gt_matched[tidx, base_idx + max_idx] = True

#         true_positives = [np.sum(gt_matched[tidx]) for tidx in range(len(tiou_thresholds))]
#         localization_accuracies = [
#             total_iou_successful[tidx] / correct_localizations[tidx] if correct_localizations[tidx] > 0 else 0
#             for tidx in range(len(tiou_thresholds))
#         ]
#         classification_accuracies = [
#             correct_classifications[tidx] / npred if correct_localizations[tidx] > 0 else 0
#             for tidx in range(len(tiou_thresholds))
#         ]
        
        
#         # classification_accuracies = [
#         #     cc / cl if cl > 0 else 0 for cc, cl in zip(correct_classifications, correct_localizations)
#         # ]
#         recalls = [tp / n_gt if n_gt > 0 else 0 for tp in true_positives]

#         return localization_accuracies, classification_accuracies, recalls

#     def evaluate(self, compute_additional_metrics=False):
#         """Evaluates a prediction file for multiple tIoU thresholds."""

#         self.ap = self.wrapper_compute_average_precision()
#         self.mAP = self.ap.mean(axis=1)
#         self.average_mAP = self.mAP.mean()

#         if compute_additional_metrics:
#             tiou = [0.3, 0.4, 0.5, 0.6, 0.7]
#             loc_acc, cls_acc, rec_acc = self.compute_accuracy(self.ground_truth, self.prediction, tiou)
#             return self.mAP, self.average_mAP, self.ap, loc_acc, cls_acc, rec_acc

#         return self.mAP, self.average_mAP, self.ap


# def compute_average_precision_detection(ground_truth, prediction, label_name, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
#     """Compute average precision (detection task) between ground truth and
#     predictions data frames. If multiple predictions occur for the same
#     predicted segment, only the one with the highest score is matched as
#     true positive.
#     """
#     ap = np.zeros(len(tiou_thresholds))
#     if prediction.empty:
#         return ap

#     npos = float(len(ground_truth))
#     lock_gt = np.ones((len(tiou_thresholds), len(ground_truth))) * -1

#     sort_idx = prediction['score'].values.argsort()[::-1]
#     prediction = prediction.loc[sort_idx].reset_index(drop=True)

#     tp = np.zeros((len(tiou_thresholds), len(prediction)))
#     fp = np.zeros((len(tiou_thresholds), len(prediction)))

#     ground_truth_gbvn = ground_truth.groupby('video-id')

#     for idx, this_pred in prediction.iterrows():

#         try:
#             ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
#         except KeyError:
#             fp[:, idx] = 1
#             continue

#         this_gt = ground_truth_videoid.reset_index()
#         tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
#                                this_gt[['t-start', 't-end']].values)

#         tiou_sorted_idx = tiou_arr.argsort()[::-1]
#         for tidx, tiou_thr in enumerate(tiou_thresholds):
#             for jdx in tiou_sorted_idx:
#                 if tiou_arr[jdx] < tiou_thr:
#                     fp[tidx, idx] = 1
#                     break
#                 if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
#                     continue

#                 tp[tidx, idx] = 1
#                 lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
#                 break

#             if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
#                 fp[tidx, idx] = 1

#     tp_cumsum = np.cumsum(tp, axis=1).astype(np.float64)
#     fp_cumsum = np.cumsum(fp, axis=1).astype(np.float64)
#     recall_cumsum = tp_cumsum / npos
#     precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

#     for tidx in range(len(tiou_thresholds)):
#         ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx, :], recall_cumsum[tidx, :])

#     return ap


import json
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .utils_eval import get_blocked_videos
from .utils_eval import interpolated_prec_rec
from .utils_eval import segment_iou

import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

import pandas as pd

class ANETdetection(object):
    GROUND_TRUTH_FIELDS = ['database']
    # GROUND_TRUTH_FIELDS = ['database', 'taxonomy', 'version']
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filename=None, prediction_filename=None,
                 ground_truth_fields=GROUND_TRUTH_FIELDS,
                 prediction_fields=PREDICTION_FIELDS,
                 tiou_thresholds=np.linspace(0.5, 0.95, 10), 
                 subset='validation', verbose=False, 
                 check_status=False):
        if not ground_truth_filename:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')
        self.subset = subset
        self.tiou_thresholds = tiou_thresholds
        self.verbose = verbose
        self.gt_fields = ground_truth_fields
        self.pred_fields = prediction_fields
        self.ap = None
        self.check_status = check_status
        # Retrieve blocked videos from server.

        if self.check_status:
            self.blocked_videos = get_blocked_videos()
        else:
            self.blocked_videos = list()

        # Import ground truth and predictions.
        self.ground_truth, self.activity_index, self.video_lst = self._import_ground_truth(
            ground_truth_filename)
        self.prediction = self._import_prediction(prediction_filename)

        if self.verbose:
            print ('[INIT] Loaded annotations from {} subset.'.format(subset))
            nr_gt = len(self.ground_truth)
            print ('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print ('\tNumber of predictions: {}'.format(nr_pred))
            print ('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

    def _import_ground_truth(self, ground_truth_filename):
        """Reads ground truth file, checks if it is well formatted, and returns
           the ground truth instances and the activity classes.

        Parameters
        ----------
        ground_truth_filename : str
            Full path to the ground truth json file.

        Outputs
        -------
        ground_truth : df
            Data frame containing the ground truth instances.
        activity_index : dict
            Dictionary containing class index.
        """
        with open(ground_truth_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format
        if not all([field in data.keys() for field in self.gt_fields]):
            raise IOError('Please input a valid ground truth file.')

        # Read ground truth data.
        activity_index, cidx = {}, 0
        video_lst, t_start_lst, t_end_lst, label_lst = [], [], [], []
        for videoid, v in data['database'].items():
            # print(v)
            if self.subset != v['subset']:
                continue
            if videoid in self.blocked_videos:
                continue
            for ann in v['annotations']:
                if ann['label'] not in activity_index:
                    activity_index[ann['label']] = cidx
                    cidx += 1
                video_lst.append(videoid)
                t_start_lst.append(float(ann['segment'][0]))
                t_end_lst.append(float(ann['segment'][1]))
                label_lst.append(activity_index[ann['label']])

        ground_truth = pd.DataFrame({'video-id': video_lst,
                                     't-start': t_start_lst,
                                     't-end': t_end_lst,
                                     'label': label_lst})
        if self.verbose:
            print(activity_index)
        return ground_truth, activity_index, video_lst

    def _import_prediction(self, prediction_filename):
        """Reads prediction file, checks if it is well formatted, and returns
           the prediction instances.

        Parameters
        ----------
        prediction_filename : str
            Full path to the prediction json file.

        Outputs
        -------
        prediction : df
            Data frame containing the prediction instances.
        """
        with open(prediction_filename, 'r') as fobj:
            data = json.load(fobj)
        # Checking format...
        if not all([field in data.keys() for field in self.pred_fields]):
            raise IOError('Please input a valid prediction file.')

        # Read predictions.
        video_lst, t_start_lst, t_end_lst = [], [], []
        label_lst, score_lst = [], []
        for videoid, v in data['results'].items():
            if videoid in self.blocked_videos:
                continue
            if videoid not in self.video_lst:
                continue
            for result in v:
                if result['label'] not in self.activity_index:
                    continue
                label = self.activity_index[result['label']]
                video_lst.append(videoid)
                t_start_lst.append(float(result['segment'][0]))
                t_end_lst.append(float(result['segment'][1]))
                label_lst.append(label)
                score_lst.append(result['score'])
        prediction = pd.DataFrame({'video-id': video_lst,
                                   't-start': t_start_lst,
                                   't-end': t_end_lst,
                                   'label': label_lst,
                                   'score': score_lst})
        return prediction

    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            if self.verbose:
                print ('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), len(self.activity_index)))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')

        results = Parallel(n_jobs=len(self.activity_index))(
                    delayed(compute_average_precision_detection)(
                        ground_truth=ground_truth_by_label.get_group(cidx).reset_index(drop=True),
                        prediction=self._get_predictions_with_label(prediction_by_label, label_name, cidx),
                        label_name=label_name,
                        tiou_thresholds=self.tiou_thresholds,
                    ) for label_name, cidx in self.activity_index.items())

        for i, cidx in enumerate(self.activity_index.values()):
            ap[:,cidx] = results[i]

        return ap

    def evaluate(self):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.ap = self.wrapper_compute_average_precision()

        self.mAP = self.ap.mean(axis=1)
        self.average_mAP = self.mAP.mean()

        if self.verbose:
            print ('[RESULTS] Performance on ActivityNet detection task.')
            print ('Average-mAP: {}'.format(self.average_mAP))
            
        return self.mAP, self.average_mAP, self.ap


def compute_average_precision_detection(ground_truth, prediction, label_name, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_thresholds : 1darray, optional
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = np.zeros(len(tiou_thresholds))
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(tiou_thresholds),len(ground_truth))) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.loc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(tiou_thresholds), len(prediction)))
    fp = np.zeros((len(tiou_thresholds), len(prediction)))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[:, idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for tidx, tiou_thr in enumerate(tiou_thresholds):
            for jdx in tiou_sorted_idx:
                if tiou_arr[jdx] < tiou_thr:
                    fp[tidx, idx] = 1
                    break
                if lock_gt[tidx, this_gt.loc[jdx]['index']] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[tidx, idx] = 1
                lock_gt[tidx, this_gt.loc[jdx]['index']] = idx
                break

            if fp[tidx, idx] == 0 and tp[tidx, idx] == 0:
                fp[tidx, idx] = 1

    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float64)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float64)
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for tidx in range(len(tiou_thresholds)):
        ap[tidx] = interpolated_prec_rec(precision_cumsum[tidx,:], recall_cumsum[tidx,:])

    # pos = 0
    # my_list = [float(len(ground_truth)), float(tp_cumsum[pos][-1]), float(fp_cumsum[pos][-1]), float(npos - tp_cumsum[pos][-1]), precision_cumsum[pos][-1].round(2), recall_cumsum[pos][-1].round(2), ap[pos].round(2)]
    # df = pd.read_csv('data.csv')
    # df[label_name] = my_list
    # df.to_csv('data.csv', index=False)
    return ap