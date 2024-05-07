from TALFi.config import config
from TALFi.evaluation.eval_detection import ANETdetection
max = config['training']['max_epoch']
predection_path = config['testing']['output_path']
gt = config['evaling']['gt_json']
model_name = config['model']['name']
tious = [0.3, 0.4, 0.5, 0.6, 0.7]
max_mAP = 0
max_epoch = 0
for i in range(10, max+1):
    print("\n", "epoch: ", i)
    prediction = predection_path + "checkpoint" + str(i) + ".json"
    anet_detection = ANETdetection(
        ground_truth_filename=gt,
        prediction_filename=prediction,
        subset='test', tiou_thresholds=tious)
    mAPs, average_mAP, ap = anet_detection.evaluate()
    for (tiou, mAP) in zip(tious, mAPs):
        print("mAP at tIoU {} is {}".format(tiou, mAP.round(4)))
    if average_mAP > max_mAP:
        max_mAP = average_mAP
        max_epoch = i
print("Max mAP is {} at epoch {}".format(max_mAP.round(4), max_epoch))