import cv2
import mmcv
from mmengine.config import Config, DictAction
from mmcv.transforms import Compose
from mmengine.utils import track_iter_progress
from mmdet.registry import VISUALIZERS
from mmdet.apis import init_detector, inference_detector
import glob
import os
import pandas as pd
from tqdm import tqdm


# Specify the path to model config and checkpoint file
config_file = 'configs/gfl/gfl_s50_fpn_2x_coco.py'
checkpoint_file = 'mfd_gfl_s50.pth'


# cfg = Config.fromfile(config_file)

# Build the model from a config file and a checkpoint file
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# Init visualizer
visualizer = VISUALIZERS.build(model.cfg.visualizer)
# The dataset_meta is loaded from the checkpoint and
# then pass to the model in init_detector
visualizer.dataset_meta = model.dataset_meta


images = glob.glob("/home/sindhura/MFD_untype/*")

df = pd.read_csv("/home/sindhura/isMath_pred.csv")

labels = []
bboxes = []
scores = []
num_pred_boxes = []

# for img in images:
for index, row in tqdm(df.iterrows()):
    img = '/home/sindhura/MFD_untype/'+str(row['Image_request_id'])+'.jpg'
    
    result = inference_detector(model, img)
    
    labels.append(result.pred_instances.labels.detach().cpu().numpy())
    bboxes.append(result.pred_instances.bboxes.detach().cpu().numpy())
    scores.append(result.pred_instances.scores.detach().cpu().numpy())
    num_pred_boxes.append(result.pred_instances.labels.size(dim=0))
    
    res = 'untype_'+os.path.basename(img)[:-4]
    
    # Show the results
    img = mmcv.imread(img, channel_order='rgb')
    # img = mmcv.imconvert(img, 'bgr', 'rgb')
    
    visualizer.add_datasample(
        res,
        img,
        data_sample=result,
        draw_gt=False,
        draw_pred=True)

df['pred_labels'] = labels
df['pred_bboxes'] = bboxes
df['pred_scores'] = scores
df['num_pred_boxes'] = num_pred_boxes

df.to_csv("/home/sindhura/isMath_final_pred.csv")