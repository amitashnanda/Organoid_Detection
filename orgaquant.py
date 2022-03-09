# import dependencies
import tensorflow as tf
from tensorflow import keras
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image, adjust_contrast
from keras_retinanet.utils.visualization import draw_box
import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
# Path to trained model
# model_path = "orgaquant_intestinal_v3"
def load_orga_model():
  assert os.path.isdir("trained_models")
  assert os.path.isfile("trained_models/orgaquant_intestinal_v3.h5")
  f = h5py.File("trained_models/orgaquant_intestinal_v3.h5")
  return models.load_model("trained_models/orgaquant_intestinal_v3.h5")

  # return models.load_model(os.path.join('trained_models', model_path + '.h5'), backbone_name='resnet50')
model = load_orga_model()
# Path to test image folder
folder_path = 'test_folder'
imagelist=[]
for root, directories, filenames in os.walk(folder_path):
  imagelist = imagelist + [os.path.join(root,x) for x in filenames if x.endswith(('.jpg','.tif','.TIF', '.png', '.jpeg', '.tiff'))]
# Between 0 and len(imagelist)
sample_image = 0
# Between 800 and 2000. Higher better
min_side = 1200
# Between 1 and 3. Higher better
contrast = 1.5
# Between 0 and 1.
threshold = 0.85
 
for i, filename in enumerate(imagelist):
   try:
       #IMAGE_PATH = os.path.join(root,filename)
       IMAGE_PATH = filename
       # load image
       image = read_image_bgr(IMAGE_PATH)
 
       # copy to draw on
       draw = image.copy()
       draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
 
       # preprocess image for network
       image = adjust_contrast(image,contrast)
       image = preprocess_image(image)
       image, scale = resize_image(image, min_side=min_side, max_side=2048)
 
       # process image
       boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
 
       # correct for image scale
       boxes /= scale
 
       out = np.empty((0,4), dtype=np.float32)
 
       # visualize detections
       for box, score, label in zip(boxes[0], scores[0], labels[0]):
           # scores are sorted so we can break
           if score < threshold:
               break
           out = np.append(out, box.reshape(1,4), axis=0)
 
           b = box.astype(int)
           draw_box(draw, b, color=(255, 0, 255))
 
       output = pd.DataFrame(out,columns=['x1', 'y1', 'x2', 'y2'], dtype=np.int16)
       output['Diameter 1 (Pixels)'] = output['x2'] - output['x1']
       output['Diameter 2 (Pixels)'] = output['y2'] - output['y1']
       output.to_csv(IMAGE_PATH + '.csv', index=False)
       plt.imsave(IMAGE_PATH + '_detected.png', draw)
   except:
       pass
 
 
 
