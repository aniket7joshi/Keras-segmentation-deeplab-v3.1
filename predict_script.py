from tensorflow import keras
# from train_script import input_shape, num_classes, backbone, get_uncompiled_model
from PIL import Image
import cv2
import numpy as np

input_shape = (300,300,3)
height, width, _ = input_shape

# model = get_uncompiled_model(input_shape, num_classes, backbone)
# model.load_weights('/mnt/mydata/dataset/Playment_top_5_dataset/deeplab_top_5_classes_focal_loss_15.h5', by_name=True)
model = keras.models.load_model('/tmp/test_dataset')
print(model.summary())



image_path = '/mnt/mydata/dataset/Playment_top_5_dataset/test_images/bdd_80c62ee8-96e3f3bf.jpg'

image = np.array(Image.open(image_path))
resized_image = cv2.resize(image, (width, height))
#cv2.imwrite('image.jpg', resized_image)

image = image.astype(np.float32)/255


prediction = model.predict(np.expand_dims(resized_image, axis=0))[0]
preds1 = np.argmax(model.predict(image), -1)[0].reshape(input_shape)


