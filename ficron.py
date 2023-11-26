import numpy as np
import tensorflow as tf
from tensorflow.keras import models
import cv2

#%%

def count_cells(img_org, return_map=False):

    size_img = np.array(img_org.shape[:2])

    size_img -= (size_img % 32)
    img_org = cv2.resize(img_org, (size_img[1], size_img[0]), interpolation=cv2.INTER_AREA)
    img1 = cv2.resize(img_org, (0, 0), fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)
    img1 = img1 / 65535.
    resul_predict = np.zeros((3,))

    mask_p = model_p.predict(np.array([img1]))[0, :, :, 0]
    mask_p = mask_p / 3400.
    resul_predict[0] = mask_p.sum()
    mask_p = cv2.resize(mask_p, (0, 0), fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA) * 4
    mask_p = mask_p * 3400.

    img1 = cv2.resize(img1, (0, 0), fx=1/2, fy=1/2, interpolation=cv2.INTER_AREA)

    mask_c = model_c.predict(np.array([img1]))[0, :, :, 0]
    resul_predict[1] = mask_c.sum() / 1031.

    img_aux = np.concatenate((np.expand_dims(img1[:, :, 0], axis=-1), np.expand_dims(mask_p, axis=-1),
                              np.expand_dims(mask_c, axis=-1)), axis=-1)
    mask_i = model_i.predict(np.array([img_aux]))[0, :, :, 0]
    thresh_inf = 0.8
    mask_i[mask_i >= thresh_inf] = 1
    mask_i[mask_i < (1 - thresh_inf)] = 0
    mask_i = mask_i * mask_c

    resul_predict[2] = mask_i.sum() / 1031.

    if return_map:
        return [mask_p, mask_c, mask_i], resul_predict
    else:
        return resul_predict


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

path_model_p = 'Models/model4_parasite.h5'
path_model_cel = 'Models/model4_macrophage.h5'
path_model_inf = 'Models/model4_infected.h5'

model_p = models.load_model(path_model_p)
model_c = models.load_model(path_model_cel)
model_i = models.load_model(path_model_inf)
