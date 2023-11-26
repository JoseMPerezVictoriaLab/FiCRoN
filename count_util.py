#%%
from PIL import Image, ImageSequence
import numpy as np
from read_lif import Reader as r_lif
import deconv as dconv
import ficron
import cv2

def read_tiff(path, size_img=None):
    """
    path - Path to the multipage-tiff file
    """
    images = []

    with Image.open(path) as img:
        for im in ImageSequence.Iterator(img):
            if size_img is None:
                images.append(np.array(im))
            else:
                images.append(np.array(im.resize((size_img, size_img))))

    img_temp = np.stack(images, axis=-1)

    return img_temp


def read_lif(path):
    series_img = r_lif.Reader('/'.join([dirs, file]))
    for choose_img in series_img:
        yield np.stack([choose_img.getFrame2D(T=0, channel=0, dtype=np.uint16),
                          choose_img.getFrame2D(T=0, channel=1, dtype=np.uint16)], axis=-1)


def croll_image(full_image, idx, image_size):
    # print(full_image.shape, idx)
    return full_image[idx[0]: idx[0] + image_size,
            idx[1]: idx[1] + image_size]


def save_multi_frame_tiff(image_array, size_img, name_file):

    list_img = [Image.fromarray(image_array[:, :, idx])
            for idx in range(image_array.shape[-1])]
    list_img[0].save(name_file, save_all=True,
                     append_images=list_img[1:])

def convert_uint16(img):
    imin = img.min(axis=(0, 1))
    imax = img.max(axis=(0, 1))

    return (65535 * (img - imin) / imax).astype(np.uint16)

def convert_uint8(img):
    imin = img.min(axis=(0, 1))
    imax = img.max(axis=(0, 1))

    return (255. * (img - imin) / imax).astype(np.uint8)


def convert_gray2bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def deconv_img(n_dcv):
    n_dcv_transf = np.moveaxis(n_dcv, [0, 1, 2], [1, 2, 0])
    dcv_img = dconv.decv_img(n_dcv_transf, dcv_kernel)
    dcv_img = convert_uint16(dcv_img)
    return dcv_img


def count_img(img_org, return_map=False, dcv=False):
    if dcv:
        img_org = deconv_img(img_org)
    return (img_org,) + ficron.count_cells(img_org, return_map=return_map)


def merge_img_map(img_org, maps):
    merge_img = img_org.copy()
    size_img = merge_img.shape[:2]

    tmp_cellmap = cv2.resize(maps[1] * 22, size_img, interpolation=cv2.INTER_NEAREST)
    mask_tmp = tmp_cellmap > 150
    merge_img[mask_tmp] = [255, 255, 0]

    tmp_cellmap = cv2.resize(maps[2] * 22, size_img, interpolation=cv2.INTER_NEAREST)
    mask_tmp = tmp_cellmap > 150
    merge_img[mask_tmp] = [255, 0, 0]

    merge_img[:, :, 2] = merge_img[:, :, 2] + cv2.resize(maps[0] * 2, size_img, interpolation=cv2.INTER_NEAREST)

    return merge_img


psf_dapi = read_tiff('PSF/PSF_DAPI_63x.tif')
psf_cellMask = read_tiff('PSF/PSF_cellmask_63x.tif')
dcv_kernel = np.concatenate([psf_cellMask, psf_dapi], axis=-1)
dcv_kernel = np.moveaxis(dcv_kernel, [0, 1, 2], [1, 2, 0])