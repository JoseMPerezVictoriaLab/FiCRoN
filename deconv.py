#%%
from flowdec import data as fd_data
from flowdec import restoration as fd_restoration
import numpy as np

#%%
algo = fd_restoration.RichardsonLucyDeconvolver(n_dims=2).initialize()

def decv_img(img_orig, kernel):
    res_1 = algo.run(fd_data.Acquisition(data=img_orig[0, :, :], kernel=kernel[0, :, :]), niter=30).data
    res_2 = algo.run(fd_data.Acquisition(data=img_orig[1, :, :], kernel=kernel[1, :, :]), niter=30).data
    res_otro = np.stack([res_1, res_2], axis=-1)

    return res_otro

