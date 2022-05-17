import numpy as np

def color_map(N=7, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)

    cmap[0] = np.array([96, 96, 96])  # impervious surface
    cmap[1] = np.array([204, 204, 0])  # agriculture
    cmap[2] = np.array([0, 204, 0])  # forest & other
    cmap[3] = np.array([0, 0, 153])  # wetland
    cmap[4] = np.array([153, 76, 0])  # soil
    cmap[5] = np.array([0, 128, 255])  # water

    #snow rgb in the paper 138, 178, 198
    cmap = cmap/255 if normalized else cmap
    return cmap
