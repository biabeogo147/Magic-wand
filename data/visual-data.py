import numpy as np
import matplotlib.pyplot as plt

# data_npz = np.load('.npz/sketchrnn_circle.full.npz', allow_pickle=True, encoding='latin1')
data_npy = np.load('.npy/circle.npy')

image_data = data_npy[100].reshape(28, 28)
plt.imshow(image_data, cmap='gray_r')
plt.title('28x28 Image Data')
plt.colorbar()
plt.show()

