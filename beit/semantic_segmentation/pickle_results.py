import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

test_dir = "/home/sandej17/building_segmentation/datasets/aerial_512/ann_dir/val/"

pick = open ("result.pkl", "rb")
unpicked = pickle.load(pick)

filepaths = [test_dir + filename for filename in sorted(os.listdir(test_dir))]

one_pick = unpicked[35].reshape(512, 512, 1)
f, ax = plt.subplots(1, 2)
ax[0].imshow(one_pick)
ax[1].imshow(mpimg.imread(filepaths[0]))
plt.savefig("pick.png")