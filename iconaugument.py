import Augmentor

#path of image
root = './single'

import os
import glob

#remove the former result
files = glob.glob(os.path.join(root, 'output/*'))
for f in files:
    os.remove(f)


p = Augmentor.Pipeline(root)

p.random_distortion(probability=0.1, grid_width=4, grid_height=4, magnitude=1)
p.zoom(probability=0.9, min_factor=0.5, max_factor=1.2)
p.crop_random(probability=0.6, percentage_area=0.8)

p.resize(probability=0.1, width=30, height=30)      # make image blur
p.resize(probability=0.2, width=60, height=60)
p.resize(probability=1.0, width=100, height=100)    # uniform the size at the end

# total amount of image you want to generate
p.sample(100000)
p.process()