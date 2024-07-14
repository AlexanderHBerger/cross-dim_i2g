import os
from skimage import io
import numpy as np
from tqdm import tqdm
from argparse import ArgumentParser
import imageio
from medpy.io import save

parser = ArgumentParser()
parser.add_argument('--data_dir',
                    default=None,
                    required=True,
                    help='Path to data directory')

def main(args):
    # Set the path for the source images directory
    src_dir = args.data_dir

    # Set the path for the output directory
    train_path = f"{src_dir}/train_data/"
    if not os.path.isdir(train_path):
        os.makedirs(train_path)
        os.makedirs(train_path + '/raw')
    else:
        raise Exception("Train folder is non-empty")
    print('Preparing Train Data')

    # Set the patch size and padding
    patch_size = [50, 50, 50]
    pad = [2,2,2]

    image_id = 0

    # Load 3D tiff to numpy array
    img_path = os.path.join(src_dir, "large_patch", f"BL6J-no1_iso3um_stitched-2.tif")
    image = np.array(io.imread(img_path))

    # Move z-axis from position 0 to position 2
    image = np.moveaxis(image, 0, 2)

    # Apply intensity cutoff filter
    threshold = 25000
    image[image > threshold] = threshold
    image = image / threshold

    # Get the dimensions of the source and target image and padding
    p_h, p_w, p_d = patch_size
    pad_h, pad_w, pad_d = pad

    p_h = p_h - 2*pad_h
    p_w = p_w - 2*pad_w
    p_d = p_d - 2*pad_d

    h, w, d = image.shape
    patch_num1 = h // 35
    patch_num2 = patch_num1 // 4
    x_ = np.int32(np.linspace(5, h-5-p_h, patch_num1))
    y_ = np.int32(np.linspace(5, w-5-p_w, patch_num1))
    z_ = np.int32(np.linspace(5, d-5-p_d, patch_num2))

    grid = np.meshgrid(x_, y_, z_, indexing='ij')

    for start in tqdm(list(np.array(grid).reshape(3, -1).T)):
        start = np.array((start[0], start[1], start[2]))

        patch = np.pad(image[start[0]:start[0]+p_h, start[1]:start[1] +
                    p_w, start[2]:start[2] + p_d], ((pad_h, pad_h), (pad_w, pad_w), (pad_d, pad_d)))
        
        
        if patch.sum() > 6000:
            save(patch, train_path+'raw/sample_'+str(image_id).zfill(6)+'_data.nii.gz')
            image_id = image_id+1

    # Print a message indicating that the script has finished running
    print("Unlabeled dataset generation complete.")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
