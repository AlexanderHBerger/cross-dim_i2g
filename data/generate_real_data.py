from tqdm import tqdm
from medpy.io import load, save
import pyvista
import numpy as np
import os
import re
from patch_generator import PatchGraphGenerator
import scipy

patch_size = [50,50,50]
pad = [2,2,2]

def save_input(save_path, idx, patch, patch_seg, patch_coord, patch_edge):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    save(patch, save_path+'raw/sample_'+str(idx).zfill(6)+'_data.nii.gz')
    save(patch_seg, save_path+'seg/sample_'+str(idx).zfill(6)+'_seg.nii.gz')
    
    patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    # print(patch_edge.shape)
    mesh.lines = patch_edge.flatten()
    mesh.save(save_path+'vtp/sample_'+str(idx).zfill(6)+'_graph.vtp')


def patch_extract(save_path, image, seg,  gen, device=None):
    """[summary]

    Args:
        image ([type]): [description]
        coordinates ([type]): [description]
        lines ([type]): [description]
        patch_size (tuple, optional): [description]. Defaults to (64,64,64).
        num_patch (int, optional): [description]. Defaults to 2.

    Returns:
        [type]: [description]
    """
    global image_id
    # TODO: edge on the boundary of patch not included, edge which passes through the volume not included
    p_h, p_w, p_d = patch_size
    pad_h, pad_w, pad_d = pad

    p_h = p_h -2*pad_h
    p_w = p_w -2*pad_w
    p_d = p_d -2*pad_d
    
    h, w, d= image.shape
    x_ = np.int32(np.linspace(5, h-5-p_h, 20))
    y_ = np.int32(np.linspace(5, w-5-p_w, 20))
    z_ = np.int32(np.linspace(2, 48, 1))
    
    ind = np.meshgrid(x_, y_, z_, indexing='ij')
    # Center Crop based on foreground

    for i, start in enumerate(list(np.array(ind).reshape(3,-1).T)):
        # print(image.shape, seg.shape)
        end = start + np.array(patch_size) - 1 - 2 * np.array(pad)
        patch = np.pad(image[start[0]:start[0]+p_h, start[1]:start[1]+p_w, start[2]:start[2]+p_d], ((pad_h,pad_h),(pad_w,pad_w),(pad_d,pad_d)))
        patch_seg = np.pad(seg[start[0]:start[0]+p_h, start[1]:start[1]+p_w, start[2]:start[2]+p_d], ((pad_h,pad_h),(pad_w,pad_w),(pad_d,pad_d)))

        # Skip if bad SNR or too much foreground
        fg_pixels = patch[patch_seg > 0]
        bg_pixels = patch[patch_seg == 0]

        avg_intensity_fg = np.mean(fg_pixels)
        avg_intensity_bg = np.mean(bg_pixels)
        std_intensity_bg = np.std(bg_pixels)

        snr = (avg_intensity_fg - avg_intensity_bg) / std_intensity_bg
        fgr = fg_pixels.size / patch.size

        if snr < 1.2 or fgr > 0.5:
            print(f"skipping: {image_id}")
            continue

        # collect all the nodes
        bounds = np.array([[start[0], end[0]], [start[1], end[1]], [start[2], end[2]]])
        _subG = gen.create_patch_graph(bounds)
        nodes, edges = gen.get_last_patch()
        if patch_seg.sum() > 10 and len(nodes) >= 3:
            nodes = np.array(nodes) - [start[0], start[1], start[2]] + [pad_h, pad_w, pad_d]
            save_input(save_path, image_id, patch, patch_seg, nodes, np.array(edges))
            image_id = image_id + 1

if __name__ == "__main__":
    root_dir = "/media/data/alex_johannes/data/real_vessels/graphs_batch1_fh"

    raw_regex = re.compile(r'.*C00.*\.nii.*$')
    seg_regex = re.compile(r'.*label.*fh.*\.nii.*')
    nodes_regex = re.compile(r'.*nodes.*\.csv')
    edges_regex = re.compile(r'.*edges.*\.csv')
    centerline_regex = re.compile(r'.*\.vvg\.gz')

    raw_files = []
    seg_files = []
    nodes_files = []
    edges_files = []
    centerline_files = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if raw_regex.match(file):
                raw_files.append(os.path.join(root, file))
            elif seg_regex.match(file):
                seg_files.append(os.path.join(root, file))
            elif nodes_regex.match(file):
                nodes_files.append(os.path.join(root, file))
            elif edges_regex.match(file):
                edges_files.append(os.path.join(root, file))
            elif centerline_regex.match(file):
                centerline_files.append(os.path.join(root, file))

    image_id = 1
    train_path = '/media/data/alex_johannes/data/real_vessels/train_data/'
    if not os.path.isdir(train_path):
        os.makedirs(train_path)
        os.makedirs(train_path+'/seg')
        os.makedirs(train_path+'/vtp')
        os.makedirs(train_path+'/raw')
    else:
        raise Exception("Train folder is non-empty")

    print('Preparing Train Data')
    for idx, seg_file in tqdm(enumerate(seg_files[:21])):
        image_data, _ = load(raw_files[idx])
        image_data = np.int32(image_data)
        seg_data, _ = load(seg_files[idx])
        seg_data = np.int8(seg_data)

        threshold = scipy.stats.median_abs_deviation(image_data.flatten(), scale = "normal") * 4 \
                    + np.median(image_data.flatten())
        image_data[image_data > threshold] = threshold
        image_data = image_data / threshold

        gen = PatchGraphGenerator(nodes_files[idx], edges_files[idx], centerline_files[idx], patch_mode='centerline')

        patch_extract(train_path, image_data, seg_data, gen)

    image_id = 1
    test_path = '/media/data/alex_johannes/data/real_vessels/test_data/'
    if not os.path.isdir(test_path):
        os.makedirs(test_path)
        os.makedirs(test_path + '/seg')
        os.makedirs(test_path + '/vtp')
        os.makedirs(test_path + '/raw')
    else:
        raise Exception("Test folder is non-empty")

    print('Preparing Test Data')
    for idx, seg_file in tqdm(enumerate(seg_files[21:])):
        image_data, _ = load(raw_files[idx])
        image_data = np.int32(image_data)
        seg_data, _ = load(seg_files[idx])
        seg_data = np.int8(seg_data)

        threshold = scipy.stats.median_abs_deviation(image_data.flatten(), scale="normal") * 4 \
                    + np.median(image_data.flatten())
        image_data[image_data > threshold] = threshold
        image_data = image_data / threshold

        gen = PatchGraphGenerator(nodes_files[idx], edges_files[idx], centerline_files[idx], patch_mode='centerline')

        patch_extract(test_path, image_data, seg_data, gen)

