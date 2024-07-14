import random

import torch
import numpy as np
import pyvista
from scipy.ndimage.morphology import grey_dilation
from scipy import ndimage
import torch.nn.functional as F
from math import radians, sin, cos
from monai.transforms import Rotate
import torchio as tio

def upsample_edges(relation_pred, edge_labels, sample_ratio, acceptance_interval):
    target_pos_edges = relation_pred[edge_labels == 1]
    target_neg_edges = relation_pred[edge_labels == 0]

    if target_neg_edges.shape[0] == 0 or target_pos_edges.shape[0] == 0:
        return relation_pred, edge_labels

    actual_ratio = target_pos_edges.shape[0] / target_neg_edges.shape[0]

    if actual_ratio < sample_ratio - acceptance_interval:
        target_num = target_neg_edges.shape[0] * sample_ratio
        num_new_edges = int(target_num - target_pos_edges.shape[0])

        new_edges = target_pos_edges.repeat(int(num_new_edges / target_pos_edges.shape[0]), 1)
        new_edges = torch.cat((new_edges, target_pos_edges[:int(num_new_edges - new_edges.shape[0])]))

        new_labels = torch.ones(num_new_edges, dtype=torch.long, device=relation_pred.device)
    elif sample_ratio + acceptance_interval < actual_ratio:
        target_num = target_pos_edges.shape[0] * (1 / sample_ratio)
        num_new_edges = int(target_num - target_neg_edges.shape[0])

        new_edges = target_neg_edges.repeat(int(num_new_edges / target_neg_edges.shape[0]), 1)
        new_edges = torch.cat((new_edges, target_neg_edges[:int(num_new_edges - new_edges.shape[0])]))

        new_labels = torch.zeros(num_new_edges, dtype=torch.long, device=relation_pred.device)
    else:
        return relation_pred, edge_labels

    return torch.cat((relation_pred, new_edges)), torch.cat((edge_labels, new_labels))

def downsample_edges(relation_pred, edge_labels, sample_ratio, acceptance_interval):
    """
    Downsamples the edges based on the given sample ratio and acceptance interval.

    Args:
        relation_pred (torch.Tensor): The predicted relation values.
        edge_labels (torch.Tensor): The labels for the edges.
        sample_ratio (float): The desired ratio of positive to negative edges.
        acceptance_interval (float): The acceptable deviation from the sample ratio.

    Returns:
        tuple: A tuple containing the downsampled relation predictions and edge labels.
    """
    target_pos_edges = relation_pred[edge_labels == 1]
    target_neg_edges = relation_pred[edge_labels == 0]

    actual_ratio = target_pos_edges.shape[0] / target_neg_edges.shape[0]

    if actual_ratio < sample_ratio - acceptance_interval:
        # In this case, we have too many negative edges, so we need to remove some
        target_num = int(target_pos_edges.shape[0] * (1 / sample_ratio))

        target_neg_edges = target_neg_edges[:target_num]
    elif sample_ratio + acceptance_interval < actual_ratio:
        # In this case, we have too many positive edges, so we need to remove some
        target_num = int(target_neg_edges.shape[0] * sample_ratio)

        target_pos_edges = target_pos_edges[:target_num]
    else:
        return relation_pred, edge_labels
    
    return (
        torch.cat((target_pos_edges, target_neg_edges)), 
        torch.cat(
            (torch.ones(target_pos_edges.shape[0], dtype=torch.long, device=relation_pred.device), 
             torch.zeros(target_neg_edges.shape[0], dtype=torch.long, device=relation_pred.device)))
        )


def image_graph_collate(batch, pre2d=False, gaussian_augment=False):
    z_pos = 0
    images = torch.cat([torch.stack(item[0]) for item in batch]).contiguous()
    segs = torch.cat([torch.stack(item[1]) for item in batch]).contiguous()
    if pre2d:
        z_pos = random.randint(0, 3) / 3
        points = [F.pad(item_, (0, 1), "constant", z_pos) for item in batch for item_ in item[2]]
    elif gaussian_augment:
        z_pos = [item_ for item in batch for item_ in item[4]]
        points = [item_ for item in batch for item_ in item[2]]
    else:
        points = [item_ for item in batch for item_ in item[2]]
    edges = [item_ for item in batch for item_ in item[3]]
    domains = torch.tensor([item[5] for item in batch]).flatten()
    return [images, segs, points, edges, z_pos, domains]

def rotate_coordinates(points, alpha, beta, gamma):
    alpha_rad = radians(alpha)
    beta_rad = radians(beta)
    gamma_rad = radians(gamma)

    yaw_matrix = torch.Tensor([
        [cos(-alpha_rad), -sin(-alpha_rad), 0],
        [sin(-alpha_rad), cos(-alpha_rad), 0],
        [0, 0, 1]
    ])

    pitch_matrix = torch.Tensor([
        [cos(-beta_rad), 0, sin(-beta_rad)],
        [0, 1, 0],
        [-sin(-beta_rad), 0, cos(-beta_rad)]
    ])

    roll_matrix = torch.Tensor([
        [1, 0, 0],
        [0, cos(-gamma_rad), -sin(-gamma_rad)],
        [0, sin(-gamma_rad), cos(-gamma_rad)]
    ])

    rotation = torch.matmul(torch.matmul(pitch_matrix, roll_matrix), yaw_matrix)

    points = torch.matmul(points - 0.5, rotation) + 0.5

    return points

def rotate_image(image, alpha, beta, gamma):
    transform = tio.Affine(
        scales=(1, 1, 1),
        degrees=(gamma, beta, alpha),
        translation=(0, 0, 0),
        center="image",
        default_pad_value="mean",
    )

    rot_image = transform(image)

    return rot_image


def save_input(path, idx, patch, patch_coord, patch_edge):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    
    # vertices, faces, _, _ = marching_cubes_lewiner(patch)
    # vertices = vertices/np.array(patch.shape)
    # faces = np.concatenate((np.int32(3*np.ones((faces.shape[0],1))), faces), 1)
    
    # mesh = pyvista.PolyData(vertices)
    # mesh.faces = faces.flatten()
    # mesh.save(path+'_sample_'+str(idx).zfill(3)+'_segmentation.stl')
    
    patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    # print(patch_edge.shape)
    mesh.lines = patch_edge.flatten()
    mesh.save(path+'_sample_'+str(idx).zfill(3)+'_graph.vtp')
    
    
def save_output(path, idx, patch_coord, patch_edge):
    """[summary]

    Args:
        patch ([type]): [description]
        patch_coord ([type]): [description]
        patch_edge ([type]): [description]
    """
    patch_edge = np.concatenate((np.int32(2*np.ones((patch_edge.shape[0],1))), patch_edge), 1)
    mesh = pyvista.PolyData(patch_coord)
    if patch_edge.shape[0]>0:
        mesh.lines = patch_edge.flatten()
    mesh.save(path+'_sample_'+str(idx).zfill(3)+'_graph.vtp')


def patchify_voxel(volume, patch_size, pad):
    p_h, p_w, p_d = patch_size
    pad_h, pad_w, pad_d = pad

    p_h = p_h -2*pad_h
    p_w = p_w -2*pad_w
    p_d = p_d -2*pad_d
    
    
    v_h, v_w, v_d = volume.shape

    # Calculate the number of patch in ach axis
    n_w = np.ceil(1.0*(v_w-p_w)/p_w+1)
    n_h = np.ceil(1.0*(v_h-p_h)/p_h+1)
    n_d = np.ceil(1.0*(v_d-p_d)/p_d+1)

    n_w = int(n_w)
    n_h = int(n_h)
    n_d = int(n_d)

    pad_1 = (n_w - 1) * p_w + p_w - v_w
    pad_2 = (n_h - 1) * p_h + p_h - v_h
    pad_3 = (n_d - 1) * p_d + p_d - v_d

    volume = np.pad(volume, ((0, pad_1), (0, pad_2), (0, pad_3)), mode='reflect')
    
    h, w, d= volume.shape
    x_ = np.int32(np.linspace(0, h-p_h, n_w))
    y_ = np.int32(np.linspace(0, w-p_w, n_h))
    z_ = np.int32(np.linspace(0, d-p_d, n_d))
    
    ind = np.meshgrid(x_, y_, z_, indexing='ij')
    
    patch_list = []
    start_ind = []
    seq_ind = []
    for i, start in enumerate(list(np.array(ind).reshape(3,-1).T)):
        patch = np.pad(volume[start[0]:start[0]+p_h, start[1]:start[1]+p_w, start[2]:start[2]+p_d], ((pad_h,pad_h),(pad_w,pad_w),(pad_d,pad_d)))
        patch_list.append(patch)
        start_ind.append(start)
        seq_ind.append([i//(y_.shape[0]*z_.shape[0]), (i%(y_.shape[0]*z_.shape[0]))//z_.shape[0], (i%(y_.shape[0]*z_.shape[0]))%z_.shape[0]])
        
    return patch_list, start_ind, seq_ind, volume.shape


def unpatchify_graph(patch_graphs, start_ind, seq_ind, pad, imsize=[128,128,128]):
    """

    :param patches:
    :param step:
    :param imsize:
    :param scale_factor:
    :return:
    """
    patch_coords, patch_edges = patch_graphs['pred_nodes'], patch_graphs['pred_rels']
    occu_matrix = np.empty((8,)+imsize)  # 8 channel occu matrix
    pred_coords = []
    pred_rels = []
    num_nodes = 0
    struct = ndimage.generate_binary_structure(3, 2)
    for i, (patch_coord, patch_edge) in enumerate(zip(patch_coords, patch_edges)):
        patch_node_label = np.zeros(imsize)
        abs_patch_coord = np.array(start_ind[i]-pad) + patch_coord*64
        pred_coords.extend(abs_patch_coord)
        abs_patch_coord = np.int64(abs_patch_coord)
        ch_idx = np.sum(2**(np.array(range(3))[::-1])*(np.array(seq_ind[i])%2))
        # print(start_ind[i], seq_ind[i], np.array(seq_ind[i])%2, ch_idx)

        # local patch occupancy
        patch_node_label[abs_patch_coord[:,0],abs_patch_coord[:,1],abs_patch_coord[:,2]] = np.array(list(range(num_nodes,num_nodes+patch_coord.shape[0])))+1

        # dialate each node regions in isotropic way
        
        # occu_matrix[ch_idx, start_ind[i][0]-pad[0]:start_ind[i][0]-pad[0]+64, start_ind[i][1]-pad[1]:start_ind[i][1]-pad[1]+64, start_ind[i][2]-pad[2]:start_ind[i][2]-pad[2]+64] = 1
        for _ in range(8):
            inst_label = grey_dilation(patch_node_label, footprint=struct) #size=(3,3,3)) # structure=struct)
            inst_label[patch_node_label>0] = patch_node_label[patch_node_label>0]
            patch_node_label = inst_label
            occu_matrix[ch_idx, patch_node_label>0] = patch_node_label[patch_node_label>0]

        # occu_matrix[patch_node_label>0.0] = patch_node_label[patch_node_label>0.0]
    
        pred_rels.extend(patch_edge+num_nodes)
        num_nodes = num_nodes+patch_coord.shape[0]
    
    pred_graph = {'pred_nodes':pred_coords,'pred_rels':pred_rels}
    return occu_matrix, pred_graph

    
def Bresenham3D(p1, p2): 
    """
    Function to compute direct connection in voxel space
    """
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    ListOfPoints = [] 
    ListOfPoints.append((x1, y1, z1)) 
    dx = abs(x2 - x1) 
    dy = abs(y2 - y1) 
    dz = abs(z2 - z1) 
    if (x2 > x1): 
        xs = 1
    else: 
        xs = -1
    if (y2 > y1): 
        ys = 1
    else: 
        ys = -1
    if (z2 > z1): 
        zs = 1
    else: 
        zs = -1
  
    # Driving axis is X-axis" 
    if (dx >= dy and dx >= dz):         
        p1 = 2 * dy - dx 
        p2 = 2 * dz - dx 
        while (x1 != x2): 
            x1 += xs 
            if (p1 >= 0): 
                y1 += ys 
                p1 -= 2 * dx 
            if (p2 >= 0): 
                z1 += zs 
                p2 -= 2 * dx 
            p1 += 2 * dy 
            p2 += 2 * dz 
            ListOfPoints.append((x1, y1, z1)) 
  
    # Driving axis is Y-axis" 
    elif (dy >= dx and dy >= dz):        
        p1 = 2 * dx - dy 
        p2 = 2 * dz - dy 
        while (y1 != y2): 
            y1 += ys 
            if (p1 >= 0): 
                x1 += xs 
                p1 -= 2 * dy 
            if (p2 >= 0): 
                z1 += zs 
                p2 -= 2 * dy 
            p1 += 2 * dx 
            p2 += 2 * dz 
            ListOfPoints.append((x1, y1, z1)) 
  
    # Driving axis is Z-axis" 
    else:         
        p1 = 2 * dy - dz 
        p2 = 2 * dx - dz 
        while (z1 != z2): 
            z1 += zs 
            if (p1 >= 0): 
                y1 += ys 
                p1 -= 2 * dz 
            if (p2 >= 0): 
                x1 += xs 
                p2 -= 2 * dz 
            p1 += 2 * dy 
            p2 += 2 * dx 
            ListOfPoints.append((x1, y1, z1)) 
    return ListOfPoints