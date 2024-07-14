import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from torchvision.transforms import Compose, Normalize

def draw_graph_3d(nodes, edges, ax):
    xs = nodes[:,0]
    ys = nodes[:,1]
    zs = nodes[:,2]
    ax.scatter(xs, ys, zs)

    # Add all edges
    for edge in edges:
        ax.plot([xs[edge[0]],xs[edge[1]]],[ys[edge[0]],ys[edge[1]]], [zs[edge[0]],zs[edge[1]]], color="green")

def create_sample_visual_3d(samples, number_samples=10):

    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, axs = plt.subplots(number_samples, 3, figsize=(1000 * px, number_samples * 300 * px), subplot_kw={'projection':'3d'})

    for i in range(number_samples):
        if i >= len(samples["segs"]):
            break
        try:
            verts, faces, normals, values = measure.marching_cubes(samples["segs"][i].squeeze().cpu().numpy(), 0)

            mesh = Poly3DCollection(verts[faces])
            mesh.set_edgecolor('k')
            axs[i, 0].add_collection3d(mesh)

            axs[i, 0].set_xlim(0, 64)
            axs[i, 0].set_ylim(0, 64)
            axs[i, 0].set_zlim(0, 64)

            plt.sca(axs[i, 1])
            draw_graph_3d(samples["nodes"][i].cpu().numpy(), samples["edges"][i].cpu().numpy(), axs[i, 1])
            plt.sca(axs[i, 2])
            draw_graph_3d(samples['pred_nodes'][i], samples['pred_rels'][i], axs[i, 2])

            axs[i, 1].set_xlim(0, 1)
            axs[i, 1].set_ylim(0, 1)
            axs[i, 1].set_zlim(0, 1)
            axs[i, 2].set_xlim(0, 1)
            axs[i, 2].set_ylim(0, 1)
            axs[i, 2].set_zlim(0, 1)
        except:
            print("visualiazion error occured, skipping sample...")

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return np.transpose(data, (2, 0, 1))

def draw_graph_2d(nodes, edges, ax):
    xs = nodes[:, 0]
    ys = nodes[:, 1]
    ax.scatter(ys, xs, c="red")

    # Add all edges
    for edge in edges:
        ax.plot([ys[edge[0]], ys[edge[1]]], [xs[edge[0]], xs[edge[1]]], color="red")


def create_sample_visual_2d(samples, number_samples=10):
    px = 1 / plt.rcParams['figure.dpi']  # pixel in inches
    fig, axs = plt.subplots(number_samples, 2, figsize=(1000 * px, number_samples * 300 * px))

    if len(samples["segs"].shape) >= 5:
        indices = np.round(np.array(samples["z_pos"]) * (samples["segs"].shape[-1] - 1)).astype(int)

    for i in range(number_samples):
        if i >= len(samples["segs"]):
            break
        if len(samples["segs"].shape) >= 5:
            print(i)
            print(samples["segs"][i].shape)
            sample = samples["segs"][i,0,:,:,indices[i]]
            print("here", sample.shape)
        else:
            sample = samples["images"][i,0]
        axs[i, 0].imshow(sample.clone().cpu().detach())
        axs[i, 1].imshow(sample.clone().cpu().detach())

        plt.sca(axs[i, 0])
        draw_graph_2d(samples["nodes"][i].clone().cpu().detach() * sample.shape[0], samples["edges"][i].clone().cpu().detach(), axs[i, 0])
        plt.sca(axs[i, 1])
        draw_graph_2d(samples["pred_nodes"][i] * sample.shape[0], samples["pred_rels"][i], axs[i, 1])

    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return np.transpose(data, (2, 0, 1))


inv_norm = Compose([
    Normalize(
        mean=[0., 0., 0.],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.225]
    ),
    Normalize(
        mean=[-0.485, -0.456, -0.406],
        std=[1., 1., 1.]
    ),
])



