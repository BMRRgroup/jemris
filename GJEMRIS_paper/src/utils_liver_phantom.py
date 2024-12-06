import matplotlib.pyplot as plt
import glob
import numpy as np
import h5py
from scipy.io import loadmat
from liver_phantom_data.DUKE_masks.phantom_parameters import parameters  # Import parameter dictionary


def load_masks(mask_path):
    """
    Load Duke phantom masks and their indices.

    Parameters:
        mask_path (str): Path to the mask files.

    Returns:
        dict: Masks with tissue types as keys.
        dict: Indices for each tissue type.
    """
    paths = glob.glob(mask_path)
    masks = {}
    indices = {}
    for i, path in enumerate(paths):
        name = path.split('mask_')[1].split('.mat')[0]
        masks[name] = loadmat(path)[f'mask_{name}']
        indices[name] = i + 1
    return masks, indices


def generate_mask_array(masks, indices):
    """
    Combine masks into a single array based on indices.

    Parameters:
        masks (dict): Tissue masks.
        indices (dict): Indices for each tissue type.

    Returns:
        numpy.ndarray: Combined mask array.
    """
    arr = np.zeros_like(next(iter(masks.values())))
    for key, mask in masks.items():
        arr += indices[key] * mask
    return arr


def create_parameter_maps(masks, indices, tissue_types, slice_ind):
    """
    Generate parameter maps based on tissue types and parameters.

    Parameters:
        masks (dict): Tissue masks.
        indices (dict): Indices for each tissue type.
        tissue_types (list): List of tissue types.
        slice_ind (int): Slice index for 3D masks.

    Returns:
        numpy.ndarray: Generated parameter maps.
    """
    param_types = list(parameters.keys())
    maps = np.zeros((masks["Liver"].shape[0], masks["Liver"].shape[1], len(param_types)))

    for i, param_type in enumerate(param_types):
        for key in tissue_types:
            if key != "Cerebrospinal_fluid":
                maps[:, :, i] += parameters[param_type][key] * masks[key][:, :, slice_ind]
            else:
                mask_fluid_new = np.zeros(masks["Liver"][:, :, slice_ind].shape)
                mask_fluid_new[masks["Liver"][:, :, slice_ind] == 6.0] = 1.0
                maps[:, :, i] += parameters[param_type][key] * mask_fluid_new
                
    maps[:, :, 4] += maps[:, :, 6]
    maps[:, :, 0] *= ((100 - maps[:, :, 5]) / 100)
    return maps


def process_geometry(maps, gamBo, N_spins_dim, res_factor, crop_start=70, crop_end=278, center_offset=67):
    """
    Crop, pad, and center the parameter maps.

    Parameters:
        maps (numpy.ndarray): Parameter maps.
        crop_start (int): Starting index for cropping.
        crop_end (int): Ending index for cropping.
        center_offset (int): Offset for centering the maps.

    Returns:
        numpy.ndarray: Processed parameter maps.
    """
    cropped_maps = maps[crop_start:crop_end, :, :]
    cropped_maps_padded = np.zeros((cropped_maps.shape[0], cropped_maps.shape[0], cropped_maps.shape[2]))
    s = cropped_maps_padded.shape
    cropped_maps_padded[:, (s[1] // 2 - center_offset):(s[1] // 2 + cropped_maps.shape[1] - center_offset), :] = cropped_maps
    maps = np.flip(np.rot90(cropped_maps_padded, axes=(0, 1)), 1)[1:-7, 1:-7]

    Nx = maps.shape[0] * N_spins_dim // res_factor
    Ny = Nx

    # Interpolate maps
    def interpolate_map(x, y, imap):
        return imap[x // (N_spins_dim // res_factor), y // (N_spins_dim // res_factor)]

    x = np.arange(Nx)
    y = np.arange(Ny)
    maps_interp = interpolate_map(*np.meshgrid(x, y, indexing='ij'), maps)

    return maps_interp

def generate_fat_phantoms(water_maps, parameters, fat_model_H, data_resolution, filename):
    """
    Generate and return stacked fat phantom maps while saving individual maps to HDF5.

    Parameters:
        masks (dict): Tissue masks.
        parameters (dict): Phantom parameter dictionary.
        fat_model_H (dict): Fat model parameters containing `relAmps` and `freqs_ppm`.
        gamBo (float): Gyromagnetic ratio times field strength (MHz).
        slice_ind (int): Slice index for the phantom.
        N_spins_dim (int): Number of spins per voxel dimension.
        res_factor (int): Resolution factor for interpolation.
        data_resolution (numpy.ndarray): Resolution information for the HDF5 file.

    Returns:
        numpy.ndarray: Stacked maps of all fat phantoms (4D array).
    """
    relAmps = fat_model_H["relAmps"]
    freqs_ppm = fat_model_H["freqs_ppm"]
    param_types = ["rho", "T1", "T2", "T2star", "chem"]

    # Preallocate space for stacked maps
    num_fat_components = len(relAmps)
    stacked_maps = []

    for fat_idx, (relAmp, freq_ppm) in enumerate(zip(relAmps, freqs_ppm)):
        filename_fat = f'{filename}_F{fat_idx + 1}'
        print(f"Generating fat phantom {filename_fat}...")

        # Initialize fat maps
        maps = np.zeros((water_maps.shape[0], water_maps.shape[1], len(param_types)))
        
        # Populate maps with fat parameters
        maps[..., 0] = water_maps[..., 0] * water_maps[..., 5] / (100 - water_maps[..., 5]) * relAmp
        mask = maps[..., 0] != 0  # Identify non-zero regions
        maps[mask, 1] = parameters["T1"]["Fat"]
        maps[mask, 2] = parameters["T2"]["Fat"]
        maps[mask, 3] = parameters["T2star"]["Fat"]
        maps[..., 4] = (water_maps[..., 6] / 3.4) * freq_ppm

        # Add processed map to stacked maps
        stacked_maps.append(maps)

        # Save individual phantom to HDF5
        # save_hdf5(filename_fat, maps[None, ...], data_resolution, data_offset=np.array([[0, 0, 0]]))
        print(f"Saved fat phantom: {filename_fat}.h5")

    # Stack all maps into a single 4D array
    stacked_maps = np.stack(stacked_maps, axis=0)
    return stacked_maps


def save_hdf5(filename, sample_jemris, data_resolution, data_offset):
    """
    Save the generated phantom data to an HDF5 file.

    Parameters:
        filename (str): File name (without extension) for the HDF5 file.
        sample_jemris (numpy.ndarray): Generated phantom data array.
        data_resolution (numpy.ndarray): Resolution metadata for the HDF5 file.
    """
    sample_jemris[1:4, :, :, 0] = 1.0 / sample_jemris[1:4, :, :, 0]  # Invert R1, R2, R2*

    with h5py.File(f'{filename}.h5', 'w-') as h5_file:
        h5_file["sample/data"] = sample_jemris
        h5_file["sample/offset"] = data_offset
        h5_file["sample/resolution"] = data_resolution
        print(f'Saved HDF5 file: {filename}.h5')


def plot_maps(maps, parameter_names, figname=None, title_prefix=""):
    """
    Plot parameter maps with colorbars and save the figure if required.

    Parameters:
        maps (numpy.ndarray): Parameter maps (3D array with shape [Nx, Ny, num_params]).
        parameter_names (list): Names of the parameters in the third dimension of `maps`.
        figname (str, optional): File name to save the figure (without extension).
        title_prefix (str, optional): Prefix to add to the titles of the plots.
    """
    num_params = len(parameter_names)
    cols = 3
    rows = (num_params + cols - 1) // cols  # Calculate rows needed for the grid

    fig, axs = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axs = axs.flatten()  # Flatten to handle single iterator
    colormaps = {
        "rho": "cividis",
        "T1": "inferno",
        "T2": "viridis",
        "T2star": "plasma",
        "Chi": "magma",
        "PDFF": "coolwarm",
        "chem": "bone"
    }

    for i in range(num_params):
        ax = axs[i]
        param_name = parameter_names[i]
        cmap = colormaps.get(param_name, "viridis")
        im = ax.imshow(maps[:, :, i], cmap=cmap)
        ax.set_title(f"{title_prefix}{param_name} map")
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im, ax=ax, format="%.2f")

    # Turn off any unused subplots
    for j in range(num_params, len(axs)):
        axs[j].axis("off")

    plt.tight_layout()
    if figname:
        plt.savefig(f"{figname}.png")
    plt.show()

def plot_pdff(M0, water_map=None, fat_map=None, figname=""):
    """
    Plot Proton Density Fat Fraction (PDFF) and optionally water and fat maps.

    Parameters:
        M0 (numpy.ndarray): PDFF map.
        water_map (numpy.ndarray): Optional water map.
        fat_map (numpy.ndarray): Optional fat map.
        figname (str): Name for saving the figure (without extension).
    """
    fig, axs = plt.subplots(3, 2, figsize=(8, 12))
    maps = {
        "PDFF": M0,
        "Water PD": water_map,
        "Fat PD": fat_map,
    }
    titles = [
        "PDFF, %", "Water PD map, A.U.", "Fat PD map, A.U."
    ]
    colormaps = {"PDFF": "inferno", "Water PD": "viridis", "Fat PD": "magma"}

    for i, (key, ax) in enumerate(axs.flat):
        if i < len(maps) and maps[key] is not None:
            im = ax.imshow(maps[key], cmap=colormaps[key])
            ax.set_title(titles[i])
            add_colorbar(im, fmt="%.f")
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    if figname:
        plt.savefig(f"{figname}.png")
    plt.show()


def add_colorbar(im, fmt="%.f"):
    """
    Add a colorbar to a plot.

    Parameters:
        im: The image plot to add a colorbar for.
        fmt (str): Format for the colorbar labels.
    """
    plt.colorbar(im, ax=im.axes, format=fmt)


