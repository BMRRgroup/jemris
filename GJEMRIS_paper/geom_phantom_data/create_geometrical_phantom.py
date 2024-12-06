import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

def create_sample_file(filename, Nx=30, Ny=20, N_spins=5):
    """
    Creates an HDF5 file containing a rectangular sample with various tissue types.

    Args:
        filename (str): Path to save the HDF5 file.
        Nx (int): Number of samples along the x-dimension.
        Ny (int): Number of samples along the y-dimension.
        N_spins (int): Number of spins along the z-dimension.
    """
    # Ensure the file does not already exist
    if os.path.exists(filename):
        raise FileExistsError(f"File {filename} already exists. To avoid overwriting, remove it first.")

    # Initialize datasets
    M0 = np.zeros((Nx, Ny, N_spins))
    R1, R2, R2s, DB = [np.zeros_like(M0) for _ in range(4)]

    # Create masks
    x = np.arange(Ny)
    y = np.arange(Nx)
    xx, yy = np.meshgrid(x, y)
    
    muscle = (xx >= 3) & (xx <= 17)
    bone = ((yy - 6) ** 2 / 25 + (xx - 10) ** 2 / 25) <= 1
    menisci = ((yy - 15) ** 2 / 4 + (xx - 10) ** 2 / 25) <= 1
    ligaments = ((xx == 6) | (xx == 5)) & (np.abs(yy - 24) < 5)
    blood = ((yy - 26) ** 2 / 4 + (xx - 14) ** 2 / 4) <= 1
    water = (((yy - 24) ** 2 + (xx - 10) ** 2) < 1) | (((yy == 20) | (yy == 21)) & ((xx == 14) | (xx == 15))) | (((yy == 20) | (yy == 21)) & (xx == 10))

    # Assign properties
    M0[muscle, :] = 10
    M0[bone, :] = 8
    M0[menisci, :] = 9
    M0[ligaments, :] = 7
    M0[blood, :] = 5
    M0[water, :] = 7

    R1[muscle, :] = 1 / 2000
    R1[bone, :] = 1 / 600
    R1[menisci, :] = R1[ligaments, :] = 1 / 900
    R1[blood, :] = 1 / 600
    R1[water, :] = 1 / 900

    R2[muscle, :] = 1 / 40
    R2[bone, :] = R2[blood, :] = 1 / 25
    R2[menisci, :] = R2[ligaments, :] = R2[water, :] = 1 / 50

    R2s[muscle, :] = 1 / 40
    R2s[bone, :] = R2s[blood, :] = 1 / 25
    R2s[menisci, :] = R2s[ligaments, :] = R2s[water, :] = 1 / 50

    # Combine datasets
    data_array = np.stack((M0, R1, R2, R2s, DB))
    print('Data array shape before transpose:', data_array.shape)

    # Visualize all maps
    visualize_maps(data_array, slice_=3)

    # Transpose data array for compatibility
    data_array = np.transpose(data_array, (3, 1, 2, 0))
    print('Data array shape after transpose:', data_array.shape)

    # Define resolution and offset
    data_resolution = np.array([[1, 1, 1 / 5]])
    data_offset = np.array([[0, 0, 0]])

    # Save to HDF5 file
    with h5py.File(filename, 'w') as f:
        f["sample/data"] = data_array
        f["sample/offset"] = data_offset
        f["sample/resolution"] = data_resolution
        print(f"File saved: {filename}")


def visualize_maps(data_array, slice_):
    """
    Visualizes all maps (M0, R1, R2, R2s, DB) for a given slice using appropriate colormaps.

    Args:
        data_array (ndarray): Combined data array containing all maps.
        slice_ (int): Index of the slice to visualize.
    """
    map_names = ["M0", "R1", "R2", "R2s", "DB"]
    colormaps = ["viridis", "plasma", "cividis", "magma", "cool"]
    num_maps = data_array.shape[0]

    fig, axs = plt.subplots(1, num_maps, figsize=(15, 5))
    for i in range(num_maps):
        im = axs[i].imshow(data_array[i, :, :, slice_], cmap=colormaps[i])
        axs[i].set_title(map_names[i])
        axs[i].axis("off")
        fig.colorbar(im, ax=axs[i], orientation="vertical")
    plt.tight_layout()
    plt.show()

# Example usage
# create_sample_file("ellips_SS_5.h5")
