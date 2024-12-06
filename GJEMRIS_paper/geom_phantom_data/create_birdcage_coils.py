import numpy as np
import matplotlib.pyplot as plt
import h5py
import sigpy.mri as mr
import os

def simulate_birdcage_coil_array(N_channels=4, dimensions=(30, 30, 30), radius=1, \
    base_filename="Sag_coil", save_dir="./output_directory/", xml_filename="CoilArray.xml"):
    """
    Simulates a birdcage coil array using SigPy, creates sensitivity maps,
    saves them as .h5 files, and generates an XML configuration file.

    Args:
        N_channels (int): Number of channels in the birdcage coil array.
        dimensions (tuple): Shape parameters (dim_x, dim_y, dim_z).
        radius (float): Radius of the coil for the SigPy function.
        base_filename (str): Base filename for .h5 files.
        save_dir (str): Directory to save .h5 files and the XML file.
        xml_filename (str): Name of the generated XML file.
    """
    # Unpack dimensions
    dim_x, dim_y, dim_z = dimensions

    # Create sensitivity maps
    mps = mr.birdcage_maps(shape=(N_channels, dim_x, dim_y, dim_z), r=radius, nzz=N_channels)

    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Visualize the central x-slices
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    for i in range(2):
        for j in range(2):
            channel_idx = 2 * i + j
            axs[i, j].imshow(np.abs(mps[channel_idx, :, :, dim_x // 2]), cmap="viridis")
            axs[i, j].set_title(f"Channel {channel_idx + 1}")
            axs[i, j].axis("off")
    plt.tight_layout()
    plt.show()

    # Save .h5 files and generate XML content
    xml_content = "<CoilArray>\n"
    for ch in range(N_channels):
        file_path = os.path.join(save_dir, f"{base_filename}_{ch + 1}.h5")
        with h5py.File(file_path, 'w') as h5_file:
            h5_file["maps/phase"] = np.angle(mps[ch])
            h5_file["maps/magnitude"] = np.abs(mps[ch])
        print(f"Saved: {file_path}")
        # Add entry to XML
        xml_content += f'    <EXTERNALCOIL Dim="3" Extent="{dim_x}" Filename="{base_filename}_{ch + 1}.h5" Name="C{ch + 1}" Points="{dim_x}"/>\n'
    xml_content += "</CoilArray>\n"

    # Save XML file
    xml_path = os.path.join(save_dir, xml_filename)
    with open(xml_path, 'w') as xml_file:
        xml_file.write(xml_content)
    print(f"XML file saved: {xml_path}")

# usage example
# simulate_birdcage_coil_array(N_channels=4, dimensions=(30, 30, 30), radius=1, base_filename="Sag_coil", save_dir="./output_directory/", xml_filename="CoilArray.xml")
