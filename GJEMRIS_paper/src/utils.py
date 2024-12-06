import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import h5py 
import os

font = 12 # 14
plt.rcParams['axes.linewidth'] = 3
plt.rcParams['font.size'] = font 
plt.rcParams['figure.figsize'] = (10,6)
plt.rcParams['ytick.labelsize'] = font
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['figure.titleweight'] = 'bold'
plt.rcParams['xtick.labelsize'] = font
plt.rcParams['grid.linewidth'] = 2
plt.rcParams['grid.linewidth'] = 2
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.dpi'] = 140


def compare_signals( file_gpu_dp, file_gpu_sp, file_mpi_dp,  folder='./speed_benchmark_data/', nt=2*64, \
                    n_channels=1, plot=True, save_plots=False, save_figname=''):
    """
    Function to load double and single precision GPU and double precision MPI-CPU signals from JEMRIS simulations, 
    and plot magnetization vectors.
    Arguments:
        file_gpu_dp, file_gpu_sp, file_mpi_dp : str
            full filenames of the simulated signals
            set to None if not included
        folder : str
            folder with the data
        nt : int
            number of timepoints to plot
        n_channels : number of coil channels simulated
        save_plots : bool
            save the figures in the 
        save_figname : str
            name of the figure to save
    """
    files = [file_gpu_dp, file_gpu_sp, file_mpi_dp]
    echo_offset = int(0*64)

    time_vectors = []
    magn_vectors = []
    for file in files:
        if file is not None:
            full_filepath = os.path.join(folder, file)
            datafile = h5py.File(full_filepath, 'r')
            time_ = np.array(datafile['signal/times'])[echo_offset:echo_offset+nt]
            magn_ = np.zeros((n_channels, nt, 3))
            for ch in range(n_channels):
                magn_[ch] = np.array(datafile['signal/channels/0'+str(ch)])[echo_offset:echo_offset+nt,:]
            time_vectors.append(time_)
            magn_vectors.append(magn_)
        else:
            time_vectors.append(None)
            magn_vectors.append(None)

    titles = [r'$M_x$', r'$M_y$', r'$M_z$']
    data_labels = ['MPI_DP', 'GPU_DP', 'GPU_SP']
    if plot:
        for ch in range(n_channels):
            fig, axs = plt.subplots(1, 3, figsize=(11, 4.5))
            plt.suptitle('Signals, channel {ch+1}', y=0.95)
            # set the spacing between subplots
            plt.subplots_adjust(hspace=0.1, wspace=0.05, top=0.82)
            for i in range(3):
                for data_ii in range(len(files)):
                    if time_vectors[data_ii] is not None:
                        axs[i].plot(time_vectors[data_ii], magn_vectors[data_ii][ch, :,  i], '--o', label=data_labels[data_ii])
                        axs[i].set_xlabel("Time, ms")
                        axs[i].set_title(titles[i], fontweight='bold')
                        axs[i].grid()
        #             axs[i].set_ylim([-0.1, 0.06])
                        if (i != 0):
                            axs[i].set_yticklabels([])
    #         plt.tight_layout()
            plt.legend()
            plt.show()
        
            if save_plots:
                plt.savefig(f'Signals_GJvsPJ_{save_figname}_ch{ch}.png')  
        
    return time_vectors, magn_vectors
            
            
def RMS(im):
    return np.sqrt(np.real(np.conj(im).T * im)/(im.shape[0]))

# def nrmse(im_corr, im_GT):
#     """
#     implementation similar to BART NRMSE
#     """
#     im1 = im_corr.reshape((-1, 1))
#     im2 = im_GT.reshape((-1, 1))
#     sc = np.conj(im2).T * im1
#     sc /= np.real(np.conj(im2).T * im2)
#     im2 *= sc
#     return RMS(im1 - im2) / RMS(im2)

def nrmse(image1, image2):
    """
    Compute the NRMSE between two complex images.

    Arguments:
        image1 : np.ndarray
            First complex image.
        image2 : np.ndarray
            Second complex image.

    Returns:
        float:
            The NRMSE value.
    """
    # Ensure the inputs are numpy arrays
    image1 = np.asarray(image1)
    image2 = np.asarray(image2)

    # Compute the RMSE
    mse = np.mean(np.abs(image1 - image2)**2)
    rmse = np.sqrt(mse)

    # Normalize by the maximum magnitude of the two images
    max_magnitude = max(np.max(np.abs(image1)), np.max(np.abs(image2)))
    nrmse = rmse / max_magnitude

    return nrmse


def error_GPUvsMPI(signal_gpu, signal_mpi_1, signal_mpi_2):
    """
    As simulations include statistical effects, there is a variability of the signal between launches.
    Therefore, the error between GPU and MPI signals is calculated as 
        NRMSE(gpu, mpi_1) - NRMSE(mpi_1,mpi_2)
    """
    return np.abs(nrmse(signal_gpu, signal_mpi_1) - nrmse(signal_mpi_1, signal_mpi_2))

def normalize(img):
    return img / (np.max(np.abs(img)))


def normalize_all(data_list, ref_data):
    max_val = np.max(np.abs(ref_data))
    return [data / max_val for data in data_list]
    
def compute_and_print_errors(gpu_data, mpi_data, mpi_data_2, label):
    rmse = error_GPUvsMPI(gpu_data, mpi_data, mpi_data_2)
    nrmse_gpu_mpi = nrmse(gpu_data, mpi_data)
    nrmse_mpi_mpi2 = nrmse(mpi_data_2, mpi_data)
    print(f"Relative {label}-space RMSE:\t{rmse:.10f}")
    print(f"NRMSE (GPU vs MPI): {nrmse_gpu_mpi:.6f},\tNRMSE (MPI vs MPI2): {nrmse_mpi_mpi2:.6f}")

    # Helper function to add colorbars to plots
def add_colorbar(im, fmt=None):
    cbar = plt.colorbar(im)
    if fmt:
        cbar.formatter.set_scientific(True)
        cbar.formatter.set_powerlimits((0, 0))
        cbar.update_ticks()
        
def simulation_visualization(magn, phase_order=[], Nkx=128, Nky=11, save=False,\
    name_add="", save_fold="", **params):
    """
    Visualize JEMRIS simulation results and compute k-space and image reconstructions.

    Arguments:
        magn : np.ndarray
            Magnetization vector (Mx, My, Mz) over time from JEMRIS output.
        phase_order : np.ndarray
            Array of size (N_timepoints, 4), describing phase encoding order.
        Nkx : int
            Number of frequency encoding (readout) points.
        Nky : int
            Number of phase encoding points.
        save : bool
            If True, saves plots to the specified folder.
        name_add : str
            Additional string for figure filenames.
        save_fold : str
            Folder path to save the plots.
        **params: Additional options:
            SEQ_NAME (str): Sequence name for titles.
            show_order (bool): If True, shows phase encoding order plot.
            show_motion (bool): If True, visualizes motion trajectory.
            traj (np.ndarray): Trajectory array of shape (,7).
            show_magn (bool): If True, plots magnetization evolution.
            echo_nr_show (list): List of echo indices to display.
            show_kspace (bool): If True, visualizes k-space data.
            kspace_scale (str): Scale for k-space plot ('abs', 'log').
            title_add (str): Additional title for plots.

    Returns:
        tuple:
            - magn_tr (np.ndarray): Complex k-space signal, magnitude normalized to [0,1].
            - recon_im (np.ndarray): Complex reconstructed image, magnitude normalized to [0,1].
    """
    # Phase encoding order visualization
    if params.get("show_order", False):
        SEQ_NAME = params.get("SEQ_NAME", "Unknown Sequence")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(phase_order[:, 0], phase_order[:, 2], s=5)
        ax.set_title(f'Phase Encoding Order\n{SEQ_NAME}')
        ax.set_xlabel('Time (A.U.)')
        ax.set_ylabel('k_y (A.U.)')
        ax.grid(True, which='both')
        plt.tight_layout()
        plt.show()

    # Magnetization evolution visualization
    magn = np.array(magn).reshape((Nky, Nkx, 3))
    if params.get("show_magn", False):
        echo_nr_show = params.get("echo_nr_show", [])
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        titles = ['Mz', 'My', 'Mx']
        for idx, component in enumerate(['2', '1', '0']):
            for N in echo_nr_show:
                axs[idx].plot(magn[N, :, int(component)])
            axs[idx].set_title(titles[idx])
            axs[idx].set_xlabel('Time (ms)')
        plt.tight_layout()

        # Transverse magnetization
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for N in echo_nr_show:
            magn_tr = magn[N, :, 0] + 1j * magn[N, :, 1]
            axs[0].plot(np.abs(magn_tr), label=f"Echo {N}")
            axs[1].plot(np.angle(magn_tr))
        axs[0].set_title('Magnitude of M_tr')
        axs[1].set_title('Phase of M_tr')
        axs[0].legend(fontsize=6)
        plt.tight_layout()
        plt.show()

    # Organizing k-space
    ky = phase_order[:, 2]
    magn_kspace = np.zeros((int(max(ky) - min(ky)) + 1, Nkx, 3), dtype=np.complex64)
    ky_center = int(-min(ky))
    for ki in range(Nky):
        magn_kspace[ky_center + int(ky[ki]), :, :] = magn[ki, :, :]

    # Transverse magnetization
    magn_tr = magn_kspace[:, :, 0] + 1j * magn_kspace[:, :, 1]
    recon_im = np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(magn_tr))).astype(np.complex64)

    # K-space visualization
    if params.get("show_kspace", False):
        kspace_scale = params.get("kspace_scale", 'abs')
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        kx_labels = np.linspace(-Nkx // 2, Nkx // 2, 5, dtype=int)
        ky_labels = np.linspace(min(ky), max(ky), 6, dtype=int)
        axs[0].imshow(np.log(np.abs(magn_tr)) if kspace_scale == 'log' else np.abs(magn_tr))
        axs[0].set_title('K-space Magnitude')
        axs[1].imshow(np.angle(magn_tr))
        axs[1].set_title('K-space Phase')
        plt.tight_layout()
        plt.show()

    # Image reconstruction visualization
    if params.get("show_img", False):
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs[0].imshow(np.abs(recon_im))
        axs[0].set_title('Image Magnitude')
        axs[1].imshow(np.angle(recon_im))
        axs[1].set_title('Image Phase')
        plt.tight_layout()
        plt.show()

    return magn_tr, recon_im


def compare_ksp_img(gpu_file, mpi_file, mpi_file_2, k_order, folder='./', 
                          n_channels=1, Nx=64, Ny=64, Nkx=64, Nky=64,
                          plot=False, save=False, save_figname=''):
    """
    Compare k-space and image reconstructions between GPU and MPI simulations.

    Arguments:
        gpu_file : str
            Path to GPU simulation output file.
        mpi_file : str
            Path to MPI simulation output file.
        mpi_file_2 : str
            Path to secondary MPI simulation output file for comparison.
        folder : str
            Folder to save outputs if `save` is True.
        n_channels : int
            Number of channels in the data.
        Nx, Ny : int
            Dimensions of the image space.
        Nkx, Nky : int
            Dimensions of the k-space.
        save : bool
            If True, saves the generated figures.
        save_figname : str
            Additional string for saved figure filenames.

    Returns:
        tuple:
            - ksp_data_cxy : list of k-space arrays from all files.
            - ksp_FS_mpi : Normalized k-space data from mpi_file.
            - img_data_cxy : list of image arrays from all files.
            - img_FS_mpi : Normalized image data from mpi_file.
    """
    # Step 1: Load and compare signals
    time_vectors, magn_vectors = compare_signals(gpu_file, mpi_file, mpi_file_2, folder, Nkx*Nky, n_channels, plot=plot)
    files = [gpu_file, mpi_file, mpi_file_2]
    n_files = len(files)

    ksp_data_cxy = []
    img_data_cxy = []

    # Step 2: Process each file to compute k-space and images
    for file_idx in range(n_files):
        ksp = np.zeros((n_channels, Nx, Ny), dtype=np.complex64)
        img = np.zeros_like(ksp)

        for ch in range(n_channels):
            magn = magn_vectors[file_idx][ch]
            ksp[ch, :, :], img[ch, :, :] = simulation_visualization(
                magn,
                phase_order=k_order,
                Nkx=Nkx,
                Nky=Nky,
                show_order=plot,
                show_magn=plot,
                echo_nr_show=np.arange(8),
                show_kspace=plot,
                show_img=plot,
                kspace_scale='log',
                save=save,
                name_add="",
                save_fold=folder
            )

        ksp_data_cxy.append(ksp)
        img_data_cxy.append(img)

    ksp_data_cxy = normalize_all(ksp_data_cxy, ksp_data_cxy[1])
    img_data_cxy = normalize_all(img_data_cxy, img_data_cxy[1])

    compute_and_print_errors(ksp_data_cxy[0], ksp_data_cxy[1], ksp_data_cxy[2], "K")
    compute_and_print_errors(img_data_cxy[0], img_data_cxy[1], img_data_cxy[2], "Image")

    return ksp_data_cxy, img_data_cxy

import numpy as np
import matplotlib.pyplot as plt

def plot_image_magnitude(image, ground_truth, intensity_scale='abs', nrmse_value="0.0", 
                         magnitude_threshold=1e-6, title_suffix="", filename_suffix="", 
                         save_plot=False, output_folder="./", difference_scale=2, simulation_time=""):
    """
    Plot and compare the magnitude of an image with its ground truth, and visualize the difference.

    Arguments:
        image : np.ndarray
            Input image to visualize.
        ground_truth : np.ndarray
            Ground truth image for comparison.
        intensity_scale : str, optional
            Intensity scale ('abs' supported; 'log' not currently implemented).
        nrmse_value : str, optional
            Normalized RMSE value to display on the plot.
        magnitude_threshold : float, optional
            Threshold for magnitude visualization (not used for 'abs' scale).
        title_suffix : str, optional
            Additional title for the plot.
        filename_suffix : str, optional
            String to append to saved filenames.
        save_plot : bool, optional
            Save the plot if True.
        output_folder : str, optional
            Directory to save the plot.
        difference_scale : float, optional
            Scaling factor for the difference image.
        simulation_time : str, optional
            Simulation time annotation for the plot.

    Returns:
        None
    """
    # Compute magnitude and difference
    magnitude = np.abs(image)
    difference_magnitude = magnitude - np.abs(ground_truth)

    # Set up data and titles for the two subplots
    data_to_plot = [magnitude, difference_scale * difference_magnitude]
    titles = [title_suffix, f"{title_suffix} - \nMPI-JEMRIS_DP"]

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(8, 5))
    for i, ax in enumerate(axes):
        ax.imshow(data_to_plot[i], vmax=1.0, cmap='gray')
        ax.set_title(titles[i])
        ax.set_xticks([]), ax.set_yticks([])
        ax.text(image.shape[1] - 15, image.shape[0] - 2, str(simulation_time), 
                color='white', fontsize=18)
        ax.secondary_xaxis('top').set_xlabel(f"{difference_scale}x, RMSE = {nrmse_value}%" if i else " ")

    # Finalize layout and save if required
    plt.tight_layout()
    if save_plot:
        save_path = f"{output_folder}image_{filename_suffix}.png"
        plt.savefig(save_path)
        print(f"Plot saved at: {save_path}")
    plt.show()

# for liver phantom simulations
import matplotlib.pyplot as plt
import numpy as np

def plot_multiecho(ims, save_plot=False, save_figname=''):
    """
    Plot multi-echo images in a 2x3 grid format with cropped regions.

    Parameters:
        ims (numpy.ndarray): 3D array of images (shape: [num_echoes, height, width]).
        save_plot (bool, optional): Whether to save the plot to a file. Defaults to False.
        save_figname (str, optional): Filename for saving the plot if `save_plot` is True. Defaults to 'plot_multiecho.png'.
    """
    # Crop images and calculate intensity range
    imgs_cropped = ims[:, 50:-50, 50:-50]
    max_val = np.max(np.abs(imgs_cropped))
    min_val = np.min(np.abs(imgs_cropped))
    
    # Create a 2x3 grid of subplots
    fig, axs = plt.subplots(2, 3, figsize=(8.5, 7))
    plt.subplots_adjust(hspace=0.15, wspace=0.0, top=0.82)
    
    # Populate each subplot with an echo image
    for i in range(2):
        for j in range(3):
            echo_idx = 3 * i + j  # Compute the echo index
            axs[i, j].imshow(
                np.abs(imgs_cropped[echo_idx, :, :]),
                vmin=min_val,
                vmax=max_val,
                cmap='gray'
            )
            axs[i, j].set_title(f"Echo {echo_idx + 1}", fontsize=10)
            axs[i, j].axis('off')  # Remove axes ticks

            # Add labels for the middle and left subplots
            if i == 1 and j == 1:
                axs[i, j].set_xlabel("FE", fontsize=9)
            if j == 0:
                axs[i, j].set_ylabel("PE", fontsize=9)
    
    # Adjust layout and save the plot if needed
    plt.tight_layout()
    if save_plot:
        plt.savefig(save_figname, dpi=300)
        print(f"Plot saved to {save_figname}")
    else:
        plt.show()



def process_liver_data(file_fullname, Nkx, Nky, Nk_lines, N_echoes, species, N_fat, N_water, \
    simulation_visualization, k_order, SEQ_NAME):
    """
    Process liver phantom data for multiple species and compute k-space and image space visualizations.

    Parameters:
        folder (str): Path to the data folder.
        Nkx (int): Number of k-space points in the x-direction.
        Nky (int): Number of k-space points in the y-direction.
        Nk_lines (int): Number of lines in k-space.
        N_echoes (int): Number of echoes.
        species (list): List of species identifiers.
        N_fat (int): Scaling factor for fat components.
        N_water (int): Scaling factor for water components.
        simulation_visualization (function): Visualization function for processing echoes.
        k_order (array-like): Phase encoding order for visualization.
        SEQ_NAME (str): Sequence name for visualization.

    Returns:
        tuple: (magn_all_species, fat_amplitudes, time, ksp_sum, ims_sum)
            - magn_all_species: List of magnitude data for each species.
            - fat_amplitudes: Array of maximum amplitudes for fat components.
            - time: Time data for all species.
            - ksp_sum: Combined k-space data.
            - ims_sum: Combined image space data.
    """
    # Initialize storage for species data and amplitudes
    magn_all_species = []
    fat_amplitudes = []

    # Process each species
    for component in species:
        filepath = f"{file_fullname}_{component}.h5"
        with h5py.File(filepath, 'r') as datafile:
            # Extract magnitude data and reshape
            magn = np.array(datafile['signal/channels/00'])[-Nkx * Nky * N_echoes:, :].reshape((Nky, Nkx * N_echoes, 3))
            magn_all_species.append(magn)
            
            # Calculate echo-specific amplitude
            for echo_idx in range(N_echoes):
                magn_echo = magn[:, Nkx * echo_idx:Nkx * (echo_idx + 1), :]
                if echo_idx % 2 == 0:
                    magn_echo = magn_echo[:, ::-1, :]  # Reverse along x for even echoes
            fat_amplitudes.append(np.max(np.abs(magn)))

    # Load timing data from the first species
    with h5py.File(f"{file_fullname}_{species[0]}.h5", 'r') as datafile:
        time = np.array(datafile['signal/times'])

    # Convert fat amplitudes to numpy array
    fat_amplitudes = np.array(fat_amplitudes)

    # Combine signals from all species
    N_total = 9 * N_fat + N_water
    magn_sum = magn_all_species[0] * N_water
    for i in range(2):  # Combine fat components
        magn_sum += N_fat * magn_all_species[i + 1]
    magn_sum /= N_total

    # Initialize arrays for k-space and image space sums
    ksp_sum = np.zeros((N_echoes, Nk_lines, Nkx), dtype='complex128')
    ims_sum = np.zeros((N_echoes, Nk_lines, Nkx), dtype='complex128')

    # Process each echo
    for echo_idx in range(N_echoes):
        magn_echo = magn_sum[:, Nkx * echo_idx:Nkx * (echo_idx + 1), :]
        if echo_idx % 2 == 0:
            magn_echo = magn_echo[:, ::-1, :]  # Reverse along x for even echoes

        # Use the visualization function for k-space and image space
        ksp_sum[echo_idx, :, :], ims_sum[echo_idx, :, :] = simulation_visualization(
            magn_echo,
            phase_order=k_order,
            Nkx=Nkx,
            Nky=Nky,
            SEQ_NAME=SEQ_NAME,
            show_order=False,
            show_img=False,
            show_magn=False,
            echo_nr_show=np.arange(90, 110),
            show_kspace=False
        )

    return magn_all_species, fat_amplitudes, time, ksp_sum, ims_sum
