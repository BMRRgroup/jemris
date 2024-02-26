General Information
===================

JEMRIS is a general MRI simulation framework.

The general process of simulation consists of preparation by choice or
implementation of sequence, sample and coil setup and the invocation
of the simulation run itself.  


Documentation
=============

It is _highly_ recommended to read the provided documentation online.
Please find the build, install, developer and user documentation under
http://www.jemris.org.


Licensing
=========

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 
02110-1301  USA

For an explicit declaration of licensing refer to the COPYING file in
the root directory of this package.


Contact
=======

This package is maintained by Tony Stoecker <tony.stoecker@dzne.de>.
Please find problem specific contact addresses on http://www.jemris.org.


Installation
============

Please visit http://www.jemris.org for detailed installation instructions.


How to report bugs
==================

If you have identified a bug in JEMRIS you are welcome to send a detailed
bug report to <tony.stoecker@dzne.de>. Please include

* Information about your build configuration

   - Contents of the config.log file.

* Information about your system

   - Which operating system and version (uname -a)
   - Which C compiler and version (gcc --version)
   - For Linux, which version of the C library

  And anything else you think is relevant.

* Information about your version of JEMRIS

   - Version and release number
   - Which options GiNaC was configured with

* How to reproduce the bug

   - If it is a systematical bug in JEMRIS please provide the
     sequence, the sample, the coils and the outputs from sequence or
     simulation GUI to help us to reproduce the bug quickly.

Patches are most welcome.  If possible please make them with diff -c and
include ChangeLog entries.

GPU-JEMRIS
==========
The simulator is adapted to move data to device and solve Bloch equations for all spins in parallel on GPU.

* **Installation**:

   - CUDA compiler and necessary CUDA libraries can be installed as:

		    apt-get install cuda-nvcc-${CUDA_VERSION} libcusolver-dev-${CUDA_VERSION} libcusparse-dev-${CUDA_VERSION} libcublas-dev-${CUDA_VERSION} libcurand-dev-${CUDA_VERSION} 

   - JEMRIS prerequisites need to be met, however, a newer version of CVode package should be installed for the GPU support. 
   	 Additionally, GPU-JEMRIS can perform computations in **single OR double precision** The chosen precision needs be specifed in the CVode installation using the following commands:

 		
		 wget https://github.com/LLNL/sundials/releases/download/v5.7.0/cvode-5.7.0.tar.gz;
		 tar -xzvf cvode-5.7.0.tar.gz; cd cvode-5.7.0; mkdir build; cd build;
		 cmake -DSUNDIALS_PRECISION=double ../ -DENABLE_CUDA=ON -DEXAMPLES_ENABLE_CUDA=ON -DENABLE_MPI=ON -DENABLE_OPENMP=ON -DPTHREAD_ENABLE=ON;
		 make; make install;

		Note: a newer cmake might be needed for the higher CVode versions, you can try smth like:
				
		 wget https://github.com/Kitware/CMake/releases/download/v3.21.6/cmake-3.21.6-linux-x86_64.sh;
		 sh cmake-3.21.6-linux-x86_64.sh --skip-license; export PATH="/lib/bin/:$PATH";

   - GPU-JEMRIS compilation is separate from the CPU version:

	     mkdir build; cd ./build; cmake -DMODEL_ON_GPU=1 ../;
	     make (-j 8); ctest -V; make install;
		
		Note: with the flag MODEL_ON_GPU=1, the CUDA scripts get compiled building a *gjemris* executable. The flag *-j (n)* allows for faster parallelized compilation. 
 
   - A Dockerfile installing all prerequisites and compiling GPU-JEMRIS is provided in ./docker/. You can build an image using:
	
		 docker build -t <image_name> --build-arg cuda_version=<12> --build-arg sundials_precision=<double> -f ./Dockerfile .


* **Tests for GPU-JEMRIS**:

   - **ctest** includes the following tests for the GPU simulator in both single and double precision:
		1. Run GPU-simulations with a simple sample - compare the signals to the approved signals from jemris-v2.8.
		2. Run GPU-simulations with a simple sample - if *jemris* (the CPU version) is installed and ctest ran for the CPU version, the new GPU signals are compared to the signals from jemris-cvode5.7.

		The list of sequences on which the GPU-JEMRIS is tested (find the files in ./share/examples/approved/):
			
		"ThreePulses.xml", "epi.xml", "gre.xml", "tse.xml", "radial.xml", "radial2.xml", "spiral.xml", "sli_sel.xml", "var_dur.xml", "extpulses.xml", "epi_modular.xml", "trapezoid.xml"; 


   - **CUDA profiling**: 
	
		[NVIDIA Nsight systems](https://docs.nvidia.com/nsight-systems/UserGuide/index.html) is a performance analysing tool that helps profiling and optimizing GPU applications. It can be used to track GPU-CPU interactions, and GPU activity and workload.		     
		To install CLI and profile with CLI on Linux machines:
		  
		  apt-get install -y cuda-nsight-systems-<CUDA_VERSION>;
   		  nsys profile [--stats=true] [-o <report_name>] <command>
		Export files can be generated in several formats, we used .qdrep in order to visualize reports using Nsight Systems GUI.
     	To visulaize nsys reports, the GUI needs to be installed on Windows/Linux/MacOS machines. Copy and open the exported report files. An examplary .qdrep report is provided in the ./nsysProf/ folder.

* **Limitations**: 
	
	GPU-JEMRIS does not include the following functionalities:
	- dynamic effects: diffusion, flow, repiration, T2s;
	- multi-Tx;
	- Bloch-McConnell model.

* **Contact** for any questions regarding the GPU patch:
	
	Aizada Nurdinova - [nurdaiza@stanford.edu](nurdaiza@stanford.edu)
