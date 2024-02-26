### Dockerimage for the double-precision GPU-JEMRIS 
### with CUDA-12.0 and CVODE-5.7.7
##  Aizada Nurdinova - 01 Apr 2022

##  cuda version depends on your device, use nvidia-smi to check the it
ARG cuda_version  
ARG cuda_version_minor="0" 

FROM nvidia/cuda:${cuda_version}.${cuda_version_minor}.0-base-ubuntu18.04
ENV DEBIAN_FRONTEND="noninteractive" TZ="Europe/Berlin"

## NVIDIA is updating and rotating the signing keys used by apt, so install new signing keys
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt update
RUN apt-get update
RUN apt-get install -y git build-essential wget apt-utils

## libraries for the JEMRIS
RUN apt-get install -y libcln6 libcln-dev pi
RUN apt-get install -y libginac-dev 
RUN apt-get install -y libxerces-c-dev 
RUN apt-get install -y libmpich-dev libopenmpi-dev
RUN apt-get install -y libboost1.62-dev 
RUN apt-get install -y libhdf5-serial-dev 

## CUDA libraries 
ARG cuda_version
ARG cuda_version_minor
ARG cuda_version_apt=${cuda_version}-${cuda_version_minor}
RUN apt-get install -y cuda-nvcc-${cuda_version_apt} libcusolver-dev-${cuda_version_apt} 
RUN apt-get install -y libcusparse-dev-${cuda_version_apt} libcublas-dev-${cuda_version_apt} libcurand-dev-${cuda_version_apt} cuda-nsight-systems-${cuda_version_apt}
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/usr/local/include/"
ENV PATH="/usr/local/cuda-${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}/bin:$PATH"

## newer cmake is needed for CVODE-5.7 - 
WORKDIR lib/
RUN wget https://github.com/Kitware/CMake/releases/download/v3.21.6/cmake-3.21.6-linux-x86_64.sh
RUN sh cmake-3.21.6-linux-x86_64.sh --skip-license
ENV PATH="/lib/bin/:$PATH" 
# if press no - export PATH="/lib/bin/:$PATH"
# if press yes - export PATH="/lib/cmake-3.23.0-rc5-linux-x86_64/bin:$PATH"

## ISMRMD installation
RUN apt-get install -y --fix-missing doxygen git-core graphviz libfftw3-dev libpugixml-dev
# ismrmd compilation fails with boost1.65 - use v1.62
#RUN git clone https://github.com/ismrmrd/ismrmrd
RUN wget https://github.com/ismrmrd/ismrmrd/archive/refs/tags/v1.13.2.tar.gz 
RUN  tar -xzvf v1.13.2.tar.gz; cd ./ismrmrd-1.13.2/; mkdir build; cd build; cmake ../; make; make install

## CVODE 
ARG sundials_precision
RUN wget https://github.com/LLNL/sundials/releases/download/v5.7.0/cvode-5.7.0.tar.gz
RUN tar -xzvf cvode-5.7.0.tar.gz
RUN cd cvode-5.7.0; mkdir build; cd build; cmake -DSUNDIALS_PRECISION=${sundials_precision} -DENABLE_CUDA=ON -DEXAMPLES_ENABLE_CUDA=ON -DENABLE_MPI=ON -DENABLE_OPENMP=ON -DPTHREAD_ENABLE=ON ../; make; make install

## GPU-JEMRIS newest version - needs autorization
# RUN git clone https://gitlab.lrz.de/BMRR/tools/gjemris.git
# RUN cd jemris/; mkdir build; cd build; cmake -DMODEL_ON_GPU=1 ../; make; ctest -V; make install;
# to install CPU-JEMRIS with CVode-v5.7 instead of the v2.* in jemris-2.8
# RUN cd jemris/build; rm -rf ./*; cmake ../; make; ctest -V; make install;

