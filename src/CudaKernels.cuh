/** @file CudaKernels.cuh
 *  @brief Implementation of GPU-JEMRIS kernels
 */
#include <cuda_runtime.h>
#include "Bloch_CV_Model.h" 
#include <cub/cub.cuh>

/* For Bloch_CV_Model.cpp */
__global__ void BlochKernel (realtype *y, realtype *y_dot,
                    realtype* d_SeqVal, bool tx_ideal, realtype* tx_coils_sum, 
                    realtype* s_vals, realtype* dB, realtype* positions, 
                    realtype* NonLinGradField, realtype GMAXoverB0,
                    int N_SpinProps, int N_spins_total,
                    int N_spins_stream);

__global__ void SolutionToNVectorKernel (realtype* sol, realtype* p_y, int N_spins_stream); 


/* For Coil.cpp */
__device__ int get_index (int px, int py, int pz, size_t* m_dims);

__device__ double UnwrapGPU(double checkwrap, bool magnitude);

__global__ void InterpolateSensitivityKernel (double* dst_sens, double* dst_phase, bool magnitude_only,
		double* src_magn, double* src_phase, realtype* positions, 
		double m_phase, double m_scale, bool m_conjugate, size_t* m_dims, int N_spins,
		realtype m_extent, int m_points, unsigned m_dim);

__global__ void Convert2double3 (double3* d_vec_out, realtype* d_vec_in, 
        int N_items, bool rx_ideal, bool m_rx_external,
		double* d_sens_all, double* d_phase_all, double phase_offset, double m_phase);


/* For Model.cpp */
__global__ void InitSolutionKernel (realtype* solution, realtype* values, int NoOfSpinProps,
	 	int NoOfCompartments, int SpinOffset, int NoSpinsStream);

/* For Sample.cpp */
__global__ void InitRNG(curandState *state);
__global__ void AddPosRandomGetDeltaBKernel (realtype* val, realtype* dB_all, 
    realtype* positions, int NoOfCompartments, int NoOfSpinProps, 
	realtype* d_m_res, int SpinOffset, int N_spins_stream, curandState_t* st);
