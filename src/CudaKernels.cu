/** @file CudaKernels.cu
 *  @brief Implementation of GPU-JEMRIS kernels
 */

#include "CudaKernels.cuh"

/**********************************************************/
// AN-2022: Bloch kernel for GPU computations
// called in the RHS function, in Bloch_CV_Model.cpp
__global__ void BlochKernel (realtype *y, realtype *y_dot,
                    realtype* d_SeqVal, bool tx_ideal, realtype* tx_coils_sum, 
                    realtype* s_vals, realtype* dB, realtype* positions, 
                    realtype* NonLinGradField, realtype GMAXoverB0,
                    int N_SpinProps, int N_spins_total,
                    // required only for the multi-stream option
                    int N_spins_stream) {
                        
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N_spins_stream) {
        int spin_id = tid; // if want multi-stream computations: + iter_stream*N_spins_stream
        // cylndrical components of mangetization
        realtype Mxy, phi, Mz; 
        realtype s, c, Mx, My, Mx_dot, My_dot, Mz_dot;
        realtype Bx, By, Bz;
        realtype DeltaB = *(dB+spin_id);
        // sample properties    
        realtype r1 = s_vals[spin_id*N_SpinProps+R1];
        realtype r2 = s_vals[spin_id*N_SpinProps+R2];
        realtype r2s = s_vals[spin_id*N_SpinProps+R2S];
        realtype m0 = s_vals[spin_id*N_SpinProps+M0];
        // RF phase at the timepoint
        realtype phase = d_SeqVal[RF_PHS];

        // calculating Bx and By is faster with uniform-Tx coil 
        // math functions on GPU are different for single and double precision
        if (tx_ideal) {
            Bx = d_SeqVal[RF_AMP]*cos(phase);
            By = d_SeqVal[RF_AMP]*sin(phase);
        }
        else {
            phase = fmod( phase, (realtype)(2*PI) ); 
            Bx = d_SeqVal[RF_AMP] * (tx_coils_sum[spin_id]) * cos(phase);
            By = d_SeqVal[RF_AMP] * (tx_coils_sum[spin_id]) * sin(phase);
        }

        Bz = positions[3*spin_id]*(*(d_SeqVal+GRAD_X))+ positions[3*spin_id+1]*(*(d_SeqVal+GRAD_Y))+ positions[3*spin_id+2]*(*(d_SeqVal+GRAD_Z))
            + DeltaB;

        // Add non-linear gradients if present
        if (NonLinGradField) 
            Bz += NonLinGradField[spin_id];
        
        // concominant field calculation step
        if (GMAXoverB0 != 0.0) 
            Bz += ((0.5*GMAXoverB0)*(pow(GRAD_X*positions[3*spin_id+2]-0.5*GRAD_Z*positions[3*spin_id],2) + 
                    pow(GRAD_Y*positions[3*spin_id+2]-0.5*GRAD_Z*positions[3*spin_id+1],2)));

        // restrict phase to [0, 2*PI]
        if (fabs(y[3*tid+PHASE]) > 1e11) {
            y[3*tid+PHASE] = fmod(y[3*tid+PHASE],(realtype)(TWOPI));
        }

        Mxy = y[3*tid + AMPL];
        phi = y[3*tid + PHASE];
        Mz = y[3*tid + ZC];

        // avoid CVODE warnings (does not change physics!)
        // trivial case: no transv. magnetisation AND no excitation
        if (Mxy<ATOL1*m0 && (d_SeqVal[RF_AMP])<BEPS) {

            y_dot[3*tid+AMPL] = 0;
            y_dot[3*tid+PHASE] = 0;
            //further, longit. magnetisation already fully relaxed
            if (fabs(m0 - Mz)<ATOL3) {
                y_dot[3*tid+ZC] = 0.;
                return;
            }

        } else {

            //compute cartesian components of transversal magnetization
            c = cos(phi);
            s = sin(phi);
            Mx = c*Mxy;
            My = s*Mxy;

            //compute bloch equations
            Mx_dot =   Bz*My - By*Mz - r2*Mx;
            My_dot = - Bz*Mx + Bx*Mz - r2*My;
            Mz_dot =   By*Mx - Bx*My ;

            //compute derivatives in cylindrical coordinates
            y_dot[3*tid+AMPL]  =  c*Mx_dot + s*My_dot;
            y_dot[3*tid+PHASE] = (c*My_dot - s*Mx_dot) / (Mxy>BEPS?Mxy:BEPS); //avoid division by zero
        }

        //longitudinal relaxation
        Mz_dot +=  r1*(m0 - Mz);
        y_dot[3*tid+ZC] = Mz_dot;
    }
}


/**********************************************************/
// kernel to assign to NVector values from the solution vector
__global__ void SolutionToNVectorKernel (realtype* sol, realtype* p_y, int N_spins_stream) {
    int spin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (spin_idx < N_spins_stream) {
        p_y[3*spin_idx+0] = sol[3*spin_idx+AMPL];
        p_y[3*spin_idx+1] = fmod(sol[3*spin_idx+PHASE], (realtype)(TWOPI));
        p_y[3*spin_idx+2] = sol[3*spin_idx+ZC];
    }
}

/* For Coil.cpp */
/**********************************************************/
// AN-2022: helpers for the interpolation on GPU 
__device__ int get_index (int px, int py, int pz, size_t* m_dims) {
	return pz*m_dims[0]*m_dims[1]+py*m_dims[0]+px;
}

__device__ double UnwrapGPU(double checkwrap, bool magnitude) {
	// only for phase interpolation:
	if (magnitude) return checkwrap;
	const double wrapfact = 1; // factor to determine when a phase wrap is likely to be detected.
	if (checkwrap>PI*wrapfact) checkwrap-=2*PI; if (checkwrap<-PI*wrapfact) checkwrap+=2*PI;
	return checkwrap;
}

// AN-2022: interpolating coil sensitivity maps on GPU 
__global__ void InterpolateSensitivityKernel (double* dst_sens, double* dst_phase, bool magnitude_only,
		double* src_magn, double* src_phase, realtype* positions, 
		double m_phase, double m_scale, bool m_conjugate, size_t* m_dims, int N_spins,
		realtype m_extent, int m_points, unsigned m_dim) {
		
		unsigned spin_idx = blockIdx.x * blockDim.x + threadIdx.x;
		if (spin_idx < N_spins) {
			
			realtype x = (positions[3*spin_idx+XC] + m_extent/2) * m_points / m_extent;
			realtype y = (positions[3*spin_idx+YC] + m_extent/2) * m_points / m_extent;
			realtype z = (m_dim==3 ? ((positions[3*spin_idx+ZC] + m_extent/2) * m_points / m_extent) : 0.0);
			
			int    px = int(x),  py = int(y), pz = int(z);
			double normx = (double)(x-px),	normy = (double)(y-py), normz = (double)(z-pz);

			// check if point is on lattice
			if (px > m_points-1 || px < 0 || py > m_points-1 || py < 0 || pz > m_points-1 || pz < 0 ) {
				dst_sens[spin_idx] = 0.0;
				if (!magnitude_only) dst_phase[spin_idx] = 0.0;
				return;
			}

			// bilinear interpolation (2D)
			int nx = (px+1 < m_points ? px+1 : m_points-1);
			int ny = (py+1 < m_points ? py+1 : m_points-1);
			double i11, i21;
			i11 = src_magn[get_index(px,py,pz,m_dims)] + (src_magn[get_index(px,ny,pz,m_dims)]-
				src_magn[get_index(px,py,pz,m_dims)])*normy;
			i21 = src_magn[get_index(nx,py,pz,m_dims)] + (src_magn[get_index(nx,ny,pz,m_dims)]-
				src_magn[get_index(nx,py,pz,m_dims)])*normy;
			double iz1 = i11+UnwrapGPU(i21-i11,true)*normx;
			double piz1;
			if (!magnitude_only) {
				double pi11, pi21;
				pi11 = src_phase[get_index(px,py,pz,m_dims)] + UnwrapGPU(src_phase[get_index(px,ny,pz,m_dims)] - 
					src_phase[get_index(px,py,pz,m_dims)],magnitude_only) * normy;
				pi21 = src_phase[get_index(nx,py,pz,m_dims)] + UnwrapGPU(src_phase[get_index(nx,ny,pz,m_dims)] - 
					src_phase[get_index(nx,py,pz,m_dims)],magnitude_only) * normy;
				piz1 = pi11 + UnwrapGPU(pi21-pi11,magnitude_only) * normx;
			}
			
			//check 2D
			if (m_dim<3) {
				dst_sens[spin_idx] = m_scale * iz1;
				if (!magnitude_only) {
					dst_phase[spin_idx] = m_phase + (m_conjugate?-1.0:1.0) * piz1;
				}
				return;
			}

			// trilinear interpolation (3D)
			int nz = (pz+1 < m_points ? pz+1 : m_points-1);
			double i12, i22;
			i12 = src_magn[get_index(px,py,nz,m_dims)] + UnwrapGPU(src_magn[get_index(px,ny,nz,m_dims)] -
				src_magn[get_index(px,py,nz,m_dims)],true) * normy;
			i22 = src_magn[get_index(nx,py,nz,m_dims)] + UnwrapGPU(src_magn[get_index(nx,ny,nz,m_dims)] -
				src_magn[get_index(nx,py,nz,m_dims)],true) * normy;
			double iz2 = i12 + UnwrapGPU(i22-i12,true) * normx;
			double piz2;
			if (!magnitude_only) {
				double pi12, pi22;
				pi12 = src_phase[get_index(px,py,nz,m_dims)] + UnwrapGPU(src_phase[get_index(px,ny,nz,m_dims)] -
					src_phase[get_index(px,py,nz,m_dims)],magnitude_only) * normy;
				pi22 = src_phase[get_index(nx,py,nz,m_dims)] + UnwrapGPU(src_phase[get_index(nx,ny,nz,m_dims)] -
					src_phase[get_index(nx,py,nz,m_dims)],magnitude_only) * normy;
				piz2 = pi12 + UnwrapGPU(pi22-pi12,magnitude_only) * normx;
			}
			
			dst_sens[spin_idx] = m_scale * (iz1 + UnwrapGPU(iz2-iz1,true) * normz);
			if (!magnitude_only) {
				dst_phase[spin_idx] = m_phase + (m_conjugate?-1.0:1.0) * (piz1 + UnwrapGPU(piz2-piz1,magnitude_only) * normz);
			}	
		}
}


/**********************************************************/
// AN-2022: converting the solution array into the CUDA double3 format 
// 			to prepare for a reduction operation on GPU
__global__ void Convert2double3 (double3* d_vec_out, realtype* d_vec_in, int N_items, bool rx_ideal,
		bool m_rx_external,
		double* d_sens_all, double* d_phase_all, double phase_offset, double m_phase) {
			// m_phase is the initial phase shift
    int spin_idx =  blockIdx.x * blockDim.x + threadIdx.x;
    if (spin_idx < N_items) {
		if (rx_ideal) {
			d_vec_out[spin_idx].x = (double)(d_vec_in[3*spin_idx] * cos(m_phase+phase_offset + d_vec_in[3*spin_idx+1]));
			d_vec_out[spin_idx].y = (double)(d_vec_in[3*spin_idx] * sin(m_phase+phase_offset + d_vec_in[3*spin_idx+1]));
			d_vec_out[spin_idx].z = (double)(d_vec_in[3*spin_idx+2]);
		} else {
			double tm;
			if (m_rx_external) {
				tm = (m_phase + phase_offset + d_phase_all[spin_idx] + (double)d_vec_in[3*spin_idx+1]);
			} else {
				tm = (m_phase + phase_offset + d_phase_all[0] + (double)d_vec_in[3*spin_idx+1]);
			}
			d_vec_out[spin_idx].x = (double)(d_vec_in[3*spin_idx]) * d_sens_all[spin_idx] * cos(tm); 
			d_vec_out[spin_idx].y = (double)(d_vec_in[3*spin_idx]) * d_sens_all[spin_idx] * sin(tm);
			d_vec_out[spin_idx].z = (double)(d_vec_in[3*spin_idx+2]) * d_sens_all[spin_idx];
		}
	}
}
/* END: For Coil.cpp */

/* For Model.cpp */
/**************************************************/
// GPU kernel to init the solution vector by M0 
__global__ void InitSolutionKernel (realtype* solution, realtype* values, int NoOfSpinProps,
	 	int NoOfCompartments, int SpinOffset, int NoSpinsStream) {

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int spin_idx = SpinOffset + tid;
	if (tid < NoSpinsStream) {
		// if compartments are included
		// for (int i=0; i<NoOfCompartments; i++){
		solution[3*spin_idx+0] = 0.0;
		solution[3*spin_idx+1] = 0.0;
		solution[3*spin_idx+2] = 1.0 * values[spin_idx*NoOfSpinProps+3]; 
	}
}
/* END: For Model.cpp */

/* For Sample.cpp */
/**********************************************************/
// initialize random states vector of size streamsize = N spins on a stream
__global__ void InitRNG(curandState *state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init((unsigned long)clock(), idx, 0, &state[idx]);
}

/**********************************************************/
// GPU kernel to add spin position randomness and randommness to the deltaB values if R2s!=R2
__global__ void AddPosRandomGetDeltaBKernel (realtype* val, realtype* dB_all, realtype* positions, 
	int NoOfCompartments, int NoOfSpinProps, 
	realtype* d_m_res, int SpinOffset, int N_spins_stream, curandState_t* st){ 

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int spin_idx = SpinOffset + tid;
	if (tid < N_spins_stream)	{ 
#if defined(SUNDIALS_SINGLE_PRECISION) 
		val[spin_idx*NoOfSpinProps+XC] += curand_normal(&st[tid]) * d_m_res[XC] * d_m_res[3] / 100.0;		
		val[spin_idx*NoOfSpinProps+YC] += curand_normal(&st[tid]) * d_m_res[YC] * d_m_res[3] / 100.0;
		val[spin_idx*NoOfSpinProps+ZC] += curand_normal(&st[tid]) * d_m_res[ZC] * d_m_res[3] / 100.0;
#elif defined(SUNDIALS_DOUBLE_PRECISION) 
		val[spin_idx*NoOfSpinProps+XC] += curand_normal_double(&st[tid]) * d_m_res[XC] * d_m_res[3] / 100.0;		
		val[spin_idx*NoOfSpinProps+YC] += curand_normal_double(&st[tid]) * d_m_res[YC] * d_m_res[3] / 100.0;
		val[spin_idx*NoOfSpinProps+ZC] += curand_normal_double(&st[tid]) * d_m_res[ZC] * d_m_res[3] / 100.0;
#endif
		// extract spin positions to the separate array
		positions[3*spin_idx+0] = val[spin_idx*NoOfSpinProps+XC];
		positions[3*spin_idx+1] = val[spin_idx*NoOfSpinProps+YC];
		positions[3*spin_idx+2] = val[spin_idx*NoOfSpinProps+ZC];
		
		realtype r2s = val[spin_idx*NoOfSpinProps+R2S];
		realtype r2 = val[spin_idx*NoOfSpinProps+R2];
		realtype r2prime = ((r2s > r2) ? (r2s - r2) : 0.0);
		// 0.001 is bcs we compute in ms, but the value is in secs 
		dB_all[spin_idx] = (0.001*(val[spin_idx*NoOfSpinProps+DB])) + tan(PI*(curand_uniform_double(&st[tid])-.5)) * r2prime;

	}
}
/* END: For Sample.cpp */




