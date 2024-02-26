/** @file Coil.cpp
 *  @brief Implementation of JEMRIS Coil
 */

/*
 *  JEMRIS Copyright (C) 
 *                        2006-2023  Tony Stoecker
 *                        2007-2018  Kaveh Vahedipour
 *                        2009-2019  Daniel Pflugfelder
 *                                  
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 */

#include "Coil.h"
#include "Model.h"
#include "DynamicVariables.h"

// AN-2022 
#ifdef MODEL_ON_GPU
#include <cub/cub.cuh>
#endif	// AN-2022***


/**********************************************************/
Coil::~Coil() {
	
	if (m_signal   != NULL) delete    m_signal;
#ifdef MODEL_ON_GPU
	cudaFreeHost(h_sol);
	cudaFree(d_sol);
	cudaFree(d_sol_vec);
	cudaFree(d_temp_storage);
	World* pW = World::instance();
	if (pW->m_rx_external)
		cudaFree(senspha_gpu);
	if (pW->dynamic || m_interpolate) {
		cudaFree(sensmag_gpu_dyn);
		if (pW->m_rx_external)
			cudaFree(senspha_gpu_dyn);
	}
	if (!(pW->m_rx_ideal)) {
		cudaFree(map_dims);
		cudaFree(sensmag_gpu);
	}
#endif
	
}

/**********************************************************/
void Coil::Initialize (DOMNode* node) {
	
	m_node     = node;
	
	string s = StrX(((DOMElement*) node)->getAttribute (StrX("Name").XMLchar() )).std_str() ;
	
	if (s.empty()) {
		((DOMElement*) node)->setAttribute(StrX("Name").XMLchar(),node->getNodeName());
		SetName( StrX(node->getNodeName()).std_str() );
	}
	
	
}

/**********************************************************/
void Coil::InitSignal(long lADCs) {
	
    if (m_signal!=NULL)
        delete m_signal;
	
    m_signal = new Signal (lADCs, World::instance()->GetNoOfCompartments());
	
}

/**********************************************************/
void Coil::Receive (long lADC) {
	
	World* pW = World::instance();
    m_signal->Repo()->TP(lADC) = pW->time;
	
    double sens  = GetSensitivity (m_signal->Repo()->TP(lADC));
    double phase = GetPhase       (m_signal->Repo()->TP(lADC));
	
	long   pos   = m_signal->Repo()->Position(lADC); 
	
	double tm    = 0.0;

	for (int i = 0; i < m_signal->Repo()->Compartments(); i++) {

		tm = - pW->phase + phase + World::instance()->solution[PHASE+ i*3];

		m_signal->Repo()->at(pos +     i*3) += sens  * pW->solution[i*3 + AMPL] * cos (tm);
		m_signal->Repo()->at(pos + 1 + i*3) += sens  * pW->solution[i*3 + AMPL] * sin (tm);
		m_signal->Repo()->at(pos + 2 + i*3) += sens  * pW->solution[i*3 + 2];

	}
}

/**********************************************************/
void Coil::GridMap () {

    double position[3]  = {0.0,0.0,0.0};
    double max = 0.0;

    for (int k=0; k< (m_dim==3?m_points:1); k++) {

        position [ZC] = (m_dim==3?k*m_extent/m_points-m_extent/2:0.0);

        for (int j=0; j<m_points; j++) {

            position [YC] = j*m_extent/m_points-m_extent/2;

            for (int i=0; i<m_points; i++) {

                position [XC] = i*m_extent/m_points-m_extent/2;
                double mag   = m_scale*GetSensitivity(position);
                m_sensmag(i,j,k) = mag;
                max = (max>mag?max:mag);
                m_senspha(i,j,k) = ( (m_conjugate?-1.0:1.0) * ( GetPhase(position) + m_phase) );
                if (m_senspha(i,j,k) != 0.0) m_complex = true;
            }
        }
    }
    m_norm = 1/max;

}

double* Coil::PhaseMap () {
	return m_senspha.Ptr();
}

double* Coil::MagnitudeMap () {
	return m_sensmag.Ptr();
}

int Coil::GetPoints () {
	return m_points;
}

int Coil::GetExtent () {
	return m_extent;
}

unsigned Coil::GetNDim () {
	return m_dim;
}


/**********************************************************/
double  Coil::GetPhase (const double time) {

    if (!m_complex) return m_phase;

    double position[3];
    position[0]=World::instance()->Values[XC];
    position[1]=World::instance()->Values[YC];
    position[2]=World::instance()->Values[ZC];
    DynamicVariables* dv = DynamicVariables::instance();
    dv->m_Motion->GetValue(time,position);
//MODIF
    dv->m_Flow->GetValue(time,position);
//MODIF***
    dv->m_Respiration->GetValue(time,position);
//Mod

//

    if (m_interpolate)
		return ( m_phase + (m_conjugate?-1.0:1.0) * InterpolateSensitivity(position,false));
	else
		return ( m_phase + (m_conjugate?-1.0:1.0) * GetPhase(position));
}

/**********************************************************/
double  Coil::GetSensitivity (const double time) {

    double position[3];
    position[0]=World::instance()->Values[XC];
    position[1]=World::instance()->Values[YC];
    position[2]=World::instance()->Values[ZC];
    DynamicVariables* dv = DynamicVariables::instance();
    dv->m_Motion->GetValue(time,position);
//MODIF
    dv->m_Flow->GetValue(time,position);
//MODIF***
    dv->m_Respiration->GetValue(time,position);
//Mod

//

	if (m_interpolate) {
		return m_scale*InterpolateSensitivity(position);
	}
	else {
		return m_scale*GetSensitivity(position);
	}
}


/**********************************************************/
double Coil::InterpolateSensitivity (const double* position, bool magnitude){

	// expects  -m_extent/2 <= position[j] <= m_extent/2
    double x = (position[XC]+m_extent/2)*m_points/m_extent;
    double y = (position[YC]+m_extent/2)*m_points/m_extent;
    double z = (m_dim==3?(position[ZC]+m_extent/2)*m_points/m_extent:0.0);
	int    px   = int(x),  py   = int(y), pz   = int(z);
	double normx = x-px ,	normy = y-py ,	normz = z-pz;

	//check if point is on lattice
	if (px>m_points-1 || px<0 || py>m_points-1 || py<0 || pz>m_points-1 || pz<0 ) return 0.0;

    //bilinear interpolation (2D)
	int nx = (px+1<m_points?px+1:m_points-1);
	int ny = (py+1<m_points?py+1:m_points-1);
	double i11, i21;
	if (magnitude) {
		i11 = m_sensmag(px,py,pz) + (m_sensmag(px,ny,pz)-m_sensmag(px,py,pz))*normy;
		i21 = m_sensmag(nx,py,pz) + (m_sensmag(nx,ny,pz)-m_sensmag(nx,py,pz))*normy;
	} else {
		i11 = m_senspha(px,py,pz)+Unwrap(m_senspha(px,ny,pz)-m_senspha(px,py,pz),magnitude)*normy;
		i21 = m_senspha(nx,py,pz)+Unwrap(m_senspha(nx,ny,pz)-m_senspha(nx,py,pz),magnitude)*normy;
	}
	double iz1 = i11+Unwrap(i21-i11,magnitude)*normx;

	//check 2D
	if (m_dim<3) return iz1;

    //trilinear interpolation (3D)
	int nz = (pz+1<m_points?pz+1:m_points-1);
	double i12, i22;
	if (magnitude) {
		i12 = m_sensmag(px,py,nz)+Unwrap(m_sensmag(px,ny,nz)-m_sensmag(px,py,nz),magnitude)*normy;
		i22 = m_sensmag(nx,py,nz)+Unwrap(m_sensmag(nx,ny,nz)-m_sensmag(nx,py,nz),magnitude)*normy;
	} else {
		i12 = m_senspha(px,py,nz)+Unwrap(m_senspha(px,ny,nz)-m_senspha(px,py,nz),magnitude)*normy;
		i22 = m_senspha(nx,py,nz)+Unwrap(m_senspha(nx,ny,nz)-m_senspha(nx,py,nz),magnitude)*normy;
	}
	double iz2 = i12+Unwrap(i22-i12,magnitude)*normx;

	return (iz1+Unwrap(iz2-iz1,magnitude)*normz);

}

/**********************************************************/
double Coil::Unwrap(double checkwrap, bool magnitude){
	// only for phase interpolation:
	if (magnitude) return checkwrap;
	const double wrapfact = 1; // factor to determine when a phase wrap is likely to be detected.
	if (checkwrap>PI*wrapfact) checkwrap-=2*PI; if (checkwrap<-PI*wrapfact) checkwrap+=2*PI;
	return checkwrap;
}

/**********************************************************/

bool Coil::Prepare  (const PrepareMode mode) {

	bool success = false;
	m_azimuth = 0.0;
	m_polar   = 0.0;
	m_scale   = 1.0;
	m_norm    = 1.0;
	m_phase   = 0.0;
	m_dim     = 3;
	m_extent  = 0;
	m_points  = 0;
	m_complex = false;
	m_conjugate = false;

	ATTRIBUTE("XPos"   , m_position [XC]);
	ATTRIBUTE("YPos"   , m_position [YC]);
	ATTRIBUTE("ZPos"   , m_position [ZC]);
	ATTRIBUTE("Azimuth", m_azimuth      );
	ATTRIBUTE("Polar"  , m_polar        );
	ATTRIBUTE("Scale"  , m_scale        );
	ATTRIBUTE("Phase"  , m_phase        );
	ATTRIBUTE("Conj"   , m_conjugate    );
	ATTRIBUTE("Dim"    , m_dim          );
	ATTRIBUTE("Extent" , m_extent       );
	ATTRIBUTE("Points" , m_points       );

	m_mode		= mode;
	m_signal	= NULL;

    success = Prototype::Prepare(mode);

    m_phase   *= PI/180.0;
    m_polar   *= PI/180.0;
    m_azimuth *= PI/180.0;
    m_interpolate = (m_points>0 && m_extent>0.0);

    // dimensions with m_points==0 may lead to undefined memory access
    if (m_points==0) m_points=1;
    m_sensmag = NDData<double> (m_points, m_points, (m_dim==3?m_points:1));
    m_senspha = NDData<double> (m_points, m_points, (m_dim==3?m_points:1));

    return success;

}

// AN-2022: functions for the GPU model
#ifdef MODEL_ON_GPU 
/**********************************************************/
// AN-2022: prepare the coil maps for tranfer to GPU
void Coil::GetMapsAll (double* sens_magn_all, double* sens_phase_all, double* sample_values, 
		int N_props, bool magnitude_only) {

    World* pW = World::instance();

	double position[3];
    for (int spin_ii=0; spin_ii<pW->TotalSpinNumber; spin_ii++){
		position[0] = sample_values[N_props*spin_ii+XC];
		position[1] = sample_values[N_props*spin_ii+YC];
		position[2] = sample_values[N_props*spin_ii+ZC];
		sens_magn_all[spin_ii] = m_scale*GetSensitivity(position);
		if (!magnitude_only)
			sens_phase_all[spin_ii] = ( m_phase + (m_conjugate?-1.0:1.0) * GetPhase(position)); 
		
	}
}

/**********************************************************/
// AN-2022: allocating memory for the Receive on GPU
// different arrays for different channels
void Coil::InitSolutionArraysGPU () {

	// temporary storage for the computed signals on host and device
	cudaMallocHost ( (void**)&h_sol, sizeof(double3) );
	cudaMalloc ( (void**)&d_sol, sizeof(double3) );

	World* pW = World::instance();
	// stores double3 array of (sens*solution)
	cudaMalloc ( (void**)&d_sol_vec, (pW->TotalSpinNumber)*sizeof(double3) );

	// memory for the reduce operation in cub
	d_temp_storage = NULL;
	temp_storage_bytes = 0;
	cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sol_vec, d_sol, pW->TotalSpinNumber, sum_op, init);
    cudaMalloc((void**)&d_temp_storage, temp_storage_bytes);
	// initial computed signal value
	init.x = init.y = init.z = 0.;
}

/**********************************************************/
// AN-2022: initialize coil sensitivity maps on GPU
void Coil::InitCoilSensGPU (size_t* sample_dims, double* sample_values, cudaStream_t stream) {

	World* pW = World::instance();
	if (m_interpolate) {
		// just copy the existing interpolated maps to GPU
		int n_points = 1;
		for (int i=0; i<m_dim; i++) {
			n_points *= m_points;
		} 

		cudaMalloc( (void**)&sensmag_gpu, n_points*sizeof(double) );
		if (pW->m_rx_external) {
			// ony external coil type needs coil phase maps
			cudaMalloc( (void**)&senspha_gpu, n_points*sizeof(double) );
		}
		// the maps are ready to copy anytime - use Async
		cudaMemcpyAsync( sensmag_gpu, m_sensmag.Ptr(), n_points*sizeof(double), cudaMemcpyHostToDevice, stream );
		if (pW->m_rx_external) {
			cudaMemcpyAsync( senspha_gpu, m_senspha.Ptr(), n_points*sizeof(double), cudaMemcpyHostToDevice, stream );
		} else {
			senspha_gpu = new double[1];
			double position[3] = {0., 0., 0.};
			senspha_gpu[0] = ( m_phase + (m_conjugate?-1.0:1.0) * GetPhase(position));
		}
		// dimensions are needed for the interpolation
		cudaMalloc( (void**)&map_dims, 3*sizeof(size_t) );
		cudaMemcpyAsync( map_dims, &(m_sensmag.Dims()[0]), 3*sizeof(size_t), cudaMemcpyHostToDevice, stream  );
		// clear the m_sensmag - cannot as it's local to the object
		// interpolate the maps for all spins and keep them in the dynamic buffer
		BufferDynCoils ();
		UpdateDynMapsGPU ();

	} else {
		// without interpolate, prepare the maps and copy to gpu 
		cudaMalloc( (void**)&sensmag_gpu, pW->TotalSpinNumber*sizeof(double) );
		double* sens_mag_tmp = new double[pW->TotalSpinNumber];
		double* sens_phase_tmp; 
		bool magnitude_only = true;
		if (pW->m_rx_external) {
			cudaMalloc( (void**)&senspha_gpu, pW->TotalSpinNumber*sizeof(double) );
			sens_phase_tmp = new double[pW->TotalSpinNumber];
			magnitude_only = false;
		}

		GetMapsAll (sens_mag_tmp, sens_phase_tmp, sample_values, sample_dims[0], magnitude_only);
		// wait for GetMaps to finish before copy
		cudaMemcpy( sensmag_gpu, sens_mag_tmp, pW->TotalSpinNumber*sizeof(double), cudaMemcpyHostToDevice );
		if (pW->m_rx_external) {
			cudaMemcpy( senspha_gpu, sens_phase_tmp, pW->TotalSpinNumber*sizeof(double), cudaMemcpyHostToDevice );
		} else {
			senspha_gpu = new double[1];
			double position[3] = {0., 0., 0.};
			senspha_gpu[0] = ( m_phase + (m_conjugate?-1.0:1.0) * GetPhase(position));
		}
		// dimensions are needed for the interpolation
		cudaMalloc( (void**)&map_dims, m_dim*sizeof(size_t) );
		cudaMemcpy( map_dims, sample_dims, m_dim*sizeof(size_t), cudaMemcpyHostToDevice  );
		if (pW->dynamic) 
			BufferDynCoils();
	}
	
}

/**********************************************************/
// AN-2022: helpers for the interpolation on GPU 
__device__ int get_index (int px, int py, int pz, size_t* m_dims) {
	return pz*m_dims[0]*m_dims[1]+py*m_dims[0]+px;
}

__device__ double UnwrapGPU(double checkwrap, bool magnitude){
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
// AN-2022: interpolate the coils sensitivity maps in case the 
// 			dynamic effects are on 
void Coil::UpdateDynMapsGPU (cudaStream_t stream/*=0*/) {

    World* pW = World::instance();

	int grid = (pW->TotalSpinNumber + block - 1) / block; 
	bool magnitude_only = true;
	if (pW->m_rx_external) {
		magnitude_only = false;
	} 
	// call the kernel 
	InterpolateSensitivityKernel <<< grid, block, 0, stream >>> (sensmag_gpu_dyn, senspha_gpu_dyn, magnitude_only,
		sensmag_gpu, senspha_gpu, pW->SpinPositions, 
		m_phase, m_scale, m_conjugate, map_dims, pW->TotalSpinNumber,
		(realtype)m_extent, m_points, m_dim);

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

/**********************************************************/
// AN-2022: receive spins on the given stream
void Coil::ReceiveGPU (long lADC, cudaStream_t stream) {
	
	World* pW = World::instance();
	int grid = (pW->TotalSpinNumber + block - 1) / block; 

	// update sensitivity maps corresponding to all spins at the new positions if there're dynamic effects
	if ((pW->dynamic) && (!pW->m_rx_ideal)) 
			UpdateDynMapsGPU(stream);

	// compute the product (sens_maps*solution) and convert it into double3 array
	if ((pW->dynamic) || m_interpolate) {
		// use memory for dynamic coil sensitivities
		Convert2double3 <<< grid, block, 0, stream >>> (d_sol_vec, pW->solution, 
			pW->TotalSpinNumber, (pW->m_rx_ideal), pW->m_rx_external,
		(sensmag_gpu_dyn), 
		(senspha_gpu_dyn), 
		(-(pW->phase)), m_phase);
	} else {
		// use memory for static coil sensitivities
		Convert2double3 <<< grid, block, 0, stream >>> (d_sol_vec, pW->solution, 
			pW->TotalSpinNumber, (pW->m_rx_ideal), pW->m_rx_external,
		(sensmag_gpu), 
		(senspha_gpu), 
		(-(pW->phase)), m_phase);
	}
	cudaStreamSynchronize(stream);
	// reduce the double3 array and write the ansqer into d_sol
    cub::DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_sol_vec, d_sol, 
		pW->TotalSpinNumber, sum_op, init, stream);
	// copy the computed signal to the host
	cudaMemcpyAsync (h_sol, d_sol, sizeof(double3), cudaMemcpyDeviceToHost, stream);
}	

/**********************************************************/
// AN-2022: kept outside of the Receive function to write the solution
//			once after all streams are finished receiving
void Coil::WriteSignal (long lADC_prev) {

	m_signal->Repo()->TP(lADC_prev) = World::instance()->time;
	long   pos   = m_signal->Repo()->Position(lADC_prev); 
	m_signal->Repo()->at(pos + 0) = h_sol[0].x;
	m_signal->Repo()->at(pos + 1) = h_sol[0].y;
	m_signal->Repo()->at(pos + 2) = h_sol[0].z;
	
}

/**********************************************************/
// AN-2022: allocate GPU memory for 'dynamic' coil maps
//			in case the dynamic effects are on
void Coil::BufferDynCoils () {

	World* pW = World::instance();
	cudaMalloc( (void**)&sensmag_gpu_dyn, pW->TotalSpinNumber*sizeof(double) );
	if (pW->m_rx_external) {
		cudaMalloc( (void**)&senspha_gpu_dyn, pW->TotalSpinNumber*sizeof(double) );
	} else {
		senspha_gpu_dyn = new double[1];
		senspha_gpu_dyn[0] = senspha_gpu[0];
	}

}

/**********************************************************/
// AN-2022: free all the GPU memory used for Receive
void Coil::DestroySolutionArraysGPU () {
	cudaFreeHost(h_sol);
	cudaFree(d_sol);
	cudaFree(d_sol_vec);
	cudaFree(d_temp_storage);
	World* pW = World::instance();
	if (pW->m_rx_external)
		cudaFree(senspha_gpu);
	if (pW->dynamic || m_interpolate) {
		cudaFree(sensmag_gpu_dyn);
		if (pW->m_rx_external)
			cudaFree(senspha_gpu_dyn);
	}
	if (!(pW->m_rx_ideal)) {
		cudaFree(map_dims);
		cudaFree(sensmag_gpu);
	}

}
#endif	// AN-2022*** 