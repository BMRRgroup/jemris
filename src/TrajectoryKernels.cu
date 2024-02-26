
#include "TrajectoryKernels.cuh"
#include <cuda_runtime.h>

#include <unistd.h>

/***********************************************************/
void Call_ChangePosKernel (realtype trans_x, realtype trans_y, realtype trans_z, realtype rot_x, realtype rot_y, 
			realtype rot_z, realtype cosx, realtype sinx, realtype cosy, realtype siny, 
			realtype cosz, realtype sinz, 
			realtype rot_origin_x, realtype rot_origin_y, realtype rot_origin_z) {

    World* pW = World::instance();
	int N_spins = pW->TotalSpinNumber;
    // cout << "Call_ChangePosKernel N_spins = " << N_spins << endl;
	int block = 512;
	int grid = (N_spins + block - 1) / block;  
	cudaStream_t currStream = pW->currStream;
    // realtype* params;
    // cudaMalloc ( (void**)&params, 15*sizeof(realtype)  );
    // cudaMemcpyAsync ( params, &trans_x, sizeof(realtype), cudaMemcpyHostToDevice, currStream );
    // cudaMemcpyAsync ( params, &trans_x, sizeof(realtype), cudaMemcpyHostToDevice, currStream );
    // printf( "Ty =  %f    rot_origin_y = %f \n", trans_y, rot_origin_y );
	ChangePosKernel <<< grid, block, 0, currStream >>> (N_spins, pW->GetNoOfSpinProps(), pW->Values, 
            pW->SpinPositions, 
			trans_x, trans_y, trans_z, rot_x, rot_y, rot_z, cosx, sinx, cosy, siny, cosz, sinz, 
			rot_origin_x, rot_origin_y, rot_origin_z);
}

/***********************************************************/
__global__ void ChangePosKernel ( unsigned N_spins_total, int N_SpinProps, realtype* s_vals,
            realtype* positions, 
			realtype trans_x, realtype trans_y, realtype trans_z, realtype rot_x, realtype rot_y, 
			realtype rot_z, realtype cosx, realtype sinx, realtype cosy, realtype siny, 
			realtype cosz, realtype sinz, 
			realtype rot_origin_x, realtype rot_origin_y, realtype rot_origin_z) { 

	unsigned spin_idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (spin_idx < N_spins_total)	{ 
		// realtype cosx, sinx; 
		realtype old_x=s_vals[N_SpinProps*spin_idx+0];
		realtype old_y=s_vals[N_SpinProps*spin_idx+1];
		realtype old_z=s_vals[N_SpinProps*spin_idx+2];
		realtype x,y,z;
        // if (spin_idx == 10) {
        //     printf( "Spin position %f %f %f\n", old_x, old_y, old_z );
        // }

		// transform to rotation point:
		x = old_x - rot_origin_x;
		y = old_y - rot_origin_y;
		z = old_z - rot_origin_z;

		// rotation around x-axis:
		if (rot_x != 0) {
			// cosx = cos(rot_x);
			// sinx = sin(rot_x);
			old_y = y;
			old_z = z;
			y = old_y * cosx - old_z * sinx;
			z = old_y * sinx + old_z * cosx;
		}

		// rotation around y-axis:
		if (rot_y != 0) {
			// cosx = cos(rot_y);
			// sinx = sin(rot_y);
			old_x = x;
			old_z = z;
			x = old_z * siny + old_x * cosy;
			z = old_z * cosy - old_x * siny;
		}

		// rotation around z-axis:
		if (rot_z != 0) {
			// cosx = cos(rot_z);
			// sinx = sin(rot_z);
			old_x = x;
			old_y = y;
			x = old_x * cosz - old_y * sinz;
			y = old_x * sinz + old_y * cosz;
		}

		// translation (from rotation origin + motion translation):
		positions[3*spin_idx+0] = x + rot_origin_x + trans_x;
		positions[3*spin_idx+1] = y + rot_origin_y + trans_y;
		positions[3*spin_idx+2] = z + rot_origin_z + trans_z;
        // if (spin_idx == 10) {
        //     printf( "After Spin position %f %f %f\n", positions[3*spin_idx+0], positions[3*spin_idx+1], positions[3*spin_idx+2] );
        // }
	}
}
