
#include "TrajectoryMotion.h"
// #include "SequenceTree.h"
// #include "ConcatSequence.h"

// AN-2022: added CUDA kernels
// #include <cuda_runtime.h>

void Call_ChangePosKernel (realtype trans_x, realtype trans_y, realtype trans_z, realtype rot_x, realtype rot_y, 
			realtype rot_z, realtype cosx, realtype sinx, realtype cosy, realtype siny, 
			realtype cosz, realtype sinz, 
			realtype rot_origin_x, realtype rot_origin_y, realtype rot_origin_z);

/***********************************************************/
__global__ void ChangePosKernel ( unsigned N_spins_total, int N_SpinProps, realtype* s_vals, realtype* positions,
			realtype trans_x, realtype trans_y, realtype trans_z, realtype rot_x, realtype rot_y, 
			realtype rot_z, realtype cosx, realtype sinx, realtype cosy, realtype siny, 
			realtype cosz, realtype sinz, 
			realtype rot_origin_x, realtype rot_origin_y, realtype rot_origin_z);