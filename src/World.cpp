/** @file World.cpp
 *  @brief Implementation of JEMRIS World
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

#include "World.h"
#include "Model.h"
#include "SequenceTree.h"

World* World::m_instance = 0;

/***********************************************************/
World* World::instance() {

    if (m_instance == 0) {

        m_instance = new World();

	//MODIF
        fstream logAct("logActivation.txt",ios::in);
        if(logAct.is_open())	{
		logAct>>m_instance->logFile;
		logAct>>m_instance->logTrajectories;
        }
        else
        {
		m_instance->logFile 	    = 0;
		m_instance->logTrajectories = 0;
	    }
        logAct.close();
        m_instance->m_trajBegin         =  0;
        m_instance->m_trajSize          =  1;
        //MODIF***
        m_instance->time                =  0.0;
        m_instance->total_time          =  0.0;
        m_instance->phase               = -1.0;

#ifndef MODEL_ON_GPU // AN-2022
        m_instance->saveEvolFunPtr      = &Model::saveEvolution;
        m_instance->deltaB              =  0.0;
        m_instance->NonLinGradField     =  0.0;
        
        m_instance->m_myRank            = -1;
        m_instance->m_useLoadBalancing  = true;
        m_instance->m_no_processes      = 1;  /* default: serial jemris */
        m_instance->m_startSpin         = 0;
#else
        m_instance->saveEvolFunPtrGPU   = &Model::saveEvolutionGPU;
        // AN-2022: bools defining the coil array type
        m_instance->m_tx_ideal          = false;
        m_instance->m_rx_ideal          = false;
        m_instance->m_tx_external       = false;
        m_instance->m_rx_external       = false;
        m_instance->has_nonLinGrad       = false;
        m_instance->dynamic             = false;
        m_instance->NonLinGradField_GPU = nullptr;
        m_instance->NonLinGradField     = nullptr;

#endif 
// AN-2022***
        m_instance->GMAXoverB0          =  0.0;
        m_instance->LargestM0           =  0.0;
        m_instance->RandNoise           =  0.0;
        m_instance->saveEvolStepSize    =  0;
        m_instance->saveEvolFileName    =  "";
        m_instance->saveEvolOfstream    = NULL;
        m_instance->solverSuccess       = true;
        m_instance->m_noofspinprops     = 9;

        m_instance->pSeqTree            = NULL;
        m_instance->pAtom               = NULL;
        m_instance->pStaticAtom         = NULL;

        m_instance->m_slice             = 0;
        m_instance->m_set               = 0;
        m_instance->m_contrast          = 0;
        m_instance->m_average           = 0;
        m_instance->m_shot              = 0;
        m_instance->m_shotmax           = 1;
        m_instance->m_partition         = 0;
        m_instance->m_partitionmax      = 1;

        m_instance->m_seqSignature      = "";
    }

    XMLPlatformUtils::Initialize ();
    
    return m_instance;

}

/***********************************************************/
double World::ConcomitantField (double* G) {

	if (GMAXoverB0==0.0) 
		return 0.0;

	return ((0.5*GMAXoverB0)*(pow(G[0]*Values[ZC]-0.5*G[2]*Values[XC],2) + pow(G[1]*Values[ZC]-0.5*G[2]*Values[YC],2))) ;

}

/***********************************************************/
void World::SetNoOfSpinProps (int n) { 

	// valid also for multi pool sample
	if ( m_noofspincompartments > 1 ){

// AN-2022
#ifndef MODEL_ON_GPU
		int m_ncoprops =  (n - 4) / m_noofspincompartments;
		m_noofspinprops = n;
		Values = new double [n*m_ncoprops];
		for ( int i = 0; i < n; i++ )
			for ( int j = 0; j<m_ncoprops; j++ )
				Values [m_ncoprops*j+i] = 0.0;
#else
        cout << "GPU version was not implemeted to account for multiple compartments" << endl;
#endif

	}else{

		m_noofspinprops = n;

// AN-2022
#ifndef MODEL_ON_GPU
		Values = new double [n];
		for ( int i = 0; i < n; i++ )
			Values [i] = 0.0;
#else
        gpuErrchk( cudaMalloc((void**)&(Values), TotalSpinNumber*n*sizeof(realtype)) );
        gpuErrchk( cudaMalloc((void**)&(SpinPositions), 3*TotalSpinNumber*sizeof(realtype)) ); 
	    gpuErrchk( cudaMalloc((void**)&(deltaB), TotalSpinNumber*sizeof(realtype)) ); 
        gpuErrchk( cudaMalloc((void**)&(solution), 3*TotalSpinNumber*m_noofspincompartments*sizeof(realtype)) );
#endif
// AN-2022***

	}
}

void World::InitHelper (long size)  {
  if (size > 0)
    helper.resize(size);

}

int World::GetNoOfCompartments () {
	return m_noofspincompartments;
}

void World::SetNoOfCompartments (int n) {

	m_noofspincompartments = n;

// AN-2022
#ifndef MODEL_ON_GPU
    if (solution.size() < 3*n) {
    	solution.clear();
        solution.resize(m_noofspincompartments * 3);
    }
#endif
// AN-2022***

}
   

World::~World () { 
	
	m_instance=0; 

// AN-2022
#ifndef MODEL_ON_GPU	
	if (Values)
		delete Values; 
#else
    cudaFree(Values);
    cudaFree(solution);
    cudaFree(deltaB);
    cudaFree(SpinPositions);
    // AN_2022: host memory buffers to save the spin evolution        
    if (saveEvolStepSize != 0) {
        cudaFreeHost(h_solution);
        cudaFreeHost(h_values);
    }
    cudaFree((d_SeqVal_GPU));
#endif
// AN-2022***
}

/***********************************************************/
// AN-2022: initialize the nonlinear gradient fields 
void World::InitNonLinGradField() {

#ifndef MODEL_ON_GPU
    World::instance()->NonLinGradField = 0.0;
#else
    for (int i=0; i<TotalSpinNumber; i++) {
        NonLinGradField[i] = 0.;
    }
#endif
}

#ifdef MODEL_ON_GPU // AN-2022
/***********************************************************/
// AN-2022: helper function to convert a double precision array into a single precision array
realtype* double2floatArray(double* arr, unsigned n) {

    if (arr == nullptr) return nullptr;

    if (sizeof(realtype)==sizeof(double)) 
        return (realtype*)arr;
    else {
        realtype* ret = new realtype[n];
        for (int i = 0; i < n; i++) 
            ret[i] = (realtype)arr[i];
        return ret;
    }
}

/***********************************************************/
// AN-2022: helper to check the GPU error
void gpuAssert(cudaError_t code, const char *file, const int line, bool abort/*=true*/) {
// inline
    if (code != cudaSuccess) 
    {
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line); 
        if (abort) exit(code);
    }
}
// void gpuErrchk(cudaError_t ans) {gpuAssert((ans), __FILE__, __LINE__);};

/***********************************************************/
// AN-2022: allocate buffer host memory to save the evolution history
void World::BufferSaveEvolutionGPU () {

	gpuErrchk( 
        cudaMallocHost((void**)&h_solution, 3*(GetNoOfCompartments())*(TotalSpinNumber)*sizeof(realtype)) );
	gpuErrchk( 
        cudaMallocHost((void**)&h_values, (GetNoOfSpinProps())*(TotalSpinNumber)*sizeof(realtype)) );
}

#endif // AN-2022***