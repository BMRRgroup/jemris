/** @file Model.h
 *  @brief Implementation of JEMRIS Model
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

#ifndef MODEL_H_
#define MODEL_H_

#include <math.h>

#include "World.h"
#include "Sample.h"
#include "Module.h"
#include "AtomicSequence.h"
#include "ConcatSequence.h"
#include "Container.h"
#include "ContainerSequence.h"

// AN-2022
#ifdef MODEL_ON_GPU
#include "cuda_runtime.h"
#include <sundials/sundials_types.h>   /* definition of type double */
#endif // AN-2022***
using namespace std;

//class declarations
class CoilArray;


//! Base class for MR model solver

class Model {

 public:

	/**
	 * @brief Constructor
	 */
	Model();

	/**
	 * @brief Default Destructor
	 */
	virtual ~Model() {	
#ifdef MODEL_ON_GPU	
		cudaFree(dM);
#endif
	};

	/**
	 * @brief Prepare for simulations.
	 *
	 * @param pRxCoilArray    Coil array receiving and storing the signal processed.
	 * @param pTxCoilArray    Coil array transmitting the sequence to be processed.
	 * @param pConcatSequence Complete MR sequence.
	 * @param pSample         Sample.
	 */
	void Prepare(CoilArray* pRxCoilArray, CoilArray* pTxCoilArray, ConcatSequence* pConcatSequence, Sample* pSample);

	/**
 	 *  @brief Set MR sequence.
	 */
	inline void SetSequence(ConcatSequence* pConcatSequence){m_concat_sequence  = pConcatSequence;};
	
	/**
 	 *  @brief Solve differential equations.
	 */
	void Solve();
	
#ifndef MODEL_ON_GPU // AN-2022
	/**
	 * @brief Save time evolution to disk.
	 *
 	 * Static function to write the time-evolution of the magnetisation
 	 * for every spin to binary file.
	 */
	static void saveEvolution (long index, bool close_files) ;

#elif MODEL_ON_GPU == 1
	void SolveGPU();
    static void saveEvolutionGPU (long index, bool close_files, cudaStream_t stream) ;
#endif
	// AN-2022***

	void SetDumpProgress(bool val) { m_do_dump_progress = val; };

 protected:

#ifndef MODEL_ON_GPU

	/**
	 * @brief Initialise solver.
	 */
    virtual void InitSolver() = 0;

	/**
	 * @brief Free solver.
	 */
    virtual void FreeSolver() = 0;

	/**
	 * @brief Calculate specific solution
	 *
	 * Calculate specific solution in an atomic sequence
	 * All setting for the computation are in m_World
	 *
	 * @param next_tStop Next time stop
	 *
	 * @return       Result
	 */
	virtual bool Calculate(double next_tStop) = 0;

	/**
 	 * Run through the sequence tree and
	 * execute Calculate for each atom
	 *
	 * @param dTimeShift  The time shift with respect to the whole sequence
	 * @param lIndexShift The ADC number shift with respect to the whole sequence
	 * @param module      The current sequence module to be simulated
	 */
	void RunSequenceTree (double& dTimeShift, long& lIndexShift, Module* module);
	
#else	// AN-2022
	virtual bool 	CalculateGPU(double next_tStop, cudaStream_t stream) = 0;
	virtual void 	InitSolverGPU(cudaStream_t stream, bool alloc_nvector_gpu) = 0;
	void 			RunSequenceTreeGPU (double& dTimeShift, long& lIndexShift, Module* module);
	/**
	 * @brief Free solver.
	 */
    virtual void FreeSolverGPU() = 0;
#endif
	// AN-2022***

#ifdef MODEL_ON_GPU		// AN-2022
	realtype* 		dM;  				/** @brief Back-up for the solution on GPU */
	cudaStream_t*	streams;			/** @brief Non-default streams for asynchorous run on GPU  */
	int				streamSize[NoOfStreams];			/** @brief If the ensemble split between streams, the size of a chunk */
#endif // AN-2022***

    World*          m_world;	        /**< @brief Simulation world                             */
    CoilArray*      m_rx_coil_array;    /**< @brief Receive coil array                          */
    CoilArray*      m_tx_coil_array;    /**< @brief Transmit coil array                           */
    ConcatSequence* m_concat_sequence;  /**< @brief Top node of the sequence tree for simulation */
    Sample*         m_sample;           /**< @brief Sample                                       */
    realtype	    m_accuracy_factor;  /**< @brief increase accuracy by this factor in case of numerical problems */

    bool            m_do_dump_progress; /**< @brief If true, percentage progress during Solve() is written to .jemris_progress.out */

 private:

    bool            m_aux; //for debugging
#ifndef MODEL_ON_GPU	// AN-2022
    /**
     * updates process counter file
     */
    void UpdateProcessCounter (const long lSpin);
#else
	/**
     * updates process counter file and progressbar
     */
    void UpdateProcessCounterGPU (const long lADC);
#endif	// AN-2022***

    /**
     * dumps restart information(serial jemris.)
     */
    void DumpRestartInfo(long lSpin);



};

#endif /*MODEL_H_*/

