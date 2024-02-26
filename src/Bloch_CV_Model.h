/** @file Bloch_CV_Model.h
 *  @brief Implementation of JEMRIS Bloch_CV_Model.h
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

#ifndef BLOCH_CV_MODEL_H_
#define BLOCH_CV_MODEL_H_

#include "Model.h"
#include "config.h"

//CVODE includes:
#ifdef HAVE_CVODE_CVODE_H
    #include "cvode/cvode.h"
#endif
#ifdef HAVE_NVECTOR_NVECTOR_SERIAL_H
    #include "nvector/nvector_serial.h"
#endif
#ifdef HAVE_CVODE_CVODE_DIAG_H
    #include "cvode/cvode_diag.h"         /* prototypes for CVODE diagonal solver (required since CVODE 5.x) */
#endif

// AN-2022 includes fro CVode_5.7
#include <sundials/sundials_types.h>   /* definition of type double */
#include "sunnonlinsol/sunnonlinsol_fixedpoint.h" /* for the FP solver, which was called CV_FUNCTIONAL in CVODE < 3.0 */
// AN-2022***

#define NEQ   3                   // number of equations
// AN-2022: different tolerances for the single precision solver
#if defined(SUNDIALS_SINGLE_PRECISION)
#define RTOL  1e-5f     // ellips 1e-4      
#define ATOL1 5e-5f     // 1e-3 made it slower  
#define ATOL2 1e-4f
#define ATOL3 1e-4f        
#define BEPS  5e-5f  // ellips 1e-5    
#elif defined(SUNDIALS_DOUBLE_PRECISION)
#define RTOL  1e-6                // scalar relative tolerance
#define ATOL1 1e-8                // vector absolute tolerance components
#define ATOL2 1e-8
#define ATOL3 1e-8          
#define BEPS  1e-8         
#endif

//! Structure keeping the vectors for cvode
struct nvec {
    N_Vector y;      /**< CVODE vector */
    N_Vector abstol; /**< CVODE vector */
#ifdef MODEL_ON_GPU
    // AN-2022: use scalar abstol which is the minimum along the 3 dimensions
    realtype m_abstol;
#endif
};

/**
 * @brief Numerical solving of Bloch equations
 * As an application of the CVODE solver
 * by Lawrence Livermore National Laboratory - Livermore, CA
 * http://www.llnl.gov/CASC/sundials
 */

//! MR model solver using CVODE
class Bloch_CV_Model : public Model {

 public:

    /**
     * @brief Default destructor
     */
    virtual ~Bloch_CV_Model      () {
        CVodeFree(&m_cvode_mem);
        // AN-2022
        SUNNonlinSolFree(NLS);
#ifdef MODEL_ON_GPU
        for (int istream=0; istream<NoOfStreams; istream++) {
            cudaStreamDestroy(streams[istream]);
        }
#endif    
        // AN-2022***
    };

    /**
     * @brief Constructor
     */
    Bloch_CV_Model               ();


 protected:
#ifndef MODEL_ON_GPU
    /**
     * @brief Initialise solver
     *
     * Inistalise N_Vector and attach it to my world
     */
    virtual void InitSolver      ();

    /**
     *  see Model::Calculate_onGPU()
     */
    virtual bool Calculate       (double next_tStop);
        
    /**
     * @brief Free solver
     *
     * Release the N_Vector
     */
    virtual void FreeSolver      ();

#else
    // AN-2022
    virtual bool CalculateGPU       (double next_tStop, cudaStream_t stream);
    virtual void InitSolverGPU (cudaStream_t stream, bool alloc_nvector_gpu);
    /**
     * @brief Free solver
     *
     * Release the N_Vector
     */
    virtual void FreeSolverGPU      ();
#endif

    /**
     * @brief Summery output
     *
     * More elaborate description here please
     */
    void         PrintFinalStats ();

 private:

    // CVODE related
    void*  m_cvode_mem;	 /**< @brief pointer to cvode malloc */
#ifndef MODEL_ON_GPU
    double m_tpoint;	 /**< @brief current time point */
    double m_reltol;	 /**< @brief relative error tolerance for CVODE */
#else
    // AN-2022: changed to realtype to allow single-precision
    realtype m_tpoint;	 /**< @brief current time point */
    realtype m_reltol;	 /**< @brief relative error tolerance for CVODE */
#endif
    SUNNonlinearSolver NLS; // nonlinear solver instead of the Newton's method used in jemris-2-8-3
};

#endif /*BLOCH_CV_MODEL_H_*/
