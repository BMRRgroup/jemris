/** @file Coil.h
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

#ifndef COIL_H_
#define COIL_H_

#include "Prototype.h"
#include "Signal.h"
#include "Declarations.h"
#include "NDData.h"

#ifdef MODEL_ON_GPU // AN-2022
#include <cuda_runtime.h>

/** AN-2022
* @brief sum operator of the two double3 variables
*/
struct CustomSum {
    __device__
    double3 operator()(const double3& a, const double3& b) const {
        // return a+b;
      double3 r;
      r.x = a.x + b.x;
      r.y = a.y + b.y;
      r.z = a.z + b.z;
      return r;
    }
};
#endif  // AN-2022***

/**
 * @brief Base class of coil objects
 */

class Coil : public Prototype {

 public:

    /**
     * @brief Destructor
     */
    virtual ~Coil          ();

    /**
     * @brief Get the B1+ magnitude at point (x,y,z) of the current spin
     *
	 * @return          Sensitivity with respect to spin in World
     */
    double  GetSensitivity (const double time=0) ;

    /**
     * @brief Get the B1+ phase at point (x,y,z) of the current spin
     *
	 * @return          Phase with respect to spin in World
     */
    double  GetPhase (const double time=0) ;

    /**
     * @brief Interpolate the sensitivity at point (x,y,z)
     *
	 * @return          Interpolated Sensitivity
     */
    double InterpolateSensitivity (const double* position, bool magnitude=true);

    /**
     * @brief Get the B1+ magnitude at point (x,y,z)
     *
     * This method must be implemented by every derived coil.
     *
     * @param position  At position.
     * @return          Sensitivity with respect to spin in World.
     */
    virtual double  GetSensitivity (const double* position) = 0;

    /**
     * @brief Get the B1+ phase at point (x,y,z)
     *
     * This method may be implemented by every derived coil. Otherwise phase is zero.
     *
     * Important: the phase of Coils needs to be implemented with unit radians!
     * (In contrast to the phase of RF pulses which has units degrees.)
     *
     * @param position  At position.
     * @return          B1+ phase with respect to spin in World.
     */
    virtual double  GetPhase (const double* position) {return 0.0;};

    /**
     * @brief Initialize my signal repository
     *
     * @param lADCs     Number of ADCs
     */
    void    InitSignal     (long lADCs);

    /**
     * @brief Receive signal from World
     *
     * @param lADC      Receive the signal for this particular ADC
     */
    void    Receive        (long lADC);

    /**
     * @brief Transmit signal.
     */
    void    Transmit        ();

    /**
     * @brief Get the received signal of this coil
     *
     * @return          The received signal
     */
    Signal* GetSignal       () { return m_signal; }

    /**
     *  See Prototype::Clone
     */
    Coil* Clone() const = 0;

    /**
     * @brief Dump sensitivity map on the XML defined grid
     */
	void GridMap     ();

    /**
     * @brief Map magnitudes
     *
     * @return Magnitudes
     */
	double* MagnitudeMap     ();

    /**
     * @brief Map phases
     *
     * @return phases
     */
	double* PhaseMap     ();

    /**
     * @brief Prepare coil with given attributes.
     *
     * @param mode Sets the preparation mode, one of enum PrepareMode {PREP_INIT,PREP_VERBOSE,PREP_UPDATE}.
     */
    virtual bool Prepare  (const PrepareMode mode);

    /**
     * @brief Initialize this prototype.
     *
     * The first step after cloning!
     * The method sets the Name of the Prototype, and pointers to
     * the referring node and the (unique) SequenceTree.
     *
     * @param node The DOMNode referring to this prototype.
     */
    void Initialize  (DOMNode* node );

	int GetPoints ();

	int GetExtent ();

	unsigned GetNDim ();

	double GetNorm (){return m_norm;};

#ifdef MODEL_ON_GPU 
// AN-2022
    
    /**
     * @brief calculate coil sensitivity maps for all spins and save them in the provided pointers sens_all, phase_all
     */
    void GetMapsAll (double* sens_magn_all, double* sens_phase_all, double* sample_values, 
		int N_props, bool magnitude_only);   

    /**
     * @brief precalculate coil sensitivities for all spins, allocate memory on GPU and tranfer the maps
     */
    void InitCoilSensGPU (size_t* sample_dims, double* sample_vals, cudaStream_t stream);

    /**
     * @brief initialize GPU memory for solution vectors and reduction operator 
     */
    void InitSolutionArraysGPU ();

    /**
     * @brief in case of dynamic effects, allocate extra GPU memory for "dynamic" coil sensitivities
     *        in case of interpolated coil maps, use dynamic buffer to store interpolated maps
     */
    void BufferDynCoils ();
    
    /**
     * @brief interpolate the coil sensitivity maps on GPU
     */
    void UpdateDynMapsGPU (cudaStream_t stream=0);   
    
    /**
     * @brief receive on all streams async-ly and write signals to repository at once
                each stream is assigned to a receive channel in multi-channel
     */
    void ReceiveGPU (long lADC, cudaStream_t stream);

    /**
     * @brief write bulk magnetization into signal repository
     */
    void WriteSignal (long lADC);

    /**
     * @brief clean up the GPU memory for solution vectors and reduction operator 
     */
    void DestroySolutionArraysGPU ();

#endif  // AN-2022***

 protected:

    /**
     * Constructor
     */
    Coil() {};

    double		    m_position[3];	/**< Center location   */
    Signal*		    m_signal;    	/**< Signal repository */
    unsigned		m_mode;      	/**< My mode (RX/TX)      */
    double			m_azimuth; 		/**< Change of coordinate system: azimuth angle*/
    double			m_polar;   		/**< Change of coordinate system: polar angle*/
    double			m_scale;   		/**< Scaling factor for sensitivities */
    double			m_norm;   		/**< Normalization factor for sensitivities */
    double			m_phase;   		/**< Constant phase shift */
    bool            m_interpolate;  /**< Whether to precompute sensitivities in an array */
    bool			m_complex;		/**< True, if sensitivity map is complex (non-zero phase entries).*/
    bool		    m_conjugate;	/**< Complex conjugate the sensitivites, if true.*/
    unsigned		m_dim;     		/**< Dimensions (2D or 3D) of the array*/
    double			m_extent;  		/**< Array extend of support region [mm] */
    int				m_points;  		/**< Sampling points of the array */

    NDData<double>  m_sensmag;
    NDData<double>  m_senspha;

    double Unwrap(double diff,bool magnitude); /**< helper function to check for phase wraps in interpolation of phase maps. */

// AN-2022: members needed for the GPU computations
#ifdef MODEL_ON_GPU
    double3*        h_sol;          /**< temporary storage for the signal on host   */
    double3*        d_sol;          /**< temporary storage for the signal on device   */
    double3*        d_sol_vec;      /**< temporary storage for the solution on device   */
    size_t          temp_storage_bytes;     /**< storage buffer size for the reception reduction operator   */
    void*           d_temp_storage;        /**< temporary storage buffer for the reception reduction operator   */
    double3         init;           /**< initial value for the reception reduction operator   */
    CustomSum       sum_op;         /**< reception reduction operator, separate for all coil channels  */
    double*         sensmag_gpu;        /**< coil sensitivity magnitude on GPU   */
    double*         senspha_gpu;        /**< coil sensitivity phase on GPU   */
    double*         sensmag_gpu_dyn;    /**< interpolated coil sensitivity magnitude and phase on GPU   */
    double*         senspha_gpu_dyn;    /**< in case dynamic effects are on   */   
    size_t*         map_dims;       /**< dimensions of the provided coil sensitivity maps   */
#endif  // AN-2022***

};

#endif /*COIL_H_*/
