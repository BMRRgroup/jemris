/** @file CoilArray.h
 *  @brief Implementation of JEMRIS CoilArray
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

#ifndef COILARRAY_H_
#define COILARRAY_H_

#include "XMLIO.h"

#include "Signal.h"
#include "CoilPrototypeFactory.h"

#ifdef MODEL_ON_GPU
// AN-2022
#include <cuda_runtime.h>
#endif

class Coil;

/**
 *  @brief Coil configuration and sensitivities
 */
class CoilArray  {

 public:

    /**
     * @brief Default constructor
     *
     * Empty constructor will create a single channel ideal coil with flat sensitivity map.
     */
    CoilArray();

    /**
     * @brief Initialize the array and its elements
     *
     * @param uri Configuration file. (Assigned to Simulation in simu.xml)
     */
    void Initialize  (string uri);

    /**
     * @brief Populate coil array
     *        Run over coil array configuration tree and Populate the array
     */
    // AN-2022: added variable defining the coil type
    unsigned int  Populate (string* coil_name=nullptr);

    /**
     * @brief Default destructor
     */
    virtual ~CoilArray     ();

    /**
     * @brief Clone a coil
     *
     * @param  ptr  Pointer to myself
     * @param  node DOMNode with coil configuation
     *
     * @return      Created coil
     */
    static unsigned int CreateCoil (void* ptr,DOMNode* node);

    /**
     * @brief Run over XML tree and return nodes
     *
     * @return All nodes in the tree.
     *
     */
    DOMNode*        RunTree (DOMNode* node, void* ptr, unsigned int (*fun) (void*, DOMNode*));


    /**
     * @brief Get the number of channels
     *
     * @return The number of my channels
     */
    inline unsigned int GetSize   () { return m_coils.size(); };

    /**
     * @brief Create the signal structures of given size for all my coils.
     *
     * @param lADCs Number of ADC events.
     */
    void InitializeSignals (long lADCs);

    /**
     * @brief Recieve a signal from the World singleton with my coils for a given event.
     *
     * @param lADC position of this ADC event within all ADC events.
     */
    void Receive           (long lADC);

    /**
     * @brief Dump all signals
     * Dump the signals from all coils to discrete files.
     */
	IO::Status DumpSignals       (string tmp_prefix = "", bool normalize = true);

    /**
     * @brief Dump all signals
     * Dump the signals from all coils to ISMRMRD file.
     */
	IO::Status DumpSignalsISMRMRD       (string prefix = "_ismrmrd", bool normalize = true);

    /**
     * @brief Dump all sensitivities
     * Dump the sensitivities from all coils to discrete files.
     */
	IO::Status DumpSensMaps      (bool verbose = false);


    /**
     * @brief Set signal file-name prefix
     * Set the prefix string for signal binary filenames.
     * @param val the prefix
     */
    void SetSignalPrefix      (string val) {m_signal_prefix = val;};

    string GetSignalPrefix      () {return m_signal_prefix;};

    void SetSenMaplPrefix      (string val) {m_senmap_prefix = val;};

    string GetSenMaplPrefix      () {return m_senmap_prefix;};

    /**
     * @brief Set SensMap output directory
     * Directory the SensMap is saved to
     * @param dir the directory (it is assumed that it exists)
     */
    void SetSenMapOutputDir(string dir) { m_senmap_output_dir = dir; };

    string GetSenMapOutputDir      () {return m_senmap_output_dir;};

    /**
     * @brief Set signals output directory
     * Directory the signal binary is saved to
     * @param dir the directory (it is assumed that it exists)
     */
    void SetSignalOutputDir(string dir) { m_signal_output_dir = dir; };

    string GetSignalOutputDir      () {return m_signal_output_dir;};

    /**
     * @brief Get a particular coil
     *
     * @param  channel The number of the particular channel.
     * @return The requested coil.
     */
    Coil* GetCoil          (unsigned channel);

    /**
     * @brief Prepare my coils
     *
     * @param  mode Prepare mode
     *
     * @return      Success
     */
    bool Prepare (const PrepareMode mode);

    /**
     * @brief
     */
    void setMode (unsigned short mode) { m_mode = mode; }

    /**
     * @brief reads restart signal.
     */
    int ReadRestartSignal();

#ifdef MODEL_ON_GPU   
// AN-2022

    /**
     * @brief pre-calculate coil sensitivities for all spins, allocate memory on GPU and tranfer the maps
     */
    void InitCoilSensGPU (size_t* sample_dims, double* sample_vals, cudaStream_t* streams);

    /**
     * @brief initialize GPU memory for solution vectors and reduction operator 
     */
    void InitSolutionArraysGPU ();

    /**
     * @brief in case of dynamic effects, allocate extra GPU memory for "dynamic" coil sensitivities
     */
    void BufferDynCoils ();
    
    /**
     * @brief receive on all streams async-ly and write signals to repository at once
                each stream is assigned to a receive channel in multi-channel
     */
    void ReceiveGPU (long lADC, int iter_stream, int SpinOffset, 
        int StreamSize, cudaStream_t* streams);

    /**
     * @brief write bulk magnetization into signal repository
     */
    void WriteSignal (long lADC);

    /**
     * @brief clean up the GPU memory for solution vectors and reduction operator 
     */
    void DestroySolutionArraysGPU ();

#endif

 private:

    vector<Coil*>         m_coils;         /**< @brief My coils         */
    double                m_radius;        /**< @brief My radius        */
    unsigned short        m_mode;          /**< @brief My mode (RX/TX)  */
    string	              m_signal_prefix; /**< @brief prefix string to signal binary filenames */
    string	              m_senmap_prefix; /**< @brief prefix string to sensitivity map filenames */
    string	              m_signal_output_dir;  /**< @brief string to signal directory            */
    string	              m_senmap_output_dir;  /**< @brief string to sensitivity map directory   */

    CoilPrototypeFactory* m_cpf;           /**< @brief Coil factory    */
    DOMDocument*          m_dom_doc;       /**< @brief DOM document containing configuration */
    Parameters*           m_params;        /**< @brief My parameters   */
    XMLIO*                m_xio;           /**< @brief My XML IO module   */

    bool                  m_normalized=false; /**< @brief WIP parameter to check if data was already normalized by DumpSignals(). */

};

#endif /*COILARRAY_H_*/
