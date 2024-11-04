/** @file Bloch_CV_Model.cpp
 *  @brief Implementation of JEMRIS Bloch_CV_Model
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

/* 
AN-2022: updated to support the newer CVode version 5.7;
         included components for the numerical model on GPU
*/ 

#include "Bloch_CV_Model.h"
#include "DynamicVariables.h"
//MODIF
#include <iostream>
#include <fstream>
//MODIF***

// AN-2022
#ifdef MODEL_ON_GPU
#include "Declarations.h"
// #include <cuda_runtime.h>
#include "CudaKernels.cuh"
#include <nvector/nvector_cuda.h> 
#endif
// AN-2022***

#ifndef MODEL_ON_GPU // AN-2022
/**********************************************************/
inline static int bloch (realtype rt, N_Vector y, N_Vector ydot, void *pWorld) {

    World* pW = (World*) pWorld;
    DynamicVariables* dv = DynamicVariables::instance();
	double t = (double) rt;

    if (t < 0.0 || t > pW->pAtom->GetDuration()) {
    	// this case can happen when searching for step size; in this area no solution is needed
    	// -> set ydot to any defined value.
    	NV_Ith_S(ydot,AMPL) = 0;
    	NV_Ith_S(ydot,PHASE) = 0;
    	NV_Ith_S(ydot,ZC) = 0;
    	return 0;
    }

	double time = pW->total_time+t;
    double Mxy, phi, Mz; /*cylndrical components of mangetization*/
    double s, c, Mx, My, Mx_dot, My_dot, Mz_dot;
    Mz_dot = 0.0;

    //sample variables:
    double r1 = pW->Values[R1];
    double r2 = pW->Values[R2];
    double m0 = pW->Values[M0];
    double position[3];
    position[0] = pW->Values[XC];position[1]=pW->Values[YC];position[2]=pW->Values[ZC];
    double DeltaB = pW->deltaB;

    // update sample variables if they are dynamic:
    dv->m_Diffusion->GetValue(time, position);
//MODIF
    if(pW->logFile)
    {
        //cout<<"Spin number: "<<pW->SpinNumber<<endl;
        if((pW->SpinNumber==0) && (time==0))  {
            fstream log0("FLOW.log",ios::out|ios::trunc);
            log0.close();
            }
        fstream log1("FLOW.log",ios::out|ios::app);
        log1<<"t "<<time<<"  "<<"spin"<<pW->SpinNumber<<" position: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<" Activation: "<<dv->m_Flow->spinActivation(pW->SpinNumber)<<endl;
        log1.close();
    }
    long trajNumber=pW->getTrajBegin()+pW->SpinNumber;
    dv->m_Flow->GetValue(time, position, trajNumber);
    if(pW->logFile)
    {
        fstream log2("FLOW.log",ios::out|ios::app);
        log2<<"t "<<time<<"  "<<"spin"<<pW->SpinNumber<<" position: "<<position[0]<<" "<<position[1]<<" "<<position[2]<<" Activation: "<<dv->m_Flow->spinActivation(pW->SpinNumber)<<endl<<endl;
        log2.close();
    }
    if(pW->logTrajectories)
    {
        if((pW->SpinNumber==0) && (time==0))  {
            fstream logTraj0("trajectories.log",ios::out|ios::trunc);
            logTraj0.close();
            }
        if(dv->m_Flow->spinActivation(pW->SpinNumber))  {
            fstream logTraj("trajectories.log",ios::out|ios::app);
            if(time==0)  logTraj<<endl;
            logTraj<<pW->SpinNumber<<" "<<time<<" "<<position[0]<<" "<<position[1]<<" "<<position[2]<<endl;
            logTraj.close();
            }
    }
//MODIF***
//Mod
    dv->m_Respiration->GetValue(time, position);
//
    dv->m_Motion->GetValue(time, position);
    dv->m_T2prime->GetValue(time, &DeltaB);
    dv->m_R1->GetValue(time, &r1);
    dv->m_R2->GetValue(time, &r2);
    dv->m_M0->GetValue(time, &m0);

//MODIF
    //check spin active: if not, set transv. magnetization to 0
    if (! dv->m_Flow->spinActivation(pW->SpinNumber)) {
    	NV_Ith_S(y,AMPL) = 0;
    	NV_Ith_S(ydot,PHASE) = 0;
    	NV_Ith_S(ydot,ZC) = 0;
    	return 0;
    }
//MODIF***

    //get current B-field values from the sequence
    double  d_SeqVal[5]={0.0,0.0,0.0,0.0,0.0};									// [B1magn,B1phase,Gx,Gy,Gz]
    pW->pAtom->GetValue( d_SeqVal, t );        								    // calculates also pW->NonLinGradField
    if (pW->pStaticAtom != NULL) pW->pStaticAtom->GetValue( d_SeqVal, time );	// calculates static offsets
    pW->pAtom->GetValueLingeringEddyCurrents(d_SeqVal,t);					    // calculates lingering eddy currents

    double Bx, By, Bz;

    //Transverse Components: RF field
    Bx = d_SeqVal[RF_AMP]*cos(d_SeqVal[RF_PHS]);
    By = d_SeqVal[RF_AMP]*sin(d_SeqVal[RF_PHS]);

    //Longitudinal component: Gradient field and off-resonance contributions
    Bz = position[0]*d_SeqVal[GRAD_X]+ position[1]*d_SeqVal[GRAD_Y]+ position[2]*d_SeqVal[GRAD_Z]
         + DeltaB + pW->ConcomitantField(&d_SeqVal[GRAD_X]) + pW->NonLinGradField;

    //NV_Ith_S is the solution magn. vector with components AMPL,PHASE,ZC
    // check if double precision is still enough for sin/cos:
    if (fabs(NV_Ith_S(y,PHASE))>1e11 ) {
        //important: restrict phase to [0, 2*PI]
        NV_Ith_S(y,PHASE) = fmod (NV_Ith_S(y,PHASE), TWOPI);
    }

    Mxy = NV_Ith_S(y,AMPL);
    phi = NV_Ith_S(y,PHASE);
    Mz  = NV_Ith_S(y,ZC);

    // cartesian components of transverse magnetization
    c = cos(phi);
    s = sin(phi);
    Mx = c*Mxy;
    My = s*Mxy;

    // bloch equations
    Mx_dot =   Bz*My - By*Mz - r2*Mx;
    My_dot = - Bz*Mx + Bx*Mz - r2*My;
    Mz_dot =   By*Mx - Bx*My ;

    // derivatives in cylindrical coordinates
    NV_Ith_S(ydot,AMPL)  =  c*Mx_dot + s*My_dot;
    NV_Ith_S(ydot,PHASE) = (c*My_dot - s*Mx_dot) / (Mxy>ATOL1?Mxy:ATOL1); //avoid division by zero

    //longitudinal relaxation
    Mz_dot +=  r1*(m0 - Mz);
    NV_Ith_S(ydot,ZC) = Mz_dot;

    return 0;

}

/**********************************************************/
Bloch_CV_Model::Bloch_CV_Model     () : m_tpoint(0) {

    m_world->solverSettings = new nvec;
/*    for (int i=0;i<OPT_SIZE;i++) {m_iopt[i]=0; m_ropt[i]=0.0;}
    m_iopt[MXSTEP] = 100000;
    m_ropt[HMAX]   = 100000.0;// the maximum stepsize in msec of the integrator*/
    m_reltol       = RTOL;

    // create cvode memory pointer; no mallocs done yet.
    m_cvode_mem = CVodeCreate (CV_ADAMS);


    // cvode allocate memory.
    // do CVodeMalloc with dummy values y0,abstol once here;
    // -> CVodeReInit can later be used
    N_Vector y0, abstol;
    y0		= N_VNew_Serial(NEQ);
    abstol	= N_VNew_Serial(NEQ);
    ((nvec*) (m_world->solverSettings))->abstol = N_VNew_Serial(NEQ);

    NV_Ith_S(y0, AMPL)  = 0;
    NV_Ith_S(y0, PHASE) = 0;
    NV_Ith_S(y0, ZC)    = 0;

   //MODIF  Define CVODE errors with external file if exists
    /*
    NV_Ith_S(abstol, AMPL)  = ATOL1;
    NV_Ith_S(abstol, PHASE) = ATOL2;
    NV_Ith_S(abstol, ZC)    = ATOL3;  */

    ifstream CVODEfile;
    long int MXSTEP;

    CVODEfile.open("CVODEerr.dat", ifstream::in);
    if (CVODEfile.is_open()) {
        CVODEfile>>m_reltol>>NV_Ith_S(abstol, AMPL)>>NV_Ith_S(abstol, PHASE)>>NV_Ith_S(abstol, ZC)>>MXSTEP;
        cout<<"CVODE file open"<<endl;
    }
    else  {
        m_reltol       = RTOL;
        NV_Ith_S(abstol, AMPL)  = ATOL1;
        NV_Ith_S(abstol, PHASE) = ATOL2;
        NV_Ith_S(abstol, ZC)    = ATOL3;
        MXSTEP=100000;
    }
    //MODIF***

    if(CVodeSetUserData(m_cvode_mem, (void *) m_world) !=CV_SUCCESS) {
    	cout << "CVode function data could not be set. Panic!" << endl;exit (-1);
    }
    if(CVodeInit(m_cvode_mem,bloch,0,y0) != CV_SUCCESS ) {
    	cout << "CVodeInit failed! aborting..." << endl;exit (-1);
    }
    if(CVodeSVtolerances(m_cvode_mem, m_reltol, abstol)!= CV_SUCCESS){
    	cout << "CVodeSVtolerances failed! aborting..." << endl;exit (-1);
    }

    // AN-2022
    int                 mxiter  = 20;
    int                 maa     = 0;           // acceleration vectors
    double              damping = RCONST(1.0); 
    NLS = SUNNonlinSol_FixedPoint(y0, maa); // y0 - a NVECTOR template for
	// cloning vectors needed within the solver
    if((void *)NLS == NULL) {
	cout << "SUNNonlinSol_FixedPoint initialization failed!" << endl; exit(-1);
    }
    if(SUNNonlinSolSetMaxIters(NLS, mxiter) != CV_SUCCESS) {
	cout << "Setting SUNNonlinSol max iterations failed!" << endl; exit(-1);
    }
    if(SUNNonlinSolSetDamping_FixedPoint(NLS, damping) != CV_SUCCESS) {
	cout << "Setting SUNNonlinSol damping coefficient failed!" << endl; exit(-1);
    }
    if (CVodeSetNonlinearSolver(m_cvode_mem, NLS) != CV_SUCCESS){
	cout << "Setting the SUNNonlinSol failed" << endl; exit(-1);
    }
    /* AN: optional CVDiag method if explicit solver is preferred 
    if (CVDiag(m_cvode_mem, NLS) != CV_SUCCESS) {
	    cout << "CVDiag failed " << endl; exit(-1);
    }
    */
    CVodeSetErrFile(m_cvode_mem, NULL);



    N_VDestroy_Serial(y0);
    N_VDestroy_Serial(abstol);


    CVodeSetMaxNumSteps(m_cvode_mem, MXSTEP);//MODIF

    // maximum number of warnings t+h = t (if number negative -> no warnings are issued )
    CVodeSetMaxHnilWarns(m_cvode_mem,2);


}

/**********************************************************/
void Bloch_CV_Model::InitSolver    () {

    ((nvec*) (m_world->solverSettings))->y = N_VNew_Serial(NEQ);
    NV_Ith_S( ((nvec*) (m_world->solverSettings))->y,AMPL )  = m_world->solution[AMPL] ;
    NV_Ith_S( ((nvec*) (m_world->solverSettings))->y,PHASE ) = fmod(m_world->solution[PHASE],TWOPI) ;
    NV_Ith_S( ((nvec*) (m_world->solverSettings))->y,ZC )    = m_world->solution[ZC] ;

    ((nvec*) (m_world->solverSettings))->abstol = N_VNew_Serial(NEQ);
    NV_Ith_S( ((nvec*) (m_world->solverSettings))->abstol,AMPL )  = ATOL1*m_accuracy_factor;
    NV_Ith_S( ((nvec*) (m_world->solverSettings))->abstol,PHASE ) = ATOL2*m_accuracy_factor;
    NV_Ith_S( ((nvec*) (m_world->solverSettings))->abstol,ZC )    = ATOL3*m_accuracy_factor;

    m_reltol = RTOL*m_accuracy_factor;

    int flag;
    flag = CVodeReInit(m_cvode_mem,0,((nvec*) (m_world->solverSettings))->y);
    if(flag != CV_SUCCESS ) {
    	cout << "CVodeReInit failed! aborting..." << endl;
    	if (flag == CV_MEM_NULL) cout << "MEM_NULL"<<endl;
    	if (flag == CV_NO_MALLOC) cout << "CV_NO_MALLOC"<<endl;
    	if (flag == CV_ILL_INPUT) cout << "CV_ILL_INPUT"<<endl;

    	exit (-1);
    }


}

/**********************************************************/
void Bloch_CV_Model::FreeSolver    () {

	N_VDestroy_Serial(((nvec*) (m_world->solverSettings))->y     );
	N_VDestroy_Serial(((nvec*) (m_world->solverSettings))->abstol);

}

/**********************************************************/
bool Bloch_CV_Model::Calculate(double next_tStop){

	if ( m_world->time <= 0.0)  m_world->time = RTOL;

	m_world->solverSuccess=true;

	CVodeSetStopTime(m_cvode_mem, next_tStop);

	int flag;
	do {
		flag=CVode(m_cvode_mem, m_world->time, ((nvec*) (m_world->solverSettings))->y, &m_tpoint, CV_NORMAL);

	} while ((flag==CV_TSTOP_RETURN) && (m_world->time-TIME_ERR_TOL > m_tpoint ));


    //give up if mxstep reached. Return success and hope for the best.
    if (flag == CV_TOO_MUCH_WORK)   {
        return m_world->solverSuccess; 
    }

	if(flag < 0) { m_world->solverSuccess=false; }

	//reinit needed?
	if (m_world->phase == -2.0 && m_world->solverSuccess) {
		CVodeReInit(m_cvode_mem,m_world->time + TIME_ERR_TOL,((nvec*) (m_world->solverSettings))->y);
		// avoiding warnings: (no idea why initial guess of steplength does not work right here...)
		CVodeSetInitStep(m_cvode_mem,m_world->pAtom->GetDuration()/1e9);
	}

	m_world->solution[AMPL]  = NV_Ith_S(((nvec*) (m_world->solverSettings))->y, AMPL );
	m_world->solution[PHASE] = NV_Ith_S(((nvec*) (m_world->solverSettings))->y, PHASE );
	m_world->solution[ZC]    = NV_Ith_S(((nvec*) (m_world->solverSettings))->y, ZC );

	//higher accuracy than 1e-10 not useful. Return success and hope for the best.
	if(m_accuracy_factor < 1e-10) { m_world->solverSuccess=true; }

	return m_world->solverSuccess;
}

/**********************************************************/
void Bloch_CV_Model::PrintFinalStats () {

//    printf("\nFinal Statistics.. \n\n");
//    printf("nst = %-6ld nfe  = %-6ld nsetups = %-6ld \n", m_iopt[NST], m_iopt[NFE] , m_iopt[NSETUPS]);
//    printf("nni = %-6ld ncfn = %-6ld netf = %ld\n \n"   , m_iopt[NNI], m_iopt[NCFN], m_iopt[NETF]);

}


#elif MODEL_ON_GPU == 1 // AN-2022
/**********************************************************/
// RHS of the Bloch equations on GPU
inline static int blochGPU (realtype rt, N_Vector y, N_Vector y_dot, void *pWorld) {

    World* pW = (World*) pWorld;
    // get the main stream where to launch the blochKernel
    cudaStream_t currStream = pW->currStream;
    int N_spins_stream = pW->TotalSpinNumber;
    DynamicVariables* dv = DynamicVariables::instance();
	double t = (double) rt;

    if (t < 0.0 || t > pW->pAtom->GetDuration()) {
        realtype* p_ydot_h = N_VGetHostArrayPointer_Cuda(y_dot);
    	// this case can happen when searching for step size; in this area no solution is needed
        // -> set ydot to any defined value.
    	for (int lspin=0; lspin<N_spins_stream; lspin++) {
            p_ydot_h[3*lspin+0] = 0.;
            p_ydot_h[3*lspin+1] = 0.;
            p_ydot_h[3*lspin+2] = 0.;
        }
        N_VCopyToDevice_Cuda(y_dot);
    	return 0;
    }
	double time = pW->total_time+t;

    // AN-2022: no diffusion and some other dynamic functions are not yet implemented

    // update the SpinPositions if there are dynamic effects
    if (pW->dynamic) {
        dv->m_Motion->GetValue(time, &t); // dummpy pointer as position is different for all spins
        // dv->m_T2prime->GetValue(time, &DeltaB);
        // dv->m_R1->GetValue_AllSpins(time, 1);
        // dv->m_R2->GetValue_AllSpins(time, 2);
        // dv->m_M0->GetValue_AllSpins(time, 3);
    }

    // AN-2022: spin activation check step is skipped

    //get current magn field parameters from the sequence
    int N_SeqParams = 5;
    double d_SeqVal[N_SeqParams] = {0.}; // [B1magn,B1phase,Gx,Gy,Gz]
    // GetValue is left in double precision 
    pW->pAtom->GetValue(d_SeqVal, t);        								    // calculates also pW->NonLinGradField
    if (pW->pStaticAtom != NULL) pW->pStaticAtom->GetValue(d_SeqVal, time);	// calculates static offsets

    // From the AtomicSeq.cpp: is here to keep the other file unchanged
    if (pW->pAtom->HasNonLinGrad()) {
        // with lingering fields, change pW->NonLinGradField_GPU 
        pW->pAtom->GetValueLingeringEddyCurrents(d_SeqVal,t);	         // calculates lingering eddy currents
        cudaMemcpyAsync (pW->NonLinGradField_GPU, pW->NonLinGradField, 
            pW->TotalSpinNumber*sizeof(realtype), cudaMemcpyHostToDevice, pW->currStream);
    }
    
    realtype* seq_arr;      // sequence values in realtype
    // conversion to float if needed
#if defined(SUNDIALS_SINGLE_PRECISION)
    seq_arr = double2floatArray(d_SeqVal, N_SeqParams);
#elif defined(SUNDIALS_DOUBLE_PRECISION)
    seq_arr = d_SeqVal;
#endif
     
    // copy the sequence values to GPU
	cudaMemcpyAsync (pW->d_SeqVal_GPU, seq_arr, N_SeqParams*sizeof(realtype), 
        cudaMemcpyHostToDevice, currStream);

    realtype* p_y_d = N_VGetDeviceArrayPointer_Cuda(y);
    realtype* p_ydot_d = N_VGetDeviceArrayPointer_Cuda(y_dot);
    int grid = (N_spins_stream + block - 1) / block; // blocks per grid, block size is hardcoded
    BlochKernel<<< grid, block, 0, currStream >>>(p_y_d, p_ydot_d, pW->d_SeqVal_GPU, 
                                        pW->m_tx_ideal, pW->m_tx_coils_sum, 
                                        pW->Values, pW->deltaB, pW->SpinPositions,
                                        pW->NonLinGradField_GPU, realtype(pW->GMAXoverB0),
                                        pW->GetNoOfSpinProps(), pW->TotalSpinNumber,
                                        N_spins_stream);
    gpuErrchk(cudaGetLastError());                                    
    return(0);
}

/**********************************************************/
// helper function for CVode functions, checking the return values
static int check_retval(void *returnvalue, const char *funcname, int opt)
{
  int *retval;

  /* Check if SUNDIALS function returned NULL pointer - no memory allocated */
  if (opt == 0 && returnvalue == NULL) {
    fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  /* Check if retval < 0 */
  else if (opt == 1) {
    retval = (int *) returnvalue;
    if (*retval < 0) {
      fprintf(stderr, "\nSUNDIALS_ERROR: %s() failed with retval = %d\n\n",
              funcname, *retval);
      return(1); }}

  /* Check if function returned NULL pointer - no memory allocated */
  else if (opt == 2 && returnvalue == NULL) {
    fprintf(stderr, "\nMEMORY_ERROR: %s() failed - returned NULL pointer\n\n",
            funcname);
    return(1); }

  return(0);

}

/**********************************************************/
// Constructor of the GPU model
Bloch_CV_Model::Bloch_CV_Model     ()  
{

    // create nvec for the model init
    m_world->solverSettings = new nvec;
    N_Vector y0;
    m_tpoint = 0;
    int N_spins_total = m_world->TotalSpinNumber;
    int retval;     // for the return value checks

    streams = (cudaStream_t *) malloc(NoOfStreams * sizeof(cudaStream_t));
    for (int i = 0; i < NoOfStreams; i++) {
        gpuErrchk(cudaStreamCreateWithFlags(&(streams[i]), cudaStreamNonBlocking));
	}

    ifstream CVODEfile;
    long int MXSTEP;

    // setting the solver tolerances
    realtype atol1, atol2, atol3;
    CVODEfile.open("CVODEerr.dat", ifstream::in);
    if (CVODEfile.is_open()) {
        CVODEfile>>m_reltol>>atol1>>atol2>>atol3>>MXSTEP;
        cout<<"CVODE file open"<<endl;
    }
    else  {
        m_reltol       = (realtype)RTOL;
        atol1  = (realtype)ATOL1;
        atol2 = (realtype)ATOL2;
        atol3   = (realtype)ATOL3;
        MXSTEP=100000;
    }

    ((nvec*) (m_world->solverSettings))->m_abstol = std::min({atol1, atol2, atol3});
    y0 = NULL;
    // cvode allocate memory.
    // do CVodeMalloc with dummy values y0,abstol once here;
    // CVodeReInit can later be used
    m_cvode_mem = CVodeCreate (CV_ADAMS);

    // Allocate thevector for ODEs
    y0 = N_VNew_Cuda((NEQ*N_spins_total));  
    if(check_retval((void*)y0, "N_VNew_Cuda", 0)) exit(-1);

    /* Use a non-default cuda stream for streaming and reduction kernel execution */
    SUNCudaThreadDirectExecPolicy stream_exec_policy(block, streams[0]);
    SUNCudaBlockReduceExecPolicy reduce_exec_policy(block, 0, streams[0]);
    retval = N_VSetKernelExecPolicy_Cuda(y0, 
            &stream_exec_policy, &reduce_exec_policy);
    if(check_retval(&retval, "N_VSetKernelExecPolicy_Cuda", 0)) exit(-1);
    
    retval = CVodeInit(m_cvode_mem, blochGPU, 0, y0);
    if(check_retval(&retval, "CVodeInit", 1)) exit(-1);

    retval = CVodeSStolerances(m_cvode_mem, m_reltol, ((nvec*) (m_world->solverSettings))->m_abstol);
    if (check_retval(&retval, "CVodeSStolerances", 1)) exit(-1);

    // use the same World for all streams
    retval = CVodeSetUserData(m_cvode_mem, (void*)m_world);
    if (check_retval(&retval, "CVodeSetUserData", 1)) exit(-1);

    int                mxiter  = 20;
    int                maa     = 0;           // acceleration vectors
    realtype           damping = RCONST(1.0); 
    NLS = SUNNonlinSol_FixedPoint(y0, maa); // y0 - a NVECTOR template for
                                            // cloning vectors needed within the solver

    if (check_retval(NLS, "SUNNonlinSol initialization", 0)) exit(-1);

    retval = SUNNonlinSolSetMaxIters(NLS, mxiter);
    if (check_retval(&retval, "SUNNonlinSolSetMaxIters", 1)) exit(-1);

    retval = SUNNonlinSolSetDamping_FixedPoint(NLS, damping);
    if (check_retval(&retval, "SUNNonlinSolSetDamping_FixedPoint", 1)) exit(-1);

    retval = CVodeSetNonlinearSolver(m_cvode_mem, NLS);
    if (check_retval(&retval, "CVodeSetNonlinearSolver", 1)) exit(-1);    

    N_VDestroy(y0);
    retval = CVodeSetMaxNumSteps(m_cvode_mem, MXSTEP);
    if (check_retval(&retval, "CVodeSetMaxNumSteps", 1)) exit(-1);  

    // maximum number of warnings t+h = t (if number negative -> no warnings are issued )
    retval = CVodeSetMaxHnilWarns(m_cvode_mem,2);
    if (check_retval(&retval, "CVodeSetMaxHnilWarns", 1)) exit(-1);

}

/**********************************************************/
// Initialize the CVode solver on GPU
// is kept with running on a CUDA stream options
// ! the only function which allocates GPU memory while running through sequence loop, necessary to avoid errors
void Bloch_CV_Model::InitSolverGPU (cudaStream_t stream, bool alloc_nvector_gpu) {

    if (alloc_nvector_gpu) {
        (((nvec*) (m_world->solverSettings))->y) = N_VNew_Cuda((NEQ*m_world->TotalSpinNumber));
        if (check_retval((void*)((nvec*)(m_world->solverSettings))->y, "N_VNew_Cuda", 0)) exit(-1);
    }

    int grid = (m_world->TotalSpinNumber + block - 1) / block; // blocks per grid
    int retval;

    /* Use a non-default cuda stream for streaming and reduction kernel execution */
    SUNCudaThreadDirectExecPolicy stream_exec_policy(block, stream);
    SUNCudaBlockReduceExecPolicy reduce_exec_policy(block, 0, stream);  
    retval = N_VSetKernelExecPolicy_Cuda(((nvec*)(m_world->solverSettings))->y, 
            &stream_exec_policy, &reduce_exec_policy);
    if(check_retval(&retval, "N_VSetKernelExecPolicy_Cuda", 0)) exit(-1);

    realtype* p_y = N_VGetDeviceArrayPointer_Cuda((((nvec*)(m_world->solverSettings))->y));
    // copy values from the solution at the previous time to NVector
    SolutionToNVectorKernel <<< grid, block, 0, stream >>> ((m_world->solution),
         p_y, m_world->TotalSpinNumber);

    if ( CVodeReInit(m_cvode_mem,0,(((nvec*) (m_world->solverSettings))->y)) != CV_SUCCESS ) {
        cout << "CVodeReInit failed! aborting..." << endl; exit(-1);
    }

}

/**********************************************************/
// Function to call the solver in the sequence loop
bool Bloch_CV_Model::CalculateGPU(double next_tStop, cudaStream_t stream){

	if ( m_world->time <= 0.0)  m_world->time = (double)RTOL;
	m_world->solverSuccess=true;

	CVodeSetStopTime(m_cvode_mem, (realtype)next_tStop);
	int flag;
	do {
        m_world->currStream = stream; // the stream blochKernel will run on
        flag=CVode(m_cvode_mem, (realtype)m_world->time, ((((nvec*)(m_world->solverSettings))->y)),
                     &m_tpoint, CV_NORMAL); 
	} while ((flag==CV_TSTOP_RETURN) && ((realtype)m_world->time-TIME_ERR_TOL > m_tpoint));

	if(flag < 0) { m_world->solverSuccess=false; }

	//reinit needed?
	if (m_world->phase == -2.0 && m_world->solverSuccess) {
        int retval;
        SUNCudaThreadDirectExecPolicy stream_exec_policy(block, stream);
        SUNCudaBlockReduceExecPolicy reduce_exec_policy(block, 0, stream);
    
        // Use a non-default CUDA stream for streaming and reduction kernel execution 
        retval = N_VSetKernelExecPolicy_Cuda(((nvec*) (m_world->solverSettings))->y,
             &stream_exec_policy, &reduce_exec_policy);
        if(check_retval(&retval, "N_VSetKernelExecPolicy_Cuda", 0)) exit(-1);

        if ( CVodeReInit(m_cvode_mem,((realtype)(m_world->time)+TIME_ERR_TOL), 
            (((nvec*) (m_world->solverSettings))->y)) != CV_SUCCESS ) {
            cout << "CVodeReInit failed! aborting..." << endl; exit(-1);
            }
    }
    // AN-2022: works only <1e-7, then fixed at the value=1e-7
    // CVodeSetInitStep(m_cvode_mem,m_world->pAtom->GetDuration()/1e9);

    realtype* p_y = N_VGetDeviceArrayPointer_Cuda((((nvec*) (m_world->solverSettings))->y));
    // copy the DE solution to the solution
    cudaMemcpyAsync( m_world->solution, p_y, 
        3*m_world->TotalSpinNumber*m_world->GetNoOfCompartments()*sizeof(realtype), 
            cudaMemcpyDeviceToDevice, stream );

	//higher accuracy than 1e-10 not useful. Return success and hope for the best.
    if(m_accuracy_factor < 1e-10) { m_world->solverSuccess=true; }
	return m_world->solverSuccess;
}


/**********************************************************/
void Bloch_CV_Model::FreeSolverGPU   () {

    N_VDestroy( (((nvec*) (m_world->solverSettings))->y) );
}

#endif
