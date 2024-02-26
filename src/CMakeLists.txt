list (APPEND CORE_SRC AnalyticCoil.cpp AnalyticCoil.h
  AnalyticGradPulse.cpp AnalyticGradPulse.h AnalyticPulseShape.cpp
  AnalyticPulseShape.h AnalyticRFPulse.cpp AnalyticRFPulse.h
  AtomicSequence.cpp AtomicSequence.h Attribute.cpp Attribute.h
  BinaryContext.cpp BinaryContext.h BinaryIO.h BinaryIO.cpp
  BiotSavartLoop.cpp BiotSavartLoop.h ConcatSequence.cpp ConcatSequence.h
  ConstantGradPulse.cpp ConstantGradPulse.h Container.cpp Container.h 
  ContainerSequence.cpp ContainerSequence.h DOMTreeErrorReporter.cpp
  DOMTreeErrorReporter.h Debug.h Declarations.h DelayAtomicSequence.cpp
  DelayAtomicSequence.h DynamicVariables.cpp DynamicVariables.h
  EddyPulse.cpp EddyPulse.h EmptyPulse.cpp EmptyPulse.h ExternalCoil.cpp
  ExternalCoil.h ExternalGradPulse.cpp ExternalGradPulse.h
  ExternalRFPulse.cpp ExternalRFPulse.h ExternalPulseData Event.h GaussianRFPulse.cpp
  GaussianRFPulse.h HardRFPulse.cpp
  HardRFPulse.h HDF5IO.cpp HDF5IO.h IdealCoil.cpp IdealCoil.h KSpace.h md5.cpp md5.h
  Microstructure.cpp Microstructure.h MicrostructureBoxes.cpp
  MicrostructureBoxes.h MicrostructureCrossingFiber.cpp
  MicrostructureCrossingFiber.h MicrostructureKissingFiber.cpp
  MicrostructureKissingFiber.h MicrostructureSingleFiber.cpp
  MicrostructureSingleFiber.h Module.cpp Module.h
  ModulePrototypeFactory.cpp ModulePrototypeFactory.h mtg_functions.h NDData.h
  OutputSequenceData.h OutputSequenceData.cpp Parameters.cpp Parameters.h 
  Prototype.cpp Prototype.h
  PrototypeFactory.cpp PrototypeFactory.h Pulse.cpp Pulse.h 
  RepIter.cpp RepIter.h MultiPoolSample.cpp MultiPoolSample.h
  SampleReorderShuffle.cpp SampleReorderShuffle.h
  SampleReorderStrategyInterface.h SechRFPulse.cpp SechRFPulse.h
  Sequence.cpp Sequence.h SequenceTree.cpp SequenceTree.h Signal.cpp
  Signal.h SimpleIO.h SimpleIO.cpp 
  SincRFPulse.cpp SincRFPulse.h SpiralGradPulse.cpp SpiralGradPulse.h
  StrX.cpp StrX.h TPOI.cpp TPOI.h Trajectory.cpp Trajectory.h
  Trajectory1D.h Trajectory1D.cpp TrajectoryDiffusion.h TrajectoryDiffusion.cpp
  TrajectoryEmpty.h TrajectoryFlow.h TrajectoryFlow.cpp 
  TrajectoryInterface.cpp TrajectoryInterface.h TrajectoryRespiration.h 
  TrajectoryRespiration.cpp TrajectoryT2s.h
  TrajectoryT2s.cpp TrapGradPulse.cpp TrapGradPulse.h
  TriangleGradPulse.cpp TriangleGradPulse.h TxRxPhase.cpp TxRxPhase.h
  XMLIO.cpp XMLIO.h config.h ginac_functions.h list.h
  list_c.h rng.cpp rng.h ) 
    

# AN-2022 -----------------------------------------------------------------------
# These files contain parts of the GPU computations but are compiled with gcc
# they can switch between GPU and CPU versions using the MODEL_ON_GPU definition
list (APPEND CORE_DUAL_SRC CoilArray.cpp CoilArray.h CoilPrototypeFactory.cpp
    CoilPrototypeFactory.h GradPulse.cpp GradPulse.h RFPulse.cpp  RFPulse.h 
    Simulator.cpp Simulator.h TrajectoryMotion.cpp TrajectoryMotion.h
    )

# These files contain parts of the GPU computations and need to be compiled with nvcc for GPU
# they can switch between GPU and CPU versions using the MODEL_ON_GPU definition
list (APPEND CORE_COMPUT_SRC Coil.cpp Sample.cpp World.cpp
    Coil.h Sample.h World.h
    Bloch_CV_Model.cpp Bloch_CV_Model.h)
    # Note for AN: Model needs to be compiled with a different definition for pjemris

# Those directories do not exist, right?
#include_directories (${PROJECT_SOURCE_DIR}/src/core
#  ${PROJECT_SOURCE_DIR}/src/matrix/io) 

if (LINUX)
  list (APPEND COMMON_LIBS dl)
endif()

list (APPEND COMMON_LIBS ${HDF5_CXX_LIBRARIES} ${Xerces_LIBRARY} ${GINAC_LIBRARIES} ${SUNDIALS_LIBRARIES} ${ISMRMRD_LIBRARIES})
if (NOT MODEL_ON_GPU)

  add_library(core STATIC ${CORE_SRC} ${CORE_COMPUT_SRC} ${CORE_DUAL_SRC} Bloch_McConnell_CV_Model.cpp Bloch_McConnell_CV_Model.h )

  if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
      target_link_libraries (core ${COMMON_LIBS} )
  endif()
  add_executable (jemris Model.h Model.cpp jemris.cpp)
  target_link_libraries (jemris ${COMMON_LIBS} core) 

  # Install the binary to bin
  install (TARGETS jemris DESTINATION "bin")

  add_executable (sanityck Model.h Model.cpp sanityck.cpp)
  target_link_libraries (sanityck ${COMMON_LIBS} core) 

  if (MPI_C_FOUND)
      add_executable (pjemris Mpi2Evolution.h
          Mpi2Evolution.cpp mpi_Model.h Model.h Model.cpp pjemris.cpp)
      find_package(Threads REQUIRED)
      target_link_libraries (pjemris core ${COMMON_LIBS}
          ${MPI_C_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
      target_compile_definitions (pjemris PRIVATE PARALLEL)
      if (MPI_C_COMPILE_FLAGS)
          set_target_properties(pjemris PROPERTIES
          COMPILE_FLAGS "${MPI_COMPILE_FLAGS} -DHAVE_MPI_THREADS")
      else()
          set_target_properties(pjemris PROPERTIES
          COMPILE_FLAGS "-DHAVE_MPI_THREADS")
      endif()
      if(MPI_LINK_FLAGS)
          set_target_properties(pjemris PROPERTIES
          LINK_FLAGS "${MPI_LINK_FLAGS}")
      endif()
      install (TARGETS pjemris DESTINATION "bin")
  endif()

  # Install the examples and matlab GUIs
  install (TARGETS jemris DESTINATION "bin")

  add_test (
  NAME sequences
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/share/examples
  COMMAND sanityck . 1)
  add_test (
  NAME signals 
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/share/examples
  COMMAND sanityck . 2)
  add_test (
  NAME coils 
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/share/examples
  COMMAND sanityck . 3)
  add_test (
  NAME pulseq_output 
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/share/examples
  COMMAND sanityck . 4)

  if (MPI_FOUND)
  add_test(
      NAME mpi 
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/share
      COMMAND ${MPIEXEC} -np 3 ${PROJECT_BINARY_DIR}/src/pjemris examples/approved/mpi_simu.xml)
  endif()

elseif (MODEL_ON_GPU MATCHES 1)

  #    gjemris compilation
  add_definitions(-w)
  find_package(CUDA)
  list(APPEND CUDA_NVCC_FLAGS "--compiler-options; -std=c++11; -w") #-arch=sm_20; 
  include_directories(/usr/local/cuda/include/)

  set_source_files_properties(${CORE_COMPUT_SRC} Model.h Model.cpp
    TrajectoryKernels.cuh TrajectoryKernels.cu
    PROPERTIES LANGUAGE CUDA)
  add_library ( core_GPU STATIC ${CORE_SRC} ${CORE_DUAL_SRC} ${CORE_COMPUT_SRC} Model.h Model.cpp
     TrajectoryKernels.cuh TrajectoryKernels.cu)
  set_target_properties(core_GPU PROPERTIES COMPILE_FLAGS -DMODEL_ON_GPU=1)
  target_link_libraries (core_GPU ${COMMON_LIBS} )

  add_executable (gjemris jemris.cpp )
  target_link_libraries (gjemris core_GPU ${COMMON_LIBS})

  # Install the binary to bin
  install (TARGETS gjemris DESTINATION "bin")

  set_source_files_properties(sanytick.cpp PROPERTIES LANGUAGE CUDA)
  add_executable (sanityck_gpu sanityck.cpp)
  set_target_properties(sanityck_gpu PROPERTIES COMPILE_FLAGS -DMODEL_ON_GPU=1)
  target_link_libraries (sanityck_gpu core_GPU ${COMMON_LIBS} ) 

  add_test (
  NAME gpu_signals
  WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/share/examples
  COMMAND sanityck_gpu . 6)

  find_program (jemris_installed jemris)
  if (jemris_installed)
    add_test (
    NAME gpu_vs_cpu_signals
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}/share/examples
    COMMAND sanityck_gpu . 7)
  endif (jemris_installed)

endif (NOT MODEL_ON_GPU)





