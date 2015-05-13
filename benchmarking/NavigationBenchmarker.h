/// \file NavigationBenchmarker.h
/// \author Guilherme Lima (lima at fnal dot gov)
//
// 2014-11-26 G.Lima - created with help from Sandro

#ifndef VECGEOM_BENCHMARKING_NAVIGATIONBENCHMARKER_H_
#define VECGEOM_BENCHMARKING_NAVIGATIONBENCHMARKER_H_

#include "base/Global.h"
#include "base/SOA3D.h"
#include "volumes/PlacedVolume.h"
#include "navigation/NavigationState.h"

namespace vecgeom {

  // VECGEOM_HOST_FORWARD_DECLARE( class VPlacedVolume; );
  // VECGEOM_HOST_FORWARD_DECLARE( class NavigationState; );

  using VPlacedVolume_t = VECGEOM_IMPL_NAMESPACE::VPlacedVolume const*;

#ifdef VECGEOM_CUDA_INTERFACE
  void GetVolumePointers( std::list<DevicePtr<cuda::VPlacedVolume>> &volumesGpu );
#endif

  Precision benchmarkLocatePoint(
    VPlacedVolume const* top,
    int nPoints,
    int nReps,
    SOA3D<Precision> const& points
    );

  Precision benchmarkSerialNavigation(
    VPlacedVolume const* top,
    int nPoints,
    int nReps,
    SOA3D<Precision> const& points,
    SOA3D<Precision> const& dirs
    );

  Precision benchmarkVectorNavigation(
    VPlacedVolume const* top,
    int nPoints,
    int nReps,
    SOA3D<Precision> const& points,
    SOA3D<Precision> const& dirs
    );

  void runNavigationBenchmarks( LogicalVolume const* top, int np, int nreps, Precision bias = 0.8);

  bool validateNavigationStepAgainstRoot(
    Vector3D<Precision> const& pos,
    Vector3D<Precision> const& dir,
    NavigationState const& testState,
    Precision maxStep,
    Precision testStep
    );

  bool validateVecGeomNavigation( int npts, SOA3D<Precision> const& points, SOA3D<Precision> const& dirs);

#ifdef VECGEOM_ROOT
  Precision benchmarkROOTNavigation(
    VPlacedVolume const* top,
    int nPoints,
    int nReps,
    SOA3D<Precision> const& points,
    SOA3D<Precision> const& dirs );
#endif

#ifdef VECGEOM_CUDA
Precision runNavigationCuda( void* gpu_ptr, void* gpu_out_ptr, int maxDepth, VPlacedVolume const* volume, unsigned npoints,
                             Precision const *const posX, Precision const *const posY, Precision const *const posZ,
                             Precision const *const dirX, Precision const *const dirY, Precision const *const dirZ,
                             Precision const *const pSteps,      Precision *const steps );
#endif

} // End namespace vecgeom

#endif // VECGEOM_BENCHMARKING_NAVIGATIONBENCHMARKER_H_
