/// \file ShapeDebugger.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_UTILITIES_SHAPEDEBUGGER_H_
#define VECGEOM_UTILITIES_SHAPEDEBUGGER_H_

#include "base/Global.h"

namespace vecgeom {

class VPlacedVolume;

/// \brief Contains methods to debug and verify correctness of shape algorithms.
class ShapeDebugger {

private:

  VPlacedVolume const* fVolume;
  int fMaxMismatches;

public:

  ShapeDebugger(VPlacedVolume const *volume);

  void SetMaxMismatches(int max);

#ifdef VECGEOM_ROOT

  /// Visualizes comparison between the contains algorithm between the VecGeom
  /// volume and its ROOT equivalent. Circles are matches, crosses are
  /// mismatches between results. Green means both are inside, red means both
  /// are outside, magenta is VecGeom inside but not ROOT, and blue is ROOT
  /// inside but not VecGeom.
  void CompareContainsToROOT(
      Vector3D<Precision> const &bounds,
      int nSamples = 1024) const;

  /// Visualizes comparison between the DistanceToIn algorithm between the
  /// VecGeom volume and its ROOT equivalent. Yellow crosses are misses from
  /// both algorithms. Red rays are VecGeom intersections that miss in ROOT.
  /// Dashed blue rays are ROOT intersections that miss in VecGeom. Green are
  /// intersections that agree between ROOT and VecGeom. Magenta rays are
  /// VecGeom intersections that have a different result from ROOT, with the
  /// solid line being the VecGeom result and the dashed line being the ROOT
  /// result.
  void CompareDistanceToInToROOT(
      Vector3D<Precision> const &bounds,
      int nSamples = 1024) const;

  /// Visualizes comparison between the DistanceToOut algorithm between the
  /// VecGeom volume and its ROOT equivalent. Red lines are ROOT intersections
  /// where VecGeom misses the surface. Green lines are agreements between
  /// VecGeom and ROOT. Purple lines are different results between VecGeom and
  /// ROOT, where the solid line is VecGeom's result and the dashed line is
  /// ROOT's result.
  void CompareDistanceToOutToROOT(
      Vector3D<Precision> const &bounds,
      int nSamples = 1024) const;

#endif

};

} // End namespace vecgeom

#endif // VECGEOM_UTILITIES_SHAPEDEBUGGER_H_