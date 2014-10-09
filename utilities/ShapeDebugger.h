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
  /// Blue rays are ROOT intersections that miss in VecGeom. Green are
  /// intersections that agree between ROOT and VecGeom. Magenta are VecGeom
  /// intersections that have a different result from ROOT.
  void CompareDistanceToInToROOT(
      Vector3D<Precision> const &bounds,
      int nSamples = 1024) const;

#endif

};

} // End namespace vecgeom

#endif // VECGEOM_UTILITIES_SHAPEDEBUGGER_H_