/// \file Visualizer.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_UTILITIES_VISUALIZER_H_
#define VECGEOM_UTILITIES_VISUALIZER_H_

#ifndef VECGEOM_ROOT
#error "Visualizer currently only available with ROOT enabled."
#endif

#include "base/Global.h"

#include <list>
#include <utility>
#include <vector>

class TGeoShape;
class TPolyLine3D;
class TPolyMarker3D;

namespace vecgeom {

/// \brief Visualize volumes through ROOT.
class Visualizer {

private:

  int fVerbosity;
  std::list<std::pair<TGeoShape const*, bool> > fVolumes;
  std::list<std::pair<TPolyMarker3D*, bool> > fMarkers;
  std::list<std::pair<TPolyLine3D*, bool> > fLines;

public:

  Visualizer();

  ~Visualizer();

  void AddVolume(VPlacedVolume const &volume);

  void AddVolume(TGeoShape const *volume);

  void AddPoints(AOS3D<Precision> const &points);

  void AddPoints(SOA3D<Precision> const &points);

  void AddPoints(TPolyMarker3D *marker);

  void AddLine(Vector3D<Precision> const &p0, Vector3D<Precision> const &p1);

  void AddLine(TPolyLine3D *line);

  /// Runs a ROOT application, drawing the added volumes and points.
  void Show() const;

  /// Removes all previously added volumes and points.
  void Clear();

  /// \param level 0 = no output. 1 = reports when the visualizer is altered.
  void SetVerbosity(int level);

private:

  template <class ContainerType>
  void AddPointsTemplate(ContainerType const &points);

};

} // End namespace vecgeom

#endif // VECGEOM_MANAGEMENT_VISUALIZER_H_