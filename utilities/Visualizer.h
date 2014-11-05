/// \file Visualizer.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_UTILITIES_VISUALIZER_H_
#define VECGEOM_UTILITIES_VISUALIZER_H_

#ifndef VECGEOM_ROOT
#error "Visualizer currently only available with ROOT enabled."
#endif

#include "base/Global.h"

#include "volumes/PlacedVolume.h"

#include <list>
#include <memory>
#include <utility>

class TGeoMatrix;
class TGeoShape;
class TGeoVolume;
class TPolyLine3D;
class TPolyMarker3D;

namespace vecgeom {

template <typename T> class AOS3D;
template <typename T> class SOA3D;
class Transformation3D;
template <typename T> class Vector3D;

/// \brief Visualize volumes through ROOT.
class Visualizer {

private:

  int fVerbosity;
  std::list<std::tuple<std::shared_ptr<const TGeoShape>,
                       std::unique_ptr<TGeoMatrix>,
                       std::unique_ptr<TGeoVolume> > > fVolumes;
  std::list<std::unique_ptr<TPolyMarker3D> > fMarkers;
  std::list<std::unique_ptr<TPolyLine3D> > fLines;

public:

  Visualizer();

  ~Visualizer();

  void AddVolume(VPlacedVolume const &volume);

  void AddVolume(VPlacedVolume const &volume,
                 Transformation3D const &transformation);

  void AddVolume(std::shared_ptr<const TGeoShape> rootVolume);

  void AddVolume(std::shared_ptr<const TGeoShape> rootVolume,
                 Transformation3D const &position);

  void AddPoints(AOS3D<Precision> const &points);

  void AddPoints(SOA3D<Precision> const &points);

  void AddPoints(TPolyMarker3D const &marker);

  void AddLine(Vector3D<Precision> const &p0, Vector3D<Precision> const &p1);

  void AddLine(TPolyLine3D const &line);

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
