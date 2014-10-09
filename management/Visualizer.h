/// \file Visualizer.h
/// \author Johannes de Fine Licht (johannes.definelicht@cern.ch)

#ifndef VECGEOM_MANAGEMENT_VISUALIZER_H_
#define VECGEOM_MANAGEMENT_VISUALIZER_H_

#ifndef VECGEOM_ROOT
#error "Visualizer currently only available with ROOT enabled."
#endif

#include "base/Global.h"

#include <list>
#include <utility>
#include <vector>

class TPolyMarker3D;
class TGeoShape;

namespace vecgeom {

/// \brief Visualize volumes through ROOT.
class Visualizer {

private:

  int verbose;
  std::list<TGeoShape const*> fVolumes;
  std::list<std::pair<TPolyMarker3D*, bool> > fMarkers;

public:

  static Visualizer& Instance() {
    static Visualizer instance;
    return instance;
  }

  void AddVolume(VPlacedVolume const &volume);

  void AddPoints(AOS3D<Precision> const &points);

  void AddPoints(SOA3D<Precision> const &points);

  void AddPoints(TPolyMarker3D *marker);

  void Show() const;

  void Clear();

private:

  Visualizer();
  ~Visualizer();

  template <class ContainerType>
  void AddPointsTemplate(ContainerType const &points);

};

} // End namespace vecgeom

#endif // VECGEOM_MANAGEMENT_VISUALIZER_H_