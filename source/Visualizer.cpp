#include "management/Visualizer.h"

#include "base/AOS3D.h"
#include "base/SOA3D.h"
#include "volumes/PlacedVolume.h"

#include "TApplication.h"
#include "TGeoShape.h"
#include "TPolyMarker3D.h"

#include <iostream>

namespace vecgeom {

void Visualizer::AddVolume(VPlacedVolume const &volume) {
  fVolumes.push_back(volume.ConvertToRoot());
  if (verbose > 0) {
    std::cout << "Added volume " << volume << " to Visualizer.\n";
  }
}

template <class ContainerType>
void Visualizer::AddPointsTemplate(ContainerType const &points) {
  const int size = points.size();
  TPolyMarker3D *marker = new TPolyMarker3D(size);
  marker->SetMarkerColor(kRed);
  marker->SetMarkerSize(1);
  marker->SetMarkerStyle(5);
  for (int i = 0; i < size; ++i) {
    marker->SetNextPoint(points.x(i), points.y(i), points.z(i));
  }
  fMarkers.push_back(std::make_pair(marker, true));
  std::cout << "Added " << size << " points to Visualizer.\n";
}

void Visualizer::AddPoints(AOS3D<Precision> const &points) {
  AddPointsTemplate(points);
}

void Visualizer::AddPoints(SOA3D<Precision> const &points) {
  AddPointsTemplate(points);
}

void Visualizer::AddPoints(TPolyMarker3D *marker) {
  fMarkers.push_back(std::make_pair(marker, false));
  std::cout << "Added " << marker->GetN() << " points to Visualizer.\n";
}

void Visualizer::Show() const {
  TApplication app("VecGeom Visualizer", NULL, NULL);
  for (auto volume : fVolumes) {
    const_cast<TGeoShape*>(volume)->Draw();
  }
  for (auto marker : fMarkers) {
    marker.first->Draw();
  }
  app.Run();
}

void Visualizer::Clear() {
  for (auto volume : fVolumes) delete volume;
  for (auto marker : fMarkers) {
    if (marker.second) delete marker.first;
  }
  fVolumes.clear();
  fMarkers.clear();
}

Visualizer::Visualizer() : verbose(1), fVolumes() {}

Visualizer::~Visualizer() {
  Clear();
}

} // End namespace vecgeom