#include "utilities/Visualizer.h"

#include "base/AOS3D.h"
#include "base/SOA3D.h"
#include "volumes/PlacedVolume.h"

#include "TApplication.h"
#include "TGeoShape.h"
#include "TPolyLine3D.h"
#include "TPolyMarker3D.h"

#include <iostream>

namespace vecgeom {

Visualizer::Visualizer() : fVerbosity(1), fVolumes() {}

Visualizer::~Visualizer() {
  fVerbosity = 0;
  Clear();
}

void Visualizer::AddVolume(VPlacedVolume const &volume) {
  fVolumes.push_back(std::make_pair(volume.ConvertToRoot(), true));
  if (fVerbosity > 0) {
    std::cout << "Added volume " << volume << " to Visualizer.\n";
  }
}

void Visualizer::AddVolume(TGeoShape const *volume) {
  fVolumes.push_back(std::make_pair(volume, false));
  if (fVerbosity > 0) {
    std::cout << "Added volume " << volume << " to Visualizer.\n";
  }
}

void Visualizer::AddPoints(AOS3D<Precision> const &points) {
  AddPointsTemplate(points);
}

void Visualizer::AddPoints(SOA3D<Precision> const &points) {
  AddPointsTemplate(points);
}

void Visualizer::AddPoints(TPolyMarker3D *marker) {
  fMarkers.push_back(std::make_pair(marker, false));
  if (fVerbosity > 0) {
    std::cout << "Added " << marker->GetN() << " points to Visualizer.\n";
  }
}

void Visualizer::AddLine(
    Vector3D<Precision> const &p0,
    Vector3D<Precision> const &p1) {

  TPolyLine3D *line = new TPolyLine3D(2);
  line->SetPoint(0, p0[0], p0[1], p0[2]);
  line->SetPoint(1, p1[0], p1[1], p1[2]);
  line->SetLineColor(kBlue);
  fLines.push_back(std::make_pair(line, true));
  if (fVerbosity > 0) {
    std::cout << "Added line " << p0 << "--" << p1 << " to Visualizer.\n";
  }
}

void Visualizer::AddLine(TPolyLine3D *line) {
  fLines.push_back(std::make_pair(line, false));
  auto GetPoint = [&line] (int index) {
    float *pointArray = line->GetP();
    int offset = 3*index;
    return Vector3D<Precision>(
        pointArray[offset], pointArray[offset+1], pointArray[offset+2]);;
  };
  if (line->GetN() == 2) {
    std::cout << "Added line " << GetPoint(0) << "--" << GetPoint(1)
              << " to Visualizer.\n";
  } else {
    std::cout << "Added line with " << line->GetN()
              << " points to Visualizer.\n";
  }
}

void Visualizer::Show() const {
  TApplication app("VecGeom Visualizer", NULL, NULL);
  for (auto volume : fVolumes) {
    const_cast<TGeoShape*>(volume.first)->Draw();
  }
  for (auto marker : fMarkers) {
    marker.first->Draw();
  }
  for (auto line : fLines) {
    line.first->Draw();
  }
  app.Run();
}

void Visualizer::Clear() {
  for (auto volume : fVolumes) {
    if (volume.second) delete volume.first;
  }
  for (auto marker : fMarkers) {
    if (marker.second) delete marker.first;
  }
  for (auto line : fLines) {
    if (line.second) delete line.first;
  }
  fVolumes.clear();
  fMarkers.clear();
  fLines.clear();
  if (fVerbosity > 0) {
    std::cout << "Cleared Visualizer content.\n";
  }
}

void Visualizer::SetVerbosity(int level) {
  fVerbosity = level;
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
  if (fVerbosity > 0) {
    std::cout << "Added " << size << " points to Visualizer.\n";
  }
}

} // End namespace vecgeom