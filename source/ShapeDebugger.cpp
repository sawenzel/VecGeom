#include "utilities/ShapeDebugger.h"

#include "volumes/PlacedVolume.h"
#include "volumes/utilities/VolumeUtilities.h"

#ifdef VECGEOM_ROOT
#include "utilities/Visualizer.h"
#include "TGeoShape.h"
#include "TPolyLine3D.h"
#include "TPolyMarker3D.h"
#endif

#include <iostream>

namespace vecgeom {

ShapeDebugger::ShapeDebugger(VPlacedVolume const *volume)
  : fVolume(volume), fMaxMismatches(8) {}

void ShapeDebugger::SetMaxMismatches(int max) {
  if (max < 0) max = 0;
  fMaxMismatches = max;
}

#ifdef VECGEOM_ROOT

void ShapeDebugger::CompareContainsToROOT(
    Vector3D<Precision> const &bounds,
    int nSamples) const {

  TGeoShape const *rootShape = fVolume->ConvertToRoot();

  std::cout << "Comparing contains between VecGeom and ROOT:\n";

  TPolyMarker3D vecgeomInside(nSamples);
  vecgeomInside.SetMarkerStyle(5);
  vecgeomInside.SetMarkerColor(kMagenta);
  TPolyMarker3D rootInside(nSamples);
  rootInside.SetMarkerStyle(5);
  rootInside.SetMarkerColor(kBlue);
  TPolyMarker3D bothInside(nSamples);
  bothInside.SetMarkerStyle(4);
  bothInside.SetMarkerColor(kGreen);
  TPolyMarker3D bothOutside(nSamples);
  bothOutside.SetMarkerStyle(4);
  bothOutside.SetMarkerColor(kRed);

  std::vector<Vector3D<Precision> > mismatchPoints;
  std::vector<std::pair<Precision, double> > mismatchResults;
  int vecgeomCount = 0, rootCount = 0, mismatches = 0;
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> sample = volumeUtilities::SamplePoint(bounds);
    bool vecgeomResult = fVolume->Contains(sample);
    bool rootResult = rootShape->Contains(&sample[0]);
    vecgeomCount += vecgeomResult;
    rootCount += rootResult;
    if (vecgeomResult != rootResult) {
      ++mismatches;
      mismatchPoints.push_back(sample);
      mismatchResults.push_back(std::make_pair(vecgeomResult, rootResult));
    }
    if (vecgeomResult && rootResult) {
      bothInside.SetNextPoint(sample[0], sample[1], sample[2]);
    } else if (vecgeomResult && !rootResult) {
      vecgeomInside.SetNextPoint(sample[0], sample[1], sample[2]);
    } else if (!vecgeomResult && rootResult) {
      rootInside.SetNextPoint(sample[0], sample[1], sample[2]);
    } else {
      bothOutside.SetNextPoint(sample[0], sample[1], sample[2]);
    }
  }
  std::cout << "  VecGeom: " << vecgeomCount << " / " << nSamples
            << " contained.\n"
            << "  ROOT:    " << rootCount << " / " << nSamples
            << " contained.\n"
            << "  Mismatches detected: " << mismatches << " / " << nSamples
            << "\n";
  if (fMaxMismatches > 0 && mismatches > 0) {
    std::cout << "\nMismatching points [<Point>: <VecGeom> / <ROOT>]:\n";
    int i = 0, iMax = mismatchPoints.size();
    while (i < fMaxMismatches && i < iMax) {
      std::cout << mismatchPoints[i] << ": " << mismatchResults[i].first
                << " / " << mismatchResults[i].second << "\n";
      ++i;
    }
  }

  Visualizer visualizer;
  visualizer.SetVerbosity(0);
  visualizer.AddVolume(rootShape);
  visualizer.AddPoints(&vecgeomInside);
  visualizer.AddPoints(&rootInside);
  visualizer.AddPoints(&bothInside);
  visualizer.AddPoints(&bothOutside);
  visualizer.Show();

  delete rootShape;
}

void ShapeDebugger::CompareDistanceToInToROOT(
    Vector3D<Precision> const &bounds,
    int nSamples) const {

  TGeoShape const *rootShape = fVolume->ConvertToRoot();

  std::cout << "Comparing DistanceToIn between VecGeom and ROOT:\n";

  TPolyLine3D vecgeomHits(2);
  vecgeomHits.SetLineColor(kRed);
  TPolyLine3D rootHits(2);
  rootHits.SetLineColor(kBlue);
  rootHits.SetLineStyle(2);
  TPolyMarker3D bothMiss(nSamples);
  bothMiss.SetMarkerStyle(5);
  bothMiss.SetMarkerColor(kYellow);
  TPolyLine3D sameResult(2);
  sameResult.SetLineColor(kGreen);
  TPolyLine3D differentResultVecgeom(2);
  differentResultVecgeom.SetLineColor(kMagenta);
  TPolyLine3D differentResultRoot(2);
  differentResultRoot.SetLineColor(kMagenta);
  differentResultRoot.SetLineStyle(2);

  std::vector<TPolyLine3D*> rays;
  std::vector<Vector3D<Precision> > mismatchPoints;
  std::vector<Vector3D<Precision> > mismatchDirections;
  std::vector<std::pair<Precision, double> > mismatchResults;
  int vecgeomCount = 0, rootCount = 0, mismatches = 0;
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> point, direction;
    do {
      point = volumeUtilities::SamplePoint(bounds);
    } while (rootShape->Contains(&point[0]));
    direction = volumeUtilities::SampleDirection();
    Precision vecgeomResult = fVolume->DistanceToIn(point, direction);
    double rootResult =
        rootShape->DistFromOutside(&point[0], &direction[0]);
    bool vecgeomMiss = vecgeomResult == kInfinity;
    bool rootMiss = rootResult == 1e30;
    bool same = (vecgeomMiss && rootMiss) ||
                Abs(rootResult - vecgeomResult) < kTolerance;
    vecgeomCount += !vecgeomMiss;
    rootCount += !rootMiss;
    if (!same) {
      ++mismatches;
      mismatchPoints.push_back(point);
      mismatchDirections.push_back(direction);
      mismatchResults.push_back(std::make_pair(vecgeomResult, rootResult));
    }
    if (same && vecgeomMiss) {
      bothMiss.SetNextPoint(point[0], point[1], point[2]);
    } else {
      auto AddLine = [&] (
          TPolyLine3D const &line,
          Vector3D<Precision> const &intersection) {
        TPolyLine3D *ray = new TPolyLine3D(line);
        ray->SetPoint(0, point[0], point[1], point[2]);
        ray->SetPoint(1, intersection[0], intersection[1], intersection[2]);
        rays.push_back(ray);
      };
      if (!vecgeomMiss && rootMiss) {
        AddLine(vecgeomHits, point + vecgeomResult*direction);
      } else if (vecgeomMiss && !rootMiss) {
        AddLine(rootHits, point + rootResult*direction);
      } else if (same) {
        AddLine(sameResult, point + vecgeomResult*direction);
      } else {
        AddLine(differentResultVecgeom, point + vecgeomResult*direction);
        AddLine(differentResultRoot, point + rootResult*direction);
      }
    }
  }
  std::cout << "  VecGeom: " << vecgeomCount << " / " << nSamples
            << " hit the volume.\n"
            << "  ROOT:    " << rootCount << " / " << nSamples
            << " hit the volume.\n"
            << "  Mismatches detected: " << mismatches << " / " << nSamples
            << "\n";
  if (fMaxMismatches > 0 && mismatches > 0) {
    std::cout << "\nMismatching rays [<Point> -> <Direction>: "
                 "<VecGeom> / <ROOT>]:\n";
    int i = 0, iMax = mismatchPoints.size();
    while (i < fMaxMismatches && i < iMax) {
      std::cout << mismatchPoints[i] << " -> " << mismatchDirections[i]
                << ": " << mismatchResults[i].first << " / "
                << mismatchResults[i].second << "\n";
      ++i;
    }
  }

  Visualizer visualizer;
  visualizer.SetVerbosity(0);
  visualizer.AddVolume(rootShape);
  for (auto ray : rays) {
    visualizer.AddLine(ray);
  }
  visualizer.AddPoints(&bothMiss);
  visualizer.Show();

  for (auto ray : rays) delete ray;
  delete rootShape;
}

void ShapeDebugger::CompareDistanceToOutToROOT(
    Vector3D<Precision> const &bounds,
    int nSamples) const {

  TGeoShape const *rootShape = fVolume->ConvertToRoot();

  std::cout << "Comparing DistanceToOut between VecGeom and ROOT:\n";

  TPolyLine3D sameResult;
  sameResult.SetLineColor(kGreen);
  TPolyLine3D differentResultVecgeom;
  differentResultVecgeom.SetLineColor(kMagenta);
  TPolyLine3D differentResultRoot;
  differentResultRoot.SetLineColor(kMagenta);
  differentResultRoot.SetLineStyle(2);
  TPolyLine3D rootHits;
  rootHits.SetLineColor(kRed);
  rootHits.SetLineStyle(2); 

  std::vector<TPolyLine3D*> rays;
  std::vector<Vector3D<Precision> > mismatchPoints;
  std::vector<Vector3D<Precision> > mismatchDirections;
  std::vector<std::pair<Precision, double> > mismatchResults;
  int hits = 0, mismatches = 0;
  for (int i = 0; i < nSamples; ++i) {
    Vector3D<Precision> point, direction;
    do {
      point = volumeUtilities::SamplePoint(bounds);
    } while (!rootShape->Contains(&point[0]));
    direction = volumeUtilities::SampleDirection();
    Precision vecgeomResult = fVolume->DistanceToOut(point, direction);
    double rootResult =
        rootShape->DistFromInside(&point[0], &direction[0]);
    bool vecgeomMiss = vecgeomResult == kInfinity;
    bool same = !vecgeomMiss && Abs(rootResult - vecgeomResult) < kTolerance;
    if (!same) {
      ++mismatches;
      mismatchPoints.push_back(point);
      mismatchDirections.push_back(direction);
      mismatchResults.push_back(std::make_pair(vecgeomResult, rootResult));
    }
    auto AddLine = [&] (
        TPolyLine3D const &line,
        Vector3D<Precision> const &intersection) {
      TPolyLine3D *ray = new TPolyLine3D(line);
      ray->SetPoint(0, point[0], point[1], point[2]);
      ray->SetPoint(1, intersection[0], intersection[1], intersection[2]);
      rays.push_back(ray);
    };
    if (vecgeomMiss) {
      AddLine(rootHits, point + rootResult*direction);
    } else {
      ++hits;
      if (same) {
        AddLine(sameResult, point + vecgeomResult*direction);
      } else {
        AddLine(differentResultVecgeom, point + vecgeomResult*direction);
        AddLine(differentResultRoot, point + rootResult*direction);
      }
    }
  }
  std::cout << "  VecGeom: " << hits << " / " << nSamples
            << " hit the boundary.\n"
            << "  Mismatches detected: " << mismatches << " / " << nSamples
            << "\n";
  if (fMaxMismatches > 0 && mismatches > 0) {
    std::cout << "\nMismatching rays [<Point> -> <Direction>: "
                 "<VecGeom> / <ROOT>]:\n";
    int i = 0, iMax = mismatchPoints.size();
    while (i < fMaxMismatches && i < iMax) {
      std::cout << mismatchPoints[i] << " -> " << mismatchDirections[i]
                << ": " << mismatchResults[i].first << " / "
                << mismatchResults[i].second << "\n";
      ++i;
    }
  }

  Visualizer visualizer;
  visualizer.SetVerbosity(0);
  visualizer.AddVolume(rootShape);
  for (auto ray : rays) {
    visualizer.AddLine(ray);
  }
  visualizer.Show();

  for (auto ray : rays) delete ray;
  delete rootShape;
}

#endif

} // End namespace vecgeom