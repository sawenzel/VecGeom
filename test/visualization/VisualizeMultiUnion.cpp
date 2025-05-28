#include "VecGeom/base/Config.h"

#ifndef VECCORE_CUDA

#include "VecGeom/volumes/LogicalVolume.h"
#include "VecGeom/management/GeoManager.h"
#include "../benchmark/ArgParser.h"
#include "VecGeom/base/Stopwatch.h"
#include <iostream>
#include "VecGeom/volumes/MultiUnion.h"
#include "VecGeom/base/RNG.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/utilities/VolumeUtilities.h"

#ifdef VECGEOM_ROOT
#include "VecGeomTest/Visualizer.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"
#include "VecGeom/volumes/Box.h"
#endif

using namespace vecgeom;

#endif

#ifdef VECGEOM_ROOT
void AddComponentToVisualizer(MultiUnionStruct const &munion, size_t i, Visualizer &visualizer)
{
  VPlacedVolume const *placedbox = munion.fVolumes[i];
  Vector3D<double> dimensions    = ((UnplacedBox *)placedbox->GetUnplacedVolume())->dimensions();
  TPolyLine3D pl(2);
  pl.SetLineColor(kBlue);
  Vector3D<double> vert[8];
  Vector3D<double> local;
  local[0] = -dimensions[0];
  local[1] = -dimensions[1];
  local[2] = -dimensions[2];
  placedbox->GetTransformation()->InverseTransform(local, vert[0]);
  local[0] = -dimensions[0];
  local[1] = dimensions[1];
  local[2] = -dimensions[2];
  placedbox->GetTransformation()->InverseTransform(local, vert[1]);
  local[0] = dimensions[0];
  local[1] = dimensions[1];
  local[2] = -dimensions[2];
  placedbox->GetTransformation()->InverseTransform(local, vert[2]);
  local[0] = dimensions[0];
  local[1] = -dimensions[1];
  local[2] = -dimensions[2];
  placedbox->GetTransformation()->InverseTransform(local, vert[3]);
  local[0] = -dimensions[0];
  local[1] = -dimensions[1];
  local[2] = dimensions[2];
  placedbox->GetTransformation()->InverseTransform(local, vert[4]);
  local[0] = -dimensions[0];
  local[1] = dimensions[1];
  local[2] = dimensions[2];
  placedbox->GetTransformation()->InverseTransform(local, vert[5]);
  local[0] = dimensions[0];
  local[1] = dimensions[1];
  local[2] = dimensions[2];
  placedbox->GetTransformation()->InverseTransform(local, vert[6]);
  local[0] = dimensions[0];
  local[1] = -dimensions[1];
  local[2] = dimensions[2];
  placedbox->GetTransformation()->InverseTransform(local, vert[7]);

  pl.SetPoint(0, vert[0].x(), vert[0].y(), vert[0].z());
  pl.SetPoint(1, vert[1].x(), vert[1].y(), vert[1].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, vert[1].x(), vert[1].y(), vert[1].z());
  pl.SetPoint(1, vert[2].x(), vert[2].y(), vert[2].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, vert[2].x(), vert[2].y(), vert[2].z());
  pl.SetPoint(1, vert[3].x(), vert[3].y(), vert[3].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, vert[3].x(), vert[3].y(), vert[3].z());
  pl.SetPoint(1, vert[0].x(), vert[0].y(), vert[0].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, vert[4].x(), vert[4].y(), vert[4].z());
  pl.SetPoint(1, vert[5].x(), vert[5].y(), vert[5].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, vert[5].x(), vert[5].y(), vert[5].z());
  pl.SetPoint(1, vert[6].x(), vert[6].y(), vert[6].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, vert[6].x(), vert[6].y(), vert[6].z());
  pl.SetPoint(1, vert[7].x(), vert[7].y(), vert[7].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, vert[7].x(), vert[7].y(), vert[7].z());
  pl.SetPoint(1, vert[4].x(), vert[4].y(), vert[4].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, vert[0].x(), vert[0].y(), vert[0].z());
  pl.SetPoint(1, vert[4].x(), vert[4].y(), vert[4].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, vert[1].x(), vert[1].y(), vert[1].z());
  pl.SetPoint(1, vert[5].x(), vert[5].y(), vert[5].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, vert[2].x(), vert[2].y(), vert[2].z());
  pl.SetPoint(1, vert[6].x(), vert[6].y(), vert[6].z());
  visualizer.AddLine(pl);
  pl.SetPoint(0, vert[3].x(), vert[3].y(), vert[3].z());
  pl.SetPoint(1, vert[7].x(), vert[7].y(), vert[7].z());
  visualizer.AddLine(pl);
}
#endif
void help()
{
  std::cout << "### Usage: VisualizeMultiUnion -test [T] -nsolids [S] -npoints [N] ###\n";
  std::cout << "                T: 0=Inside 1=DistanceToIn 2=DistanceToOut\n";
  std::cout << "                S: number of random boxes\n";
  std::cout << "                N: number of random staring points\n";
}

int main(int argc, char *argv[])
{
#ifndef VECGEOM_ENABLE_CUDA
  OPTION_INT(npoints, 1000000);
  // OPTION_INT(nrep, 4);
  OPTION_INT(test, -1);
  OPTION_DOUBLE(nsolids, 100);
  if (test < 0) {
    help();
    return 0;
  }

  constexpr double size = 10.;

  UnplacedBox worldUnplaced = UnplacedBox(2 * size, 2 * size, 2 * size);

  UnplacedMultiUnion multiunion;
  double sized = size * std::pow(0.5 / nsolids, 1. / 3.);
  for (size_t i = 0; i < nsolids; ++i) {
    Vector3D<double> pos(RNG::Instance().uniform(-size, size), RNG::Instance().uniform(-size, size),
                         RNG::Instance().uniform(-size, size));
    double sizernd = RNG::Instance().uniform(0.8 * sized, 1.2 * sized);
    Transformation3D trans(pos.x(), pos.y(), pos.z(), RNG::Instance().uniform(-180, 180),
                           RNG::Instance().uniform(-180, 180), RNG::Instance().uniform(-180, 180));
    trans.SetProperties();
    UnplacedBox *box = new UnplacedBox(sizernd, sizernd, sizernd);
    multiunion.AddNode(box, trans);
  }
  multiunion.Close();

  std::cout << "Benchmarking multi-union solid having " << nsolids << " random boxes\n";

#ifdef VECGEOM_ROOT
  Visualizer visualizer;
  // Visualize bounding box
  Vector3D<double> deltas = 0.5 * (multiunion.GetStruct().fMaxExtent - multiunion.GetStruct().fMinExtent);
  Vector3D<double> origin = 0.5 * (multiunion.GetStruct().fMaxExtent + multiunion.GetStruct().fMinExtent);
  SimpleBox box("bbox", deltas.x(), deltas.y(), deltas.z());
  visualizer.AddVolume(box, Transformation3D(origin.x(), origin.y(), origin.z()));
  for (size_t i = 0; i < multiunion.GetNumberOfSolids(); ++i) {
    AddComponentToVisualizer(multiunion.GetStruct(), i, visualizer);
  }
  TPolyMarker3D pm(npoints);
  pm.SetMarkerColor(kRed);
  pm.SetMarkerStyle(1);
  Vector3D<double> point, direction;
  double distance;
  for (int i = 0; i < npoints; ++i) {
    point.Set(RNG::Instance().uniform(-2 * size, 2 * size), RNG::Instance().uniform(-2 * size, 2 * size),
              RNG::Instance().uniform(-2 * size, 2 * size));
    direction = volumeUtilities::SampleDirection();
    switch (test) {
    case 0:
      if (multiunion.Inside(point) != vecgeom::EInside::kOutside) pm.SetNextPoint(point[0], point[1], point[2]);
      break;
    case 1:
      distance = multiunion.DistanceToIn(point, direction);
      if (distance < vecgeom::kInfLength) {
        point += distance * direction;
        VECGEOM_ASSERT(multiunion.Inside(point) != vecgeom::EInside::kOutside);
        pm.SetNextPoint(point[0], point[1], point[2]);
      }
      break;
    case 2:
      distance = multiunion.DistanceToOut(point, direction);
      if (distance > 0 && distance < vecgeom::kInfLength) {
        point += distance * direction;
        VECGEOM_ASSERT(!multiunion.Contains(point));
        pm.SetNextPoint(point[0], point[1], point[2]);
      }
      break;
    default:
      std::cout << "Error: unknown test " << test << std::endl;
      help();
      return 1;
    };
  }

  visualizer.AddPoints(pm);
  visualizer.Show();
#endif

  return 0;
#else
  return 0;
#endif
}
