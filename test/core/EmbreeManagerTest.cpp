/*
 * EmbreeManagerTest.cpp
 *
 *  Created on: May 18, 2018
 *      Author: swenzel
 */

#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/management/EmbreeManager.h"
#include "memory.h" // for unique ptr
#include <iostream>
#include <embree3/rtcore_ray.h>

using namespace vecgeom;

int g_x;
double DistanceToIn(RTCScene scene, double x, double y, double z, double dx, double dy, double dz, int &geomID,
                    bool &inside)
{
  RTCRayHit ray; // EMBREE_FIXME: use RTCRay for occlusion rays
  ray.ray.flags = 0;
  ray.ray.org_x = x;
  ray.ray.org_y = y;
  ray.ray.org_z = z;
  ray.ray.dir_x = dx;
  ray.ray.dir_y = dy;
  ray.ray.dir_z = dz;
  ray.ray.tnear = 0.;
  ray.ray.tfar  = 1E20f;
  {
    RTCIntersectContext context;

    g_x = x;

    // we can't do a real capture but we do it via some global variables
    auto customFilter = [](const RTCFilterFunctionNArguments *args) {
      VECGEOM_ASSERT(args->N == 1);
      std::cerr << "CUSTOM FILTER USED " << g_x << "\n";
      int *valid = args->valid;
      valid[0]   = 0;
    };

    rtcInitIntersectContext(&context);
    context.filter = customFilter;
    rtcIntersect1(scene, &context, &ray);
    ray.hit.Ng_x = -ray.hit.Ng_x;
    ray.hit.Ng_y = -ray.hit.Ng_y;
    ray.hit.Ng_z = -ray.hit.Ng_z;
  }
  geomID = ray.hit.geomID;
  return ray.ray.tfar;
}

void calcLowerUpper(Vector3D<double> origin, double s, Vector3D<double> &lower, Vector3D<double> &upper)
{
  lower = Vector3D<double>(origin.x() - s, origin.y() - s, origin.z() - s);
  upper = Vector3D<double>(origin.x() + s, origin.y() + s, origin.z() + s);
}

int main()
{
  int nObjects = 2;
  int nBoxes   = 2 * nObjects;

  // setup a simple bounding box by defining lower/upper corners
  auto boxcontainer = std::make_unique<EmbreeManager::ABBox_s[]>(nBoxes);

  // displacement + size
  for (int i = 0; i < nObjects; ++i) {
    Vector3D<double> origin(5 * i, 0, 0);
    Vector3D<double> lower, upper;
    calcLowerUpper(origin, 1, lower, upper);
    boxcontainer[2 * i]     = lower;
    boxcontainer[2 * i + 1] = upper;
  }

  // this yields a structure which we can use for rayTracing with Embree
  auto structure = EmbreeManager::Instance().BuildStructureFromBoundingBoxes(boxcontainer.get(), nObjects);

  int geomID  = -1;
  bool inside = false;
  std::cerr << DistanceToIn(structure->fScene, -20, 0, 0, 1, 0, 0, geomID, inside) << "\n";
  std::cerr << "GEOMID " << geomID << "\n";
  std::cerr << DistanceToIn(structure->fScene, 0, -20, 0, 0, 1, 0, geomID, inside) << "\n";
  std::cerr << "GEOMID " << geomID << "\n";
  std::cerr << DistanceToIn(structure->fScene, 0, 0, -20, 0, 0, 1, geomID, inside) << "\n";
  std::cerr << "GEOMID " << geomID << "\n";
  std::cerr << DistanceToIn(structure->fScene, 20, 0, 0, -1, 0, 0, geomID, inside) << "\n";
  std::cerr << "GEOMID " << geomID << "\n";
  std::cerr << DistanceToIn(structure->fScene, 0, 20, 0, 0, -1, 0, geomID, inside) << "\n";
  std::cerr << "GEOMID " << geomID << "\n";
  std::cerr << DistanceToIn(structure->fScene, 0, 0, 20, 0, 0, -1, geomID, inside) << "\n";
  std::cerr << "GEOMID " << geomID << "\n";
  std::cerr << "inside " << inside << "\n";

  // going away
  std::cerr << DistanceToIn(structure->fScene, 0, 0, 20, 0, 0, 1, geomID, inside) << "\n";
  std::cerr << "GEOMID " << geomID << "\n";

  // from inside
  std::cerr << DistanceToIn(structure->fScene, 0, 0, 0, 0, 0, 1, geomID, inside) << "\n";
  std::cerr << "GEOMID " << geomID << "\n";
  std::cerr << "inside " << inside << "\n";

  // use normal to see if inside / outside of geometry

  return 0;
}
