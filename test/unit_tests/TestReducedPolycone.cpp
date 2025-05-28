/*
 * TestReducedPolycone.cpp
 *
 *  Created on: Sep 20, 2018
 *      Author: rasehgal
 */

#include <iostream>
#include "VecGeom/volumes/ReducedPolycone.h"
#include "VecGeom/volumes/UnplacedPolycone.h"
#include "VecGeom/base/Vector2D.h"
#include <iostream>
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/base/FpeEnable.h"

#undef NDEBUG

using namespace vecgeom;
bool TestReducedPolycone()
{

  Vector<Vector2D<Precision>> rzVect;
  bool contour = false;

  std::cout << "================================  (Trivial) ======================================" << std::endl;
  rzVect.push_back(Vector2D<Precision>(0., 0.));
  rzVect.push_back(Vector2D<Precision>(0., 1.));
  rzVect.push_back(Vector2D<Precision>(1., 1.));
  rzVect.push_back(Vector2D<Precision>(1., 2.));
  rzVect.push_back(Vector2D<Precision>(3., 2.));
  rzVect.push_back(Vector2D<Precision>(3., 1.));
  rzVect.push_back(Vector2D<Precision>(4., 1.));
  rzVect.push_back(Vector2D<Precision>(4., 0.));
  vecgeom::ReducedPolycone p(rzVect);
  contour = p.Check();
  VECGEOM_VALIDATE(contour, << "Trivial Fails....");

  std::cout << "========================= (Trivial with Mirror Image) ============================" << std::endl;
  rzVect.clear();

  rzVect.push_back(Vector2D<Precision>(0., 0.));
  rzVect.push_back(Vector2D<Precision>(0., 1.));
  rzVect.push_back(Vector2D<Precision>(1., 1.));
  rzVect.push_back(Vector2D<Precision>(1., 2.));
  rzVect.push_back(Vector2D<Precision>(3., 2.));
  rzVect.push_back(Vector2D<Precision>(3., 1.));
  rzVect.push_back(Vector2D<Precision>(4., 1.));
  rzVect.push_back(Vector2D<Precision>(4., 0.));
  rzVect.push_back(Vector2D<Precision>(4., -1.));
  rzVect.push_back(Vector2D<Precision>(3., -1.));
  rzVect.push_back(Vector2D<Precision>(3., -2.));
  rzVect.push_back(Vector2D<Precision>(1., -2.));
  rzVect.push_back(Vector2D<Precision>(1., -1.));
  rzVect.push_back(Vector2D<Precision>(0., -1.));
  p.SetRZ(rzVect);
  contour = p.Check();
  VECGEOM_VALIDATE(contour, << "Trivial with Mirror Image Fails....");

  std::cout << "==========  (Contour with only one point in Top and Bottom Z) ====================" << std::endl;
  rzVect.clear();

  rzVect.push_back(Vector2D<Precision>(0., 0.));
  rzVect.push_back(Vector2D<Precision>(1., 1.));
  rzVect.push_back(Vector2D<Precision>(1., 2.));
  rzVect.push_back(Vector2D<Precision>(2., 3.));
  rzVect.push_back(Vector2D<Precision>(3., 2.5));
  rzVect.push_back(Vector2D<Precision>(3., -1.));
  rzVect.push_back(Vector2D<Precision>(1., -2.));
  p.SetRZ(rzVect);
  contour = p.Check();
  VECGEOM_VALIDATE(contour, << "A2 Fails ....");

  std::cout << "============================== AnitclockWise Contour =============================" << std::endl;
  rzVect.clear();
  rzVect.push_back(Vector2D<Precision>(0., 0.));
  rzVect.push_back(Vector2D<Precision>(1., -1.));
  rzVect.push_back(Vector2D<Precision>(2., -1.));
  rzVect.push_back(Vector2D<Precision>(2., -2.));
  rzVect.push_back(Vector2D<Precision>(3., -2));
  rzVect.push_back(Vector2D<Precision>(3., -1.));
  rzVect.push_back(Vector2D<Precision>(4., -1.));
  rzVect.push_back(Vector2D<Precision>(4., 2.));
  rzVect.push_back(Vector2D<Precision>(3., 3.));
  p.SetRZ(rzVect);
  contour = p.Check();
  VECGEOM_VALIDATE(contour, << "AnitclockWise Contour Fails ....");

  std::cout << "============================== Triangular Spikes =================================" << std::endl;
  rzVect.clear();
  rzVect.push_back(Vector2D<Precision>(0., 0.));
  rzVect.push_back(Vector2D<Precision>(0., 2.));
  rzVect.push_back(Vector2D<Precision>(1., 4.));
  rzVect.push_back(Vector2D<Precision>(2., 2.));
  rzVect.push_back(Vector2D<Precision>(3., 5.));
  rzVect.push_back(Vector2D<Precision>(4., 2.));
  rzVect.push_back(Vector2D<Precision>(4., -1.));
  rzVect.push_back(Vector2D<Precision>(2., -2.));
  p.SetRZ(rzVect);
  contour = p.Check();
  VECGEOM_VALIDATE(!contour, << "Triangular Spikes Fails ....");

  std::cout << "============================== Square Spikes at End ==============================" << std::endl;
  rzVect.clear();
  rzVect.push_back(Vector2D<Precision>(0., 0.));
  rzVect.push_back(Vector2D<Precision>(0., 1.));
  rzVect.push_back(Vector2D<Precision>(1., 1.));
  rzVect.push_back(Vector2D<Precision>(1., 2.));
  rzVect.push_back(Vector2D<Precision>(2., 2.));
  rzVect.push_back(Vector2D<Precision>(2., 3.));
  rzVect.push_back(Vector2D<Precision>(4., 3.));
  rzVect.push_back(Vector2D<Precision>(4., 2.));
  rzVect.push_back(Vector2D<Precision>(6., 2.));
  rzVect.push_back(Vector2D<Precision>(6., 1.));
  rzVect.push_back(Vector2D<Precision>(4., 1.));
  rzVect.push_back(Vector2D<Precision>(4., -1.));
  rzVect.push_back(Vector2D<Precision>(3., -1.));
  rzVect.push_back(Vector2D<Precision>(3., -2.));
  rzVect.push_back(Vector2D<Precision>(2., -2.));
  rzVect.push_back(Vector2D<Precision>(2., -1.));
  rzVect.push_back(Vector2D<Precision>(1.5, -1.));
  rzVect.push_back(Vector2D<Precision>(1.5, -2.));
  rzVect.push_back(Vector2D<Precision>(0.5, -2.));
  rzVect.push_back(Vector2D<Precision>(0.5, -1.));
  rzVect.push_back(Vector2D<Precision>(0., -1.));
  p.SetRZ(rzVect);
  contour = p.Check();
  VECGEOM_VALIDATE(!contour, << "Square Spikes at End Fails ....");

  std::cout << "============================== Square Spikes in beginning ========================" << std::endl;
  rzVect.clear();
  rzVect.push_back(Vector2D<Precision>(0., 0.));
  rzVect.push_back(Vector2D<Precision>(0., 1.));
  rzVect.push_back(Vector2D<Precision>(1., 1.));
  rzVect.push_back(Vector2D<Precision>(1., 2.));
  rzVect.push_back(Vector2D<Precision>(2., 2.));
  rzVect.push_back(Vector2D<Precision>(2., 1.));
  rzVect.push_back(Vector2D<Precision>(3., 1.));
  rzVect.push_back(Vector2D<Precision>(3., 2.));
  rzVect.push_back(Vector2D<Precision>(4., 2.));
  rzVect.push_back(Vector2D<Precision>(4., 1.));
  rzVect.push_back(Vector2D<Precision>(5., 1.));
  rzVect.push_back(Vector2D<Precision>(5., 0.));
  p.SetRZ(rzVect);
  contour = p.Check();
  VECGEOM_VALIDATE(!contour, << "Square Spikes in beginning Fails ....");

  std::cout << "================================= PLUS symbole ===================================" << std::endl;
  rzVect.clear();
  rzVect.push_back(Vector2D<Precision>(0., 0.));
  rzVect.push_back(Vector2D<Precision>(0., 1.));
  rzVect.push_back(Vector2D<Precision>(1., 1.));
  rzVect.push_back(Vector2D<Precision>(1., 2.));
  rzVect.push_back(Vector2D<Precision>(3., 2.));
  rzVect.push_back(Vector2D<Precision>(3., 1.));
  rzVect.push_back(Vector2D<Precision>(4., 1.));
  rzVect.push_back(Vector2D<Precision>(4., -1.));
  rzVect.push_back(Vector2D<Precision>(3., -1.));
  rzVect.push_back(Vector2D<Precision>(3., -2.));
  rzVect.push_back(Vector2D<Precision>(1., -2.));
  rzVect.push_back(Vector2D<Precision>(1., -1.));
  rzVect.push_back(Vector2D<Precision>(0., -1.));
  p.SetRZ(rzVect);
  contour = p.Check();
  VECGEOM_VALIDATE(contour, << "PLUS symbole Fails ....");

  std::cout << "========================== Contour with only three points ========================" << std::endl;
  rzVect.clear();
  rzVect.push_back(Vector2D<Precision>(1., 1.));
  rzVect.push_back(Vector2D<Precision>(1., -1.));
  rzVect.push_back(Vector2D<Precision>(2., -1.));
  p.SetRZ(rzVect);
  contour = p.Check();
  VECGEOM_VALIDATE(contour, << "X symbole (X) Fails ....");

  std::cout << "================================= X symbole ======================================" << std::endl;
  rzVect.clear();
  rzVect.push_back(Vector2D<Precision>(1., 3.));
  rzVect.push_back(Vector2D<Precision>(1., 1.));
  rzVect.push_back(Vector2D<Precision>(1., -1.));
  rzVect.push_back(Vector2D<Precision>(2., -1.));
  rzVect.push_back(Vector2D<Precision>(1., 1.));
  rzVect.push_back(Vector2D<Precision>(2., 3.));
  p.SetRZ(rzVect);
  contour = p.Check();
  VECGEOM_VALIDATE(!contour, << "X2 symbole (X2) Fails ....");

  std::cout << "======================  (Trivial Using UnplacedPolycone) =========================" << std::endl;
  const int numRZ    = 10;
  Precision r[numRZ] = {0., 0., 0., 1., 1., 3., 3., 4., 4., 0.};
  Precision z[numRZ] = {0., 1., 1., 1., 2., 2., 1., 1., 0., 0.};
  auto poly          = GeoManager::MakeInstance<UnplacedPolycone>(0., kTwoPi, numRZ, r, z);
  std::cout << "SurfaceArea : " << poly->SurfaceArea() << std::endl;

  const int numRZ1       = 7;
  Precision polycone_r[] = {0., 1., 1., 2., 3., 3., 1.};
  Precision polycone_z[] = {0., 1., 2., 3., 2.5, -1., -2.};
  auto poly1             = GeoManager::MakeInstance<UnplacedPolycone>(0., kTwoPi, numRZ1, polycone_r, polycone_z);
  std::cout << "SurfaceArea : " << poly1->SurfaceArea() << std::endl;
  std::cout << "Contour : " << contour << std::endl;
  return true;
}

int main()
{

  TestReducedPolycone();
  std::cout << "**********************************************************************************" << std::endl;
  std::cout << "Reduced Polycone Passed..." << std::endl;
  std::cout << "**********************************************************************************" << std::endl;
}
