///
/// Simple visualizer of volumes meshed with marching cubes
///

/// Examples:
///
/// $ TestMarchingCubes -v orb -n 10 -p 20
/// $ TestMarchingCubes -v polycone -n 50 -p 0 360 4  5 10 0  5 10 5  10 15 15  10 15 20
///

//-- ensure asserts are compiled in
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "VecGeom/base/Utils3D.h"
#include "ApproxEqual.h"
#include "VecGeom/volumes/Box.h"
#include "test/benchmark/ArgParser.h"

#include "VecGeom/volumes/SolidMesh.h"
#include "VecGeom/volumes/MarchingCubes.h"
#include "VecGeom/volumes/UnplacedParallelepiped.h"
#include "VecGeom/volumes/UnplacedTrapezoid.h"
#include "VecGeom/volumes/UnplacedTet.h"
#include "VecGeom/volumes/UnplacedTrd.h"
#include "VecGeom/management/GeoManager.h"
#include "VecGeom/volumes/UnplacedSExtruVolume.h"
#include "VecGeom/volumes/UnplacedEllipticalTube.h"
#include "VecGeom/volumes/UnplacedEllipticalCone.h"
#include "VecGeom/volumes/UnplacedOrb.h"
#include "VecGeom/volumes/UnplacedParaboloid.h"
#include "VecGeom/volumes/UnplacedCone.h"
#include "VecGeom/volumes/UnplacedParaboloid.h"
#include "VecGeom/volumes/UnplacedPolycone.h"
#include "VecGeom/volumes/UnplacedPolyhedron.h"
#include "VecGeom/volumes/UnplacedGenTrap.h"
#include "VecGeom/volumes/UnplacedEllipsoid.h"
#include "VecGeom/volumes/UnplacedCutTube.h"
#include "VecGeom/volumes/UnplacedTube.h"
#include "VecGeom/volumes/UnplacedTorus2.h"
#include "VecGeom/volumes/UnplacedHype.h"
#include "VecGeom/volumes/UnplacedSphere.h"
#ifndef VECCORE_CUDA
#include "VecGeom/volumes/UnplacedExtruded.h"
#endif

#ifdef VECGEOM_ROOT
#include "VecGeomTest/Visualizer.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"
#endif

using vecgeom::Precision;

#ifdef VECGEOM_ROOT
void DrawPolygon(vecgeom::Utils3D::Polygon const &poly, vecgeom::Visualizer &visualizer, size_t color)
{
  using namespace vecgeom;
  using Vec_t = Vector3D<Precision>;
  TPolyLine3D pl(poly.fN + 1);
  pl.SetLineColor(color);
  for (size_t i = 0; i < poly.fN; ++i)
    pl.SetNextPoint(poly.GetVertex(i).x(), poly.GetVertex(i).y(), poly.GetVertex(i).z());
  pl.SetNextPoint(poly.GetVertex(0).x(), poly.GetVertex(0).y(), poly.GetVertex(0).z());
  visualizer.AddLine(pl);

  // The code below draws normals and clutters the visualization
  // of meshes made with marching cubes. Uncomment the line below to restore it.
  return;

  // Compute center of polygon
  Vec_t center;
  for (size_t i = 0; i < poly.fN; ++i)
    center += poly.GetVertex(i);
  center *= 1. / poly.fN;
  TPolyLine3D plnorm(2);
  plnorm.SetLineColor(color);
  plnorm.SetNextPoint(center[0], center[1], center[2]);
  plnorm.SetNextPoint(center[0] + poly.fNorm[0], center[1] + poly.fNorm[1], center[2] + poly.fNorm[2]);
  visualizer.AddLine(plnorm);
}

void DrawPolyhedron(const vecgeom::Utils3D::Polyhedron &polyh, vecgeom::Visualizer &visualizer, size_t color)
{
  using namespace vecgeom;

  for (size_t i = 0; i < polyh.GetNpolygons(); ++i)
    DrawPolygon(polyh.GetPolygon(i), visualizer, color);
}
#endif

vecgeom::VUnplacedVolume *CreateSexTru(bool convex)
{
#define N 10
  Precision dx = 5;
  Precision dy = 4;
  Precision dz = 3;

  Precision x[N], y[N];
  for (size_t i = 0; i < (size_t)N; ++i) {
    x[i] = dx * std::sin(i * (2. * M_PI) / N);
    y[i] = dy * std::cos(i * (2. * M_PI) / N);
  }
  if (!convex) {
    x[0] = 0;
    y[1] = 0;
  }
  return vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedSExtruVolume>(N, x, y, -dz, dz);
}

#ifndef VECCORE_CUDA
vecgeom::VUnplacedVolume *CreateExtruded(bool convex)
{
#define nvert 10
#define nsect 5

  Precision rmin = 3.;
  Precision rmax = 5.;

  vecgeom::XtruVertex2 *vertices = new vecgeom::XtruVertex2[nvert];
  vecgeom::XtruSection *sections = new vecgeom::XtruSection[nsect];
  Precision *x                   = new Precision[nvert];
  Precision *y                   = new Precision[nvert];

  Precision phi = 2. * vecgeom::kPi / nvert;
  Precision r;
  for (int i = 0; i < nvert; ++i) {
    r = rmax;
    if (i % 2 > 0 && !convex) r = rmin;
    vertices[i].x = r * vecCore::math::Cos(i * phi);
    vertices[i].y = r * vecCore::math::Sin(i * phi);
    x[i]          = vertices[i].x;
    y[i]          = vertices[i].y;
  }
  for (int i = 0; i < nsect; ++i) {
    sections[i].fOrigin.Set(0, 0, -2. + i * 4. / (nsect - 1));
    sections[i].fScale = 1;
    if (i == 0)
      sections[0].fScale = 0.5;
    else if (i == nsect - 1)
      sections[nsect - 1].fScale = 0.5;
  }

  return vecgeom::GeoManager::MakeInstance<vecgeom::UnplacedExtruded>(nvert, vertices, nsect, sections);
}
#endif

void CreateConcave(std::vector<vecgeom::Vector3D<Precision>> &v, vecgeom::Utils3D::Polygon &p)
{
  using Vec_t = vecgeom::Vector3D<Precision>;

  v.push_back(Vec_t(2, 2, 0));
  v.push_back(Vec_t(-2, 2, 0));
  v.push_back(Vec_t(-2, -2, 0));
  v.push_back(Vec_t(0, 0, 0));
  v.push_back(Vec_t(2, -2, 0));

  p.fInd = {v.size() - 5, v.size() - 4, v.size() - 3, v.size() - 2, v.size() - 1};

  p.Init();
}

void print_msg()
{
  std::cout << "\nUsage:\n"
               "./TestMarchingCubes -v [string] -n [int] -p [float float...]\n"
               "Specify all angles in degrees. Any number of spaces between the parameters allowed. -n is the "
               "nSegments for meshes, and nPoints for PIP and PIT. \n"

               "\nAvailable volumes:\n"
               " \"box\": dx dy dz \n"
               " \"parallelepiped\": dx dy dz alpha theta phi \n"
               " \"trd\": x1 x2 y1 y2 z \n"
               " \"trapezoid\": dz theta phi dy1 dx1 dx2 alpha1 dy2 dx3 dx4 alpha2 \n"
               " \"tet\": x1 y1 z1  x2 y2 z2  x3 y3 z3  x4 y4 z4 \n\n"

               " \"ellipticaltube\": dx dy dz \n"
               " \"ellipticalcone\": a b h zcut \n"
               " \"orb\": r \n"
               " \"cone\": rmin1 rmax1 rmin2 rmax2 dz sphi dphi \n"
               " \"tube\": rmin rmax z sphi dphi \n"
               " \"cuttube\": rmin rmax z sphi dphi x1 y1 z1(bottomNormal) x2 y2 z2(topNormal) \n\n"

               " \"paraboloid\": rlo rhi dz \n"
               " \"hype\": rmin rmax stIn stOut dz \n"
               " \"torus\": rmin rmax rtor sphi dphi \n"
               " \"ellipsoid\": dx dy dz zBottomCut zTopCut \n"
               " \"sphere\": rmin rmax sPhi dPhi sTheta dTheta \n\n"

               " \"polycone\": sPhi dPhi nPlanes  rmin_1 rmax_1 z_1 ... rmin_n rmax_n z_n \n"
               " \"polyhedron\": sPhi dPhi sideCount nPlanes rin_1 rout_1 z_1 ... rin_n rout_n z_n \n"
               " \"gentrap\": dz  x1 y1 ... x8 y8 \n\n"

               " \"sextruvolume\": -1 for concave +1 for convex \n"
               " \"extruded\": -1 for concave +1 for convex \n\n"

               " \"polyx\": -2 -1 .. 22\n"
               " \"polytri\": \n"
               " \"pip\": \n"
               " \"pit\": \n"

               "\n"
               "\n"
               "";
}

Precision random_double(Precision min, Precision max) { return ((Precision)rand() / RAND_MAX) * (max - min) + min; }

int main(int argc, char *argv[])
{
  using namespace vecgeom;
  using namespace vecCore::math;
  using namespace std;

  using vecgeom::Utils3D::Line;
  using vecgeom::Utils3D::Plane;
  using vecgeom::Utils3D::Polygon;
  using vecgeom::Utils3D::Polyhedron;

  string flag;
  if (argc == 1 || argc == 2 || argc == 3 || argc == 4 || (flag = argv[1]) != "-v" || (flag = argv[3]) != "-n") {
    print_msg();
    return 0;
  }

  string v = argv[2];

  if (!(!v.compare("pip") || !v.compare("pit")) && argc <= 6) {
    print_msg();
    return 0;
  }

  vector<Precision> ps;
  for (int i = 6; i < argc; i++) {
    ps.push_back(stod(string(argv[i]), NULL));
  }

#ifdef VECGEOM_ROOT
#define WORLDSIZE 10
  int nSegments                   = stod(string(argv[4]), NULL);
  VUnplacedVolume *unplacedvolume = nullptr;
  using Vec_t                     = Vector3D<Precision>;

  if (!v.compare("box")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedBox>(ps[0], ps[1], ps[2]);
  } else if (!v.compare("parallelepiped")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedParallelepiped>(ps[0], ps[1], ps[2], kDegToRad * ps[3],
                                                                      kDegToRad * ps[4], kDegToRad * ps[5]);
  } else if (!v.compare("trapezoid")) {
    // unplacedvolume = GeoManager::MakeInstance<UnplacedTrapezoid>(5., 0., 0., 3., 4., 5., 0., 3., 4., 5., 0.);
    unplacedvolume =
        GeoManager::MakeInstance<UnplacedTrapezoid>(ps[0], kDegToRad * ps[1], kDegToRad * ps[2], ps[3], ps[4], ps[5],
                                                    kDegToRad * ps[6], ps[7], ps[8], ps[9], kDegToRad * ps[10]);
  } else if (!v.compare("trd")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedTrd>(2., 5., 2., 5., 5.);
  } else if (!v.compare("tet")) {
    // Vec_t p0(0., 0., 5.), p1(-5., -5., -5.), p2(5., -5., -5.), p3(-5., 5., -5.);
    // unplacedvolume = GeoManager::MakeInstance<UnplacedTet>(p0, p1, p2, p3);
    Vec_t p0(ps[0], ps[1], ps[2]), p1(ps[3], ps[4], ps[5]), p2(ps[6], ps[7], ps[8]), p3(ps[9], ps[10], ps[11]);
    unplacedvolume = GeoManager::MakeInstance<UnplacedTet>(p0, p1, p2, p3);
  } else if (!v.compare("ellipticaltube")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedEllipticalTube>(ps[0], ps[1], ps[2]);
  } else if (!v.compare("ellipticalcone")) {
    // 1 1 5 3
    unplacedvolume = GeoManager::MakeInstance<UnplacedEllipticalCone>(ps[0], ps[1], ps[2], ps[3]);
  } else if (!v.compare("orb")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedOrb>(ps[0]);
  } else if (!v.compare("cone")) {
    // unplacedvolume = GeoManager::MakeInstance<UnplacedCone>(3., 5, 2., 4., 5, 0, kPi);
    unplacedvolume =
        GeoManager::MakeInstance<UnplacedCone>(ps[0], ps[1], ps[2], ps[3], ps[4], kDegToRad * ps[5], kDegToRad * ps[6]);

  } else if (!v.compare("paraboloid")) {
    // 0 7 5
    unplacedvolume = GeoManager::MakeInstance<UnplacedParaboloid>(ps[0], ps[1], ps[2]);
  } else if (!v.compare("ellipsoid")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedEllipsoid>(ps[0], ps[1], ps[2], ps[3], ps[4]);
  } else if (!v.compare("cuttube")) {
    // 1 0 - 1       1 0 1 normals
    unplacedvolume =
        GeoManager::MakeInstance<UnplacedCutTube>(ps[0], ps[1], ps[2], kDegToRad * ps[3], kDegToRad * ps[4],
                                                  Vec_t(ps[5], ps[6], ps[7]), Vec_t(ps[8], ps[9], ps[10]));
  } else if (!v.compare("tube")) {
    // unplacedvolume = GeoManager::MakeInstance<UnplacedTube>(2., 10., 5., 0., 2*kPi);
    unplacedvolume = GeoManager::MakeInstance<UnplacedTube>(ps[0], ps[1], ps[2], kDegToRad * ps[3], kDegToRad * ps[4]);
  } else if (!v.compare("torus")) {
    unplacedvolume =
        GeoManager::MakeInstance<UnplacedTorus2>(ps[0], ps[1], ps[2], kDegToRad * ps[3], kDegToRad * ps[4]);
  } else if (!v.compare("hype")) {
    //(const Precision rMin, const Precision rMax, const Precision stIn, const Precision stOut, const Precision dz)
    unplacedvolume = GeoManager::MakeInstance<UnplacedHype>(ps[0], ps[1], kDegToRad * ps[2], kDegToRad * ps[3], ps[4]);
  } else if (!v.compare("sphere")) {
    unplacedvolume = GeoManager::MakeInstance<UnplacedSphere>(ps[0], ps[1], kDegToRad * ps[2], kDegToRad * ps[3],
                                                              kDegToRad * ps[4], kDegToRad * ps[5]);
  } else if (!v.compare("sextruvolume")) {
    unplacedvolume = ps[0] > 0 ? CreateSexTru(true) : CreateSexTru(false);
  }
#ifndef VECCORE_CUDA
  else if (!v.compare("extruded")) {
    unplacedvolume = ps[0] > 0 ? CreateExtruded(true) : CreateExtruded(false);
  }
#endif

  else if (!v.compare("polycone")) {
    /*
    Precision rmin[]  = {0., 3., 7., 8};
    Precision rmax[]  = {5., 4., 8., 9};
    Precision z[]     = {-3, 0, 3, 8};
    */

    int Nz          = ps[2];
    Precision *rmin = new Precision[Nz];
    Precision *rmax = new Precision[Nz];
    Precision *z    = new Precision[Nz];

    size_t idx = 0;
    for (size_t i = 3; i < ps.size(); i += 3) {
      rmin[idx] = ps[i];
      rmax[idx] = ps[i + 1];
      z[idx]    = ps[i + 2];
      idx++;
    }
    unplacedvolume =
        GeoManager::MakeInstance<UnplacedPolycone>(kDegToRad * ps[0], kDegToRad * ps[1], Nz, z, rmin, rmax);
    delete[] rmin, delete[] rmax, delete[] z;
  } else if (!v.compare("polyhedron")) {

    /*
    Precision zPlanes[nPlanes] = {-7, -1, -1, 1, 7};
    Precision rInner[nPlanes]  = {3, 4, 5, 5, 7};
    Precision rOuter[nPlanes]  = {8, 9, 9, 9, 10};
    */
    int nPlanes        = ps[3];
    Precision *rInner  = new Precision[nPlanes];
    Precision *rOuter  = new Precision[nPlanes];
    Precision *zPlanes = new Precision[nPlanes];
    size_t idx         = 0;
    for (size_t i = 4; i < ps.size(); i += 3) {
      rInner[idx]  = ps[i];
      rOuter[idx]  = ps[i + 1];
      zPlanes[idx] = ps[i + 2];
      idx++;
    }

    unplacedvolume = GeoManager::MakeInstance<UnplacedPolyhedron>(kDegToRad * ps[0], kDegToRad * ps[1], ps[2], nPlanes,
                                                                  zPlanes, rInner, rOuter);
    delete[] rInner, delete[] rOuter, delete[] zPlanes;
  } else if (!v.compare("gentrap")) {
    // twisted

    /*
      Precision verticesx[8] = {-3, -3, 3, 3, -3.889087296526, 0.35355339059327, 3.889087296526, -0.35355339059327};
      Precision verticesy[8] = {-3, 3, 3, -3, 0.35355339059327, 3.889087296526, -0.35355339059327, -3.889087296526};
*/
    // no twist
    // Precision verticesx1[8] = {-3, -3, 3, 3, -2, -2, 2, 2};
    // Precision verticesy1[8] = {-3, 3, 3, -3, -2, 2, 2, -2};

    Precision verticesx[8];
    Precision verticesy[8];
    size_t idx = 0;
    for (size_t i = 1; i < ps.size(); i += 2) {
      verticesx[idx] = ps[i];
      verticesy[idx] = ps[i + 1];
      idx++;
    }

    unplacedvolume = GeoManager::MakeInstance<UnplacedGenTrap>(verticesx, verticesy, ps[0]);
  } else if (!v.compare("polytri")) {
    Visualizer visualizer;
    SimpleBox boxshape("box", 3, 3, 3);
    visualizer.AddVolume(boxshape);

    std::vector<Vec_t> v = {{2, 2, 0}, {-2, 2, 0}, {-2, -2, 0}, {0, 0, 0}, {2, -2, 0},
                            {2, 2, 1}, {-2, 2, 1}, {-2, -2, 1}, {0, 0, 1}, {2, -2, 1}};

    Utils3D::Polygon p{5, v, false};
    p.fInd = {0, 1, 2, 3, 4};
    p.Init();

    std::vector<Polygon> polys;
    p.TriangulatePolygon(polys);
    for (auto poly : polys) {
      poly.Init();
      DrawPolygon(poly, visualizer, kRed);
    }

    Utils3D::Polygon p2{5, v, false};
    p2.fInd = {5, 6, 7, 8, 9};
    p2.Init();
    DrawPolygon(p2, visualizer, kBlue);

    visualizer.Show();
    return 0;
  } else if (!v.compare("polyx")) {

    std::vector<Vec_t> v;
    vecgeom::Utils3D::Polygon p{4, v, false};
    vecgeom::Utils3D::Polygon p2{4, v, false};

    int test = ps[0];
    if (test == -2) {
      p = {4, v, false};
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3};
      p.Init();
      v.push_back(Vec_t(3, 2, 0));
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(2, -2, 0));
      v.push_back(Vec_t(3, -2, 0));

      p2.fInd = {4, 5, 6, 7};

      p2.Init();
    } else if (test == -1) {
      p = {5, v, false};
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(0, 0, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3, 4};
      p.Init();
      v.push_back(Vec_t(1, -0.5, 0));
      v.push_back(Vec_t(-1, -0.5, 0));
      v.push_back(Vec_t(-1, -1.5, 0));
      v.push_back(Vec_t(1, -1.5, 0));

      p2.fInd = {5, 6, 7, 8};

      p2.Init();
    } else if (test == 0) { // line
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3};
      p.Init();
      v.push_back(Vec_t(4, 2, 0));
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(2, -2, 0));
      v.push_back(Vec_t(4, -2, 0));
      p2.fInd = {4, 5, 6, 7};
      p2.Init();
    } else if (test == 1) { // point
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3};
      p.Init();
      v.push_back(Vec_t(-4, 6, 0));
      v.push_back(Vec_t(-4, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, 6, 0));

      p2.fInd = {4, 5, 6, 7};

      p2.Init();
    } else if (test == 2) { // region
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3};
      p.Init();
      v.push_back(Vec_t(4, 2, 0));
      v.push_back(Vec_t(0, 2, 0));
      v.push_back(Vec_t(0, -2, 0));
      v.push_back(Vec_t(4, -2, 0));

      p2.fInd = {4, 5, 6, 7};

      p2.Init();
    } else if (test == 3) {
      p = {5, v, false};
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(0, 0, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3, 4};
      p.Init();
      v.push_back(Vec_t(4, 2, 0));
      v.push_back(Vec_t(0, 2, 0));
      v.push_back(Vec_t(0, -2, 0));
      v.push_back(Vec_t(4, -2, 0));

      p2.fInd = {5, 6, 7, 8};

      p2.Init();
    } else if (test == 4) { // inside
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3};
      p.Init();
      v.push_back(Vec_t(1, 1, 0));
      v.push_back(Vec_t(-1, 1, 0));
      v.push_back(Vec_t(-1, -1, 0));
      v.push_back(Vec_t(1, -1, 0));

      p2.fInd = {4, 5, 6, 7};

      p2.Init();
    } else if (test == 5) { // no itnersection
      v.push_back(Vec_t(1, 1, 0));
      v.push_back(Vec_t(-1, 1, 0));
      v.push_back(Vec_t(-1, -1, 0));
      v.push_back(Vec_t(1, -1, 0));
      p.fInd = {0, 1, 2, 3};
      p.Init();
      v.push_back(Vec_t(2.1, 3.1, 0));
      v.push_back(Vec_t(1.1, 3.1, 0));
      v.push_back(Vec_t(1.1, 2.1, 0));
      v.push_back(Vec_t(2.1, 2.1, 0));

      p2.fInd = {4, 5, 6, 7};

      p2.Init();
    } else if (test == 6) { // no int 2
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3};
      p.Init();
      v.push_back(Vec_t(1, 1, 1));
      v.push_back(Vec_t(-1, 1, 1));
      v.push_back(Vec_t(-1, -1, 1));
      v.push_back(Vec_t(1, -1, 1));

      p2.fInd = {4, 5, 6, 7};

      p2.Init();
    } else if (test == 7) {
      p = {5, v, false};
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(0, 0, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3, 4};
      p.Init();
      v.push_back(Vec_t(1, 0.5, 0));
      v.push_back(Vec_t(-1, 0.5, 0));
      v.push_back(Vec_t(-1, -1, 0));
      v.push_back(Vec_t(1, -1, 0));

      p2.fInd = {5, 6, 7, 8};

      p2.Init();
    } else if (test == 8) {
      p = {5, v, false};
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(0, 0, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3, 4};
      p.Init();
      v.push_back(Vec_t(1, -1, 0));
      v.push_back(Vec_t(-1, -1, 0));
      v.push_back(Vec_t(-1, -1.5, 0));
      v.push_back(Vec_t(1, -1.5, 0));

      p2.fInd = {5, 6, 7, 8};

      p2.Init();
    } else if (test == 9) {
      p = {5, v, false};
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(0, 0, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3, 4};
      p.Init();
      v.push_back(Vec_t(1.5, -1, 0));
      v.push_back(Vec_t(-0.5, -1, 0));
      v.push_back(Vec_t(-0.5, -1.5, 0));
      v.push_back(Vec_t(1.5, -1.5, 0));

      p2.fInd = {5, 6, 7, 8};

      p2.Init();
    } else if (test == 10) {
      p = {5, v, false};
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(0, 0, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3, 4};
      p.Init();
      v.push_back(Vec_t(1.5, -0.5, 0));
      v.push_back(Vec_t(-0.5, -0.5, 0));
      v.push_back(Vec_t(-0.5, -1.5, 0));
      v.push_back(Vec_t(1.5, -1.5, 0));

      p2.fInd = {5, 6, 7, 8};

      p2.Init();
    } else if (test == 11) {
      p = {5, v, false};
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(0, 0, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3, 4};
      p.Init();
      v.push_back(Vec_t(1.5, -0.5, 0));
      v.push_back(Vec_t(-0.6, -0.5, 0));
      v.push_back(Vec_t(-0.6, -1.5, 0));
      v.push_back(Vec_t(1.5, -1.5, 0));

      p2.fInd = {5, 6, 7, 8};

      p2.Init();
    } else if (test == 12) {
      v.push_back(Vec_t(1.5, -0.5, 0));
      v.push_back(Vec_t(-0.6, -0.5, 0));
      v.push_back(Vec_t(-0.6, -1.5, 0));
      v.push_back(Vec_t(1.5, -1.5, 0));
      p.fInd = {0, 1, 2, 3};
      p.Init();
      v.push_back(Vec_t(1.5, -0.5, 0));
      v.push_back(Vec_t(-0.6, -0.5, 0));
      v.push_back(Vec_t(-0.6, -1.5, 0));
      v.push_back(Vec_t(1.5, -1.5, 0));

      p2.fInd = {4, 5, 6, 7};

      p2.Init();
    } else if (test == 13) {
      v.push_back(Vec_t(1, -1, 0));
      v.push_back(Vec_t(1, 1, 0));
      v.push_back(Vec_t(-1, 1, 0));
      v.push_back(Vec_t(-1, -1, 0));
      p.fInd = {0, 1, 2, 3};
      p.Init();
      v.push_back(Vec_t(0, -1, 1));
      v.push_back(Vec_t(0, -1, -1));
      v.push_back(Vec_t(0, 1, -1));
      v.push_back(Vec_t(0, 1, 1));

      p2.fInd = {4, 5, 6, 7};

      p2.Init();
    } else if (test == 14) {
      v.push_back(Vec_t(2, -1, 0));
      v.push_back(Vec_t(2, 1, 0));
      v.push_back(Vec_t(-2, 1, 0));
      v.push_back(Vec_t(-2, -1, 0));
      p.fInd = {0, 1, 2, 3};
      p.Init();
      v.push_back(Vec_t(0, -0.5, 0.5));
      v.push_back(Vec_t(0, -0.5, -0.5));
      v.push_back(Vec_t(0, 0.5, -0.5));
      v.push_back(Vec_t(0, 0.5, 0.5));

      p2.fInd = {4, 5, 6, 7};

      p2.Init();
    }

    else if (test == 15) {
      v.push_back(Vec_t(2, -1, 0));
      v.push_back(Vec_t(2, 1, 0));
      v.push_back(Vec_t(0, 1, 0));
      v.push_back(Vec_t(0, -1, 0));
      p.fInd = {0, 1, 2, 3};
      p.Init();
      v.push_back(Vec_t(0, -0.5, 0.5));
      v.push_back(Vec_t(0, -0.5, -0.5));
      v.push_back(Vec_t(0, 0.5, -0.5));
      v.push_back(Vec_t(0, 0.5, 0.5));

      p2.fInd = {4, 5, 6, 7};

      p2.Init();
    }

    else if (test == 16) {
      v.push_back(Vec_t(2, -1, 0));
      v.push_back(Vec_t(2, 1, 0));
      v.push_back(Vec_t(0, 1, 0));
      v.push_back(Vec_t(0, -1, 0));
      p.fInd = {0, 1, 2, 3};
      p.Init();
      v.push_back(Vec_t(0, -2, 0.5));
      v.push_back(Vec_t(0, -2, -0.5));
      v.push_back(Vec_t(0, -1, -0.5));
      v.push_back(Vec_t(0, -1, 0.5));

      p2.fInd = {4, 5, 6, 7};

      p2.Init();
    } else if (test == 17) {
      p = {5, v, false};
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(0, 0, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3, 4};
      p.Init();
      v.push_back(Vec_t(0, -2, 0.5));
      v.push_back(Vec_t(0, -2, -0.5));
      v.push_back(Vec_t(0, -1, -0.5));
      v.push_back(Vec_t(0, -1, 0.5));

      p2.fInd = {5, 6, 7, 8};

      p2.Init();
    }

    else if (test == 18) {
      p = {5, v, false};
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(0, 0, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3, 4};
      p.Init();
      v.push_back(Vec_t(2, 0, 2));
      v.push_back(Vec_t(-2, 0, 2));
      v.push_back(Vec_t(-2, 0, -2));
      v.push_back(Vec_t(2, 0, -2));

      p2.fInd = {5, 6, 7, 8};

      p2.Init();
    } else if (test == 19) {
      p = {5, v, false};
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(0, 0, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3, 4};
      p.Init();
      v.push_back(Vec_t(2, -2, 2));
      v.push_back(Vec_t(-2, -2, 2));
      v.push_back(Vec_t(-2, -2, -2));
      v.push_back(Vec_t(2, -2, -2));

      p2.fInd = {5, 6, 7, 8};

      p2.Init();
    }

    else if (test == 20) {
      p = {5, v, false};
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(0, 0, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3, 4};
      p.Init();
      v.push_back(Vec_t(2, 0, 2));
      v.push_back(Vec_t(-2, -2, 2));
      v.push_back(Vec_t(-2, -2, -2));
      v.push_back(Vec_t(2, 0, -2));

      p2.fInd = {5, 6, 7, 8};

      p2.Init();
    } else if (test == 21) {
      p = {5, v, false};
      v.push_back(Vec_t(2, 2, 0));
      v.push_back(Vec_t(-2, 2, 0));
      v.push_back(Vec_t(-2, -2, 0));
      v.push_back(Vec_t(0, 0, 0));
      v.push_back(Vec_t(2, -2, 0));
      p.fInd = {0, 1, 2, 3, 4};
      p.Init();
      v.push_back(Vec_t(2, 2, 2));
      v.push_back(Vec_t(-2, -2, 2));
      v.push_back(Vec_t(-2, -2, -2));
      v.push_back(Vec_t(2, 2, -2));

      p2.fInd = {5, 6, 7, 8};

      p2.Init();
    } else if (test == 22) {
      p = {7, v, false};
      v.push_back(Vec_t(1, 0, 0));
      v.push_back(Vec_t(1, 1, 0));
      v.push_back(Vec_t(-1, 1, 0));
      v.push_back(Vec_t(-1, -1, 0));
      v.push_back(Vec_t(-0.9, 0, 0));
      v.push_back(Vec_t(-0.8, -1, 0));
      v.push_back(Vec_t(-0.7, 0, 0));
      p.fInd = {0, 1, 2, 3, 4, 5, 6};
      p.Init();
      v.push_back(Vec_t(2, 2, 2));
      v.push_back(Vec_t(-2, -2, 2));
      v.push_back(Vec_t(-2, -2, -2));
      v.push_back(Vec_t(2, 2, -2));

      p2.fInd = {7, 8, 9, 10};

      p2.Init();
    }

    Utils3D::PolygonIntersection *pi = p.Intersect(p2);

    Visualizer visualizer;
    SimpleBox boxshape("box", 5, 5, 5);
    visualizer.AddVolume(boxshape);

    DrawPolygon(p, visualizer, kGreen);
    DrawPolygon(p2, visualizer, kRed);

    for (auto &point : pi->fPoints) {
      std::cout << "point " << point << '\n';
      visualizer.AddPoint(point, kBlue);
    }

    for (auto &line : pi->fLines) {
      std::cout << "line " << line.fPts[0] << ' ' << line.fPts[1] << '\n';
      visualizer.AddLine(line.fPts[0], line.fPts[1]);
    }

    for (auto &poly : pi->fPolygons) {
      std::cout << "polygon\n";
      DrawPolygon(poly, visualizer, kBlue);
    }

    visualizer.Show();
    return 0;

  } else if (!v.compare("pip")) {
    Visualizer visualizer;
    SimpleBox boxshape("box", 3, 3, 3);
    visualizer.AddVolume(boxshape);

    std::vector<Vec_t> v = {{2, 2, 0}, {-2, 2, 0}, {-2, -2, 0}, {0, 0, 0}, {2, -2, 0},
                            {2, 2, 1}, {-2, 2, 1}, {-2, -2, 1}, {0, 0, 1}, {2, -2, 1}};

    Utils3D::Polygon p{5, v, false};
    p.fInd = {0, 1, 2, 3, 4};
    p.Init();

    Precision x[2], y[2], z[2];
    p.Extent(x, y, z);

    int nPoints = nSegments;
    int offset  = 2;
    for (int i = 0; i < nPoints; i++) {

      Precision randX = random_double(x[0] - offset, x[1] + offset);
      Precision randY = random_double(y[0] - offset, y[1] + offset);
      Precision randZ = random_double(z[0] - offset, z[1] + offset);

      Vec_t randPoint = Vec_t(randX, randY, randZ);
      Precision dot   = (randPoint - (*p.fVert)[p.fInd[0]]).Dot(p.fNorm);
      // std::cout << dot << ' ' << p.fNorm;
      Vec_t projected = randPoint - p.fNorm * dot;

      bool isInside = p.IsPointInside(projected);
      if (isInside) {
        visualizer.AddPoint(projected, kGreen, 1, 1);
      } else {
        visualizer.AddPoint(projected, kRed, 1, 1);
      }
    }

    DrawPolygon(p, visualizer, kBlue);

    visualizer.Show();
    return 0;
  }

  else if (!v.compare("pit")) {
    Visualizer visualizer;
    SimpleBox boxshape("box", 3, 3, 3);
    visualizer.AddVolume(boxshape);

    std::vector<Vec_t> v = {{2, 2, 0}, {-2, 2, 0}, {-2, -2, 0}, {0, 0, 0}, {2, -2, 0},
                            {2, 2, 1}, {-2, 2, 1}, {-2, -2, 1}, {0, 0, 1}, {2, -2, 1}};

    Utils3D::Polygon p{3, v, false};
    p.fInd = {0, 1, 2};
    p.Init();

    Precision x[2], y[2], z[2];
    p.Extent(x, y, z);

    int nPoints = nSegments;
    int offset  = 2;
    for (int i = 0; i < nPoints; i++) {

      Precision randX = random_double(x[0] - offset, x[1] + offset);
      Precision randY = random_double(y[0] - offset, y[1] + offset);
      Precision randZ = random_double(z[0] - offset, z[1] + offset);

      Vec_t randPoint = Vec_t(randX, randY, randZ);
      Precision dot   = (randPoint - (*p.fVert)[p.fInd[0]]).Dot(p.fNorm);
      // std::cout << dot << ' ' << p.fNorm;
      Vec_t projected = randPoint - p.fNorm * dot;

      bool isInside = p.IsPointInside(projected);
      if (isInside) {
        visualizer.AddPoint(projected, kGreen, 1, 1);
      } else {
        visualizer.AddPoint(projected, kRed, 1, 1);
      }
    }

    DrawPolygon(p, visualizer, kBlue);

    visualizer.Show();
    return 0;
  } else {
    print_msg();
    return 0;
  }

  Visualizer visualizer;
  // adjust world box to volume
  Vector3D<Precision> vmin, vmax;
  unplacedvolume->GetBBox(vmin, vmax);
  Vector3D<Precision> extent = 0.52 * (vmax - vmin);
  SimpleBox boxshape("box", extent[0], extent[1], extent[2]);
  visualizer.AddVolume(boxshape);
  auto mesh = MarchingCubes(unplacedvolume, nSegments)->GetMesh();
  DrawPolyhedron(mesh, visualizer, kBlue);
  visualizer.Show();

#endif

  return 0;
}
