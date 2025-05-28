///
/// file:    TestUtils3D.cpp
/// purpose: Unit tests for the 3D geometry utilities
///

//-- ensure asserts are compiled in
#ifdef NDEBUG
#undef NDEBUG
#endif

#include "VecGeom/base/Utils3D.h"
#include "ApproxEqual.h"
#include "VecGeom/volumes/Box.h"
#include "test/benchmark/ArgParser.h"

#ifdef VECGEOM_ROOT
#include "VecGeomTest/Visualizer.h"
#include "TPolyMarker3D.h"
#include "TPolyLine3D.h"
#endif

using vecgeom::Precision;

bool ValidXing(vecgeom::Vector3D<Precision> const &point, vecgeom::Vector3D<Precision> const &dir,
               vecgeom::Vector3D<Precision> const &n1, Precision p1, vecgeom::Vector3D<Precision> const &n2,
               Precision p2)
{
  bool valid_dir   = ApproxEqual<Precision>(dir.Dot(n1), 0.) && ApproxEqual<Precision>(dir.Dot(n2), 0.);
  bool valid_point = ApproxEqual<Precision>(point.Dot(n1 - n2) + (p1 - p2), 0.);
  return valid_point & valid_dir;
}

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

void DrawPolyhedron(vecgeom::Utils3D::Polyhedron &polyh, vecgeom::Visualizer &visualizer, size_t color)
{
  using namespace vecgeom;

  for (size_t i = 0; i < polyh.GetNpolygons(); ++i)
    DrawPolygon(polyh.GetPolygon(i), visualizer, color);
}
#endif

int main(int argc, char *argv[])
{
  using namespace vecgeom;
  using namespace vecCore::math;
  using Vec_t = Vector3D<Precision>;

  using vecgeom::Utils3D::Line;
  using vecgeom::Utils3D::Plane;
  using vecgeom::Utils3D::Polygon;
  using vecgeom::Utils3D::Polyhedron;

  const Vec_t dirx(1., 0., 0.);
  const Vec_t diry(0., 1., 0.);
  const Vec_t dirz(0., 0., 1.);

#ifdef VECGEOM_ROOT
  OPTION_INT(vis, 0);
#endif

  ///* Plane transformations */
  Plane pl1(Vec_t(1., 0., 0.), -10.);
  Transformation3D transf1(10., 0., 0., 0., 0., 180.);
  pl1.Transform(transf1);
  VECGEOM_ASSERT(ApproxEqual<Precision>(pl1.fNorm[0], -1.) && ApproxEqual<Precision>(pl1.fDist, 0.));

  // Polygon intersection

  Utils3D::vector_t<Vec_t> pvec1 = {{10., 4., 6.}, {10., -4., 6.}, {10., -4., -6.}, {10., 4., -6.}};
  Polygon poly1(4, pvec1, Vec_t(1., 0., 0.));
  poly1.fInd = {0, 1, 2, 3};
  poly1.Init();
  Utils3D::vector_t<Vec_t> pvec2 = {{7.75736, -3.17423, 10.5854},
                                    {7.75736, -7.17423, 3.65722},
                                    {16.2426, 0.174235, -0.585422},
                                    {16.2426, 4.17423, 6.34278}};
  Polygon poly2(4, pvec2, true);
  poly2.fInd = {0, 1, 2, 3};
  poly2.Init();

  Line line1;
  VECGEOM_ASSERT(Utils3D::PolygonXing(poly1, poly2, &line1) == Utils3D::kOverlapping);

  ///* Test plane crossings */
  Vector3D<Precision> point, direction;
  Vector3D<Precision> n1, n2;
  Precision p1, p2;

  // identical planes
  n1.Set(0., 0., 1.);
  n2.Set(0., 0., 1.);
  p1 = -3.;
  p2 = -3.;
  VECGEOM_ASSERT(Utils3D::PlaneXing(Plane(n1, p1), Plane(n2, p2), point, direction) == Utils3D::kIdentical);

  // identical planes with opposite normals
  n1.Set(0., 0., 1.);
  n2.Set(0., 0., -1.);
  p1 = -3.;
  p2 = 3.;
  VECGEOM_ASSERT(Utils3D::PlaneXing(Plane(n1, p1), Plane(n2, p2), point, direction) == Utils3D::kIdentical);

  // opposite planes with opposite normals
  n1.Set(0., 0., 1.);
  n2.Set(0., 0., -1.);
  p1 = -3;
  p2 = -3;
  VECGEOM_ASSERT(Utils3D::PlaneXing(Plane(n1, p1), Plane(n2, p2), point, direction) == Utils3D::kParallel);

  // opposite planes with identical normal
  n1.Set(0., 0., 1.);
  n2.Set(0., 0., 1.);
  p1 = -3;
  p2 = 3;
  VECGEOM_ASSERT(Utils3D::PlaneXing(Plane(n1, p1), Plane(n2, p2), point, direction) == Utils3D::kParallel);

  // arbitrary parallel planes
  n1.Set(1., 2., 3.);
  n1.Normalize();
  n2 = -n1;
  p1 = 1;
  p2 = -2;
  VECGEOM_ASSERT(Utils3D::PlaneXing(Plane(n1, p1), Plane(n2, p2), point, direction) == Utils3D::kParallel);

  // +z face of a box with +x face of the same box
  n1.Set(0., 0., 1.);
  n2.Set(1., 0., 0.);
  p1 = -3;
  p2 = -2;
  VECGEOM_ASSERT(Utils3D::PlaneXing(Plane(n1, p1), Plane(n2, p2), point, direction) == Utils3D::kIntersecting);
  VECGEOM_ASSERT(ValidXing(point, direction, n1, p1, n2, p2));

  // same as above but 1 face has opposite normal
  n2 = -n2;
  p2 = -p2;
  VECGEOM_ASSERT(Utils3D::PlaneXing(Plane(n1, p1), Plane(n2, p2), point, direction) == Utils3D::kIntersecting);
  VECGEOM_ASSERT(ValidXing(point, direction, n1, p1, n2, p2));

  ///* Test box crossings */
  Vec_t box1(1., 2., 3.);
  Vec_t box2(2., 3., 4.);
  Polyhedron polyh1, polyh2;
  Utils3D::FillBoxPolyhedron(box1, polyh1);
  Utils3D::FillBoxPolyhedron(box2, polyh2);
  Transformation3D tr1, tr2, tr3;

  // Touching boxes
  tr1 = Transformation3D(0., 0., 0.);
  tr2 = Transformation3D(3., 5., 0.);
  VECGEOM_ASSERT(Utils3D::BoxCollision(box1, tr1, box2, tr2) == Utils3D::kTouching);

  // Disjoint boxes
  tr1 = Transformation3D(0., 0., 0.);
  tr2 = Transformation3D(2.5, 4.5, 10.2);
  VECGEOM_ASSERT(Utils3D::BoxCollision(box1, tr1, box2, tr2) == Utils3D::kDisjoint);

  // Overlapping boxes
  tr1 = Transformation3D(0., 0., 0.);
  tr2 = Transformation3D(2.5, 4.5, 6.5);
  VECGEOM_ASSERT(Utils3D::BoxCollision(box1, tr1, box2, tr2) == Utils3D::kOverlapping);

  tr1 = Transformation3D(-1, -0.5, 0.5);
  tr2 = Transformation3D(-3., -0.5, 0.5);
  tr3 = Transformation3D(1., 2., 3., 0., 45., 45.);
  polyh1.Transform(tr1);
  polyh2.Transform(tr3);
  VECGEOM_ASSERT(Utils3D::BoxCollision(box1, tr1, box2, tr3) == Utils3D::kOverlapping &&
                 Utils3D::BoxCollision(box1, tr2, box2, tr3) == Utils3D::kDisjoint);

  std::cout << "TestUtils3D passed\n";

#ifdef VECGEOM_ROOT
  if (vis == 0) return 0;
  Visualizer visualizer;
  SimpleBox boxshape("box", 7, 7, 7);
  visualizer.AddVolume(boxshape);
  Utils3D::vector_t<Utils3D::Line> lines;
  DrawPolyhedron(polyh1, visualizer, kBlue);
  DrawPolyhedron(polyh2, visualizer, kGreen);
  if (PolyhedronXing(polyh1, polyh2, lines) == Utils3D::kOverlapping) {
    TPolyLine3D pl(2);
    pl.SetLineColor(kRed);
    for (auto line : lines) {
      pl.SetNextPoint(line.fPts[0].x(), line.fPts[0].y(), line.fPts[0].z());
      pl.SetNextPoint(line.fPts[1].x(), line.fPts[1].y(), line.fPts[1].z());
      visualizer.AddLine(pl);
    }
  }
  visualizer.Show();
#endif

  return 0;
}
