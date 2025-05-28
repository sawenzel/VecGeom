#include "VecGeom/volumes/utilities/VolumeUtilities.h"
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/base/Transformation3D.h"
#include "VecGeom/volumes/Box.h"
#include "VecGeomTest/Visualizer.h"
#undef NDEBUG
#include "VecGeom/base/Assert.h"

using namespace vecgeom;

int main()
{
  // simple intersection tests
  {
    // intersect when one is fully contained in other
    Vector3D<Precision> lower1(0., 0., 0.);
    Vector3D<Precision> upper1(1., 1., 1.);
    Vector3D<Precision> lower2(0.5, 0.5, 0.5);
    Vector3D<Precision> upper2(0.6, 0.6, 0.6);
    VECGEOM_ASSERT(volumeUtilities::IntersectionExist(lower1, upper1, lower2, upper2));
    VECGEOM_ASSERT(volumeUtilities::IntersectionExist(lower2, upper2, lower1, upper1));
  }

  Transformation3D transform1(-1, 0, 0, 0, 0, -45);
  Transformation3D transform2(8.9, 0, 0, 45, 0, 45);

  // std::cout<<t0<<std::endl;
  // std::cout<<t0.InverseTransformDirection(Vector3D<Precision>(1,0,0))<<std::endl;

  SimpleBox box("VisualizerBox", 2, 3, 4);
  const Transformation3D *m = box.GetTransformation();
  std::cout << "Box Transformation : " << m->InverseTransformDirection(Vector3D<Precision>(1, 0, 0)) << std::endl;

  Precision dx = 2., dy = 5., dz = 6.;
  Precision b1dx = 2 * dx, b1dy = 1.5 * dy, b1dz = 1.2 * dz;
  Precision b2dx = dx, b2dy = dy, b2dz = dz;
  // Creating a box from unplaced then placed it at a known location
  // UnplacedBox worldUnplaced = UnplacedBox(dx*4, dy*4, dz*4);

  SimpleBox boxSimple1("SimpleBox", b1dx, b1dy, b1dz);
  SimpleBox boxSimple2("SimpleBox", b2dx, b2dy, b2dz);
  // LogicalVolume world ("world", &worldUnplaced);
  UnplacedBox boxUnplaced = UnplacedBox(b2dx, b2dy, b2dz);
  LogicalVolume box2("box", &boxUnplaced);

  UnplacedBox box1Unplaced = UnplacedBox(b1dx, b1dy, b1dz);
  LogicalVolume box1("box1", &box1Unplaced);

  // Transformation3D placement(0.1, 0, 0);
  // world.PlaceDaughter("box", &box2, &placement);

  // VPlacedVolume *worldPlaced = world.Place();
  // const VPlacedVolume *boxPlaced = box2.Place(&t0);

  VPlacedVolume *box1Placed = box1.Place(&transform1);
  VPlacedVolume *box2Placed = box2.Place(&transform2);
  m                         = box1Placed->GetTransformation();
  const Transformation3D *n = box2Placed->GetTransformation();

  // m = box2.GetTransformation();
  // std::cout<<"Placed Box Transformation : "<<m->InverseTransformDirection(Vector3D<Precision>(1,0,0))<<std::endl;

  Vector3D<Precision> aMin1(0, 0, 0), aMax1(0, 0, 0);
  Vector3D<Precision> aMin2(0, 0, 0), aMax2(0, 0, 0);
  box1Placed->Extent(aMin1, aMax1);
  box2Placed->Extent(aMin2, aMax2);

  // std::cout<<"LowerCorner : "<<m->InverseTransform(aMin)<<"  :: UpperCorner :
  // "<<m->InverseTransform(aMax)<<std::endl;
  // Below calculation is for generating x, y and z axis for visualization
  Vector3D<Precision> centreTransformedBox = m->InverseTransform(Vector3D<Precision>(0, 0, 0));
  Vector3D<Precision> newXAxis             = m->InverseTransformDirection(Vector3D<Precision>(1, 0, 0));
  Vector3D<Precision> newYAxis             = m->InverseTransformDirection(Vector3D<Precision>(0, 1, 0));
  Vector3D<Precision> newZAxis             = m->InverseTransformDirection(Vector3D<Precision>(0, 0, 1));
  Vector3D<Precision> pOnX                 = centreTransformedBox + (2. * newXAxis);
  Vector3D<Precision> pOnY                 = centreTransformedBox + (2. * newYAxis);
  Vector3D<Precision> pOnZ                 = centreTransformedBox + (2. * newZAxis);

  Vector3D<Precision> centreTransformedBox2 = n->InverseTransform(Vector3D<Precision>(0, 0, 0));
  Vector3D<Precision> newXAxis2             = n->InverseTransformDirection(Vector3D<Precision>(1, 0, 0));
  Vector3D<Precision> newYAxis2             = n->InverseTransformDirection(Vector3D<Precision>(0, 1, 0));
  Vector3D<Precision> newZAxis2             = n->InverseTransformDirection(Vector3D<Precision>(0, 0, 1));
  Vector3D<Precision> pOnX2                 = centreTransformedBox2 + (2. * newXAxis2);
  Vector3D<Precision> pOnY2                 = centreTransformedBox2 + (2. * newYAxis2);
  Vector3D<Precision> pOnZ2                 = centreTransformedBox2 + (2. * newZAxis2);
  //-------------------------------------------------------------------------------------------------------------

  std::cout << "Centre of transformed box  : " << centreTransformedBox << std::endl;

  // Box Intersection test with Real case
  // using two identical boxes, one without any transformation and second with some transformation
  //"\033[1;31mbold red text\033[0m\n"
  std::cout << std::endl << "\033[1;31m ---- Trying intersection Detection with Real Boxes ----\033[0m\n" << std::endl;

  std::cout << "\033[1;31m Intersection Result from new function : "
            << volumeUtilities::IntersectionExist(aMin1, aMax1, aMin2, aMax2, m, n, true) << "\033[0m\n"
            << std::endl;
  std::cout << "____________________________________________" << std::endl;

  // Using two different boxes, with different transformation
  //-------------------------------------

  // Visualization Stuff
  Visualizer visualizer;
  visualizer.AddVolume(boxSimple1, transform1);
  visualizer.AddVolume(boxSimple2, transform2);
  // visualizer.AddPoint(centreTransformedBox);
  visualizer.AddPoint(m->InverseTransform(aMin1));
  visualizer.AddPoint(m->InverseTransform(aMax1));
  visualizer.AddPoint(n->InverseTransform(aMin2));
  visualizer.AddPoint(n->InverseTransform(aMax2));
  // visualizer.AddPoint(aMin1);
  // visualizer.AddPoint(aMax1);
  // visualizer.AddPoint(aMin2);
  // visualizer.AddPoint(aMax2);

  visualizer.AddLine(centreTransformedBox, pOnX);
  visualizer.AddLine(centreTransformedBox, pOnY);
  visualizer.AddLine(centreTransformedBox, pOnZ);

  visualizer.AddLine(centreTransformedBox2, pOnX2);
  visualizer.AddLine(centreTransformedBox2, pOnY2);
  visualizer.AddLine(centreTransformedBox2, pOnZ2);

  // Making the Global coordinate axes
  Vector3D<Precision> origin(0., 0., 0.);
  Vector3D<Precision> xGlobal(100., 0., 0.);
  Vector3D<Precision> yGlobal(0., 100., 0.);
  Vector3D<Precision> zGlobal(0., 0., 100.);
  visualizer.AddLine(origin, xGlobal);
  visualizer.AddLine(origin, yGlobal);
  visualizer.AddLine(origin, zGlobal);

  visualizer.Show();
  return 0;
}
