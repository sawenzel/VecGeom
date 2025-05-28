//
//
// TestBox
//             Ensure asserts are compiled in

#undef NDEBUG
#include "VecGeom/base/Global.h"
#include "VecGeom/base/Vector3D.h"
#include "VecGeom/volumes/Box.h"
//#include "VecGeom/volumes/Orb.h"
#include "VecGeom/volumes/ThetaCone.h"
//#include "ApproxEqual.h"

#include <cmath>

#define PI 3.14159265358979323846
using namespace vecgeom;
// template <class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >
bool TestThetaCone()
{
  typedef Vector3D<Precision> Vec_t;
  // int verbose=0;

  vecgeom::Precision fR = 9.;
  Vec_t pzero(0, 0, 0);
  Vec_t pbigx(100, 0, 0), pbigy(0, 100, 0), pbigz(0, 0, 100);
  Vec_t pbigmx(-100, 0, 0), pbigmy(0, -100, 0), pbigmz(0, 0, -100);
  Vec_t ponx(fR, 0., 0.);   // point on surface on X axis
  Vec_t ponmx(-fR, 0., 0.); // point on surface on minus X axis
  Vec_t pony(0., fR, 0.);   // point on surface on Y axis
  Vec_t ponmy(0., -fR, 0.); // point on surface on minus Y axis
  Vec_t ponz(0., 0., fR);   // point on surface on Z axis
  Vec_t ponmz(0., 0., -fR); // point on surface on minus Z axis

  Vec_t ponxside(fR, 0, 0), ponyside(0, fR, 0), ponzside(0, 0, fR);
  Vec_t ponmxside(-fR, 0, 0), ponmyside(0, -fR, 0), ponmzside(0, 0, -fR);

  Vec_t vx(1, 0, 0), vy(0, 1, 0), vz(0, 0, 1);
  Vec_t vmx(-1, 0, 0), vmy(0, -1, 0), vmz(0, 0, -1);
  Vec_t vxy(1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxy(-1 / std::sqrt(2.0), 1 / std::sqrt(2.0), 0);
  Vec_t vmxmy(-1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);
  Vec_t vxmy(1 / std::sqrt(2.0), -1 / std::sqrt(2.0), 0);
  Vec_t vxmz(1 / std::sqrt(2.0), 0, -1 / std::sqrt(2.0));
  Vec_t px(30, 0, 0), py(0, 30, 0), pz(0, 0, 30);

  ThetaCone tc(0, PI / 3);
  // VECGEOM_ASSERT(tc.Inside<kScalar>(vz)==vecgeom::EInside::kSurface); //Need a check only when fStheta=0 otherwise
  // should be OK
  VECGEOM_ASSERT(tc.Inside<kScalar>(vz) ==
                 vecgeom::EInside::kInside); // This point is resolved now, The point should be inside point
  // not the surface point

  VECGEOM_ASSERT(tc.Inside<kScalar>(vmz) == vecgeom::EInside::kOutside);

  VECGEOM_ASSERT(tc.Inside<kScalar>(vx) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(tc.Inside<kScalar>(vmx) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(tc.Inside<kScalar>(vy) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(tc.Inside<kScalar>(vmy) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(tc.Inside<kScalar>(Vec_t(1, 0, 1)) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(tc.Inside<kScalar>(Vec_t(0, 1, 1)) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(tc.Inside<kScalar>(Vec_t(0, std::sqrt(3), 1)) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc.Inside<kScalar>(Vec_t(-std::sqrt(3), 0, 1)) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc.Inside<kScalar>(Vec_t(-std::sqrt(3), 0, -1)) == vecgeom::EInside::kOutside);

  // Testing Contains function
  VECGEOM_ASSERT(!tc.Contains<kScalar>(vx));
  VECGEOM_ASSERT(!tc.Contains<kScalar>(vmx));
  VECGEOM_ASSERT(!tc.Contains<kScalar>(vy));
  VECGEOM_ASSERT(!tc.Contains<kScalar>(vmy));
  VECGEOM_ASSERT(tc.Contains<kScalar>(Vec_t(1, 0, 1)));
  VECGEOM_ASSERT(tc.Contains<kScalar>(Vec_t(-1, 0, 1)));
  VECGEOM_ASSERT(!tc.Contains<kScalar>(Vec_t(0, 2, 1)));
  VECGEOM_ASSERT(tc.Contains<kScalar>(Vec_t(0, -1, 1)));

  ThetaCone tc2(2 * PI / 3, PI / 3);
  VECGEOM_ASSERT(tc2.Inside<kScalar>(vz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(tc2.Inside<kScalar>(-vz) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(tc2.Inside<kScalar>(vx) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(tc2.Inside<kScalar>(vmx) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(tc2.Inside<kScalar>(vy) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(tc2.Inside<kScalar>(vmy) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(tc2.Inside<kScalar>(Vec_t(1, 0, -1)) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(tc2.Inside<kScalar>(Vec_t(0, 1, -1)) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(tc2.Inside<kScalar>(Vec_t(0, std::sqrt(3), -1)) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc2.Inside<kScalar>(Vec_t(-std::sqrt(3), 0, -1)) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc2.Inside<kScalar>(Vec_t(-std::sqrt(3), 0, 1)) == vecgeom::EInside::kOutside);

  ThetaCone tc3(0, 2 * PI / 3);
  VECGEOM_ASSERT(tc3.Inside<kScalar>(vz) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(tc3.Inside<kScalar>(vx) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(tc3.Inside<kScalar>(vy) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(tc3.Inside<kScalar>(-vz) == vecgeom::EInside::kOutside);

  ThetaCone tc4(PI / 3, 2 * PI / 3);
  VECGEOM_ASSERT(tc4.Inside<kScalar>(vz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(tc4.Inside<kScalar>(vx) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(tc4.Inside<kScalar>(vy) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(tc4.Inside<kScalar>(-vz) == vecgeom::EInside::kInside);

  ThetaCone tc5(0, 0.50 * PI);
  VECGEOM_ASSERT(tc5.Inside<kScalar>(vz) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(tc5.Inside<kScalar>(vmz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(tc5.Inside<kScalar>(vx) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc5.Inside<kScalar>(vmx) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc5.Inside<kScalar>(vy) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc5.Inside<kScalar>(vmy) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc5.Inside<kScalar>(pz) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(tc5.Inside<kScalar>(-pz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(tc5.Inside<kScalar>(px) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc5.Inside<kScalar>(-px) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc5.Inside<kScalar>(py) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc5.Inside<kScalar>(-py) == vecgeom::EInside::kSurface);

  ThetaCone tc6(0.50 * PI, 0.50 * PI);
  VECGEOM_ASSERT(tc6.Inside<kScalar>(vz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(tc6.Inside<kScalar>(vmz) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(tc6.Inside<kScalar>(vx) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc6.Inside<kScalar>(vmx) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc6.Inside<kScalar>(vy) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc6.Inside<kScalar>(vmy) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc6.Inside<kScalar>(pz) == vecgeom::EInside::kOutside);
  VECGEOM_ASSERT(tc6.Inside<kScalar>(-pz) == vecgeom::EInside::kInside);
  VECGEOM_ASSERT(tc6.Inside<kScalar>(px) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc6.Inside<kScalar>(-px) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc6.Inside<kScalar>(py) == vecgeom::EInside::kSurface);
  VECGEOM_ASSERT(tc6.Inside<kScalar>(-py) == vecgeom::EInside::kSurface);

  return true;
}

int main(int argc, char *argv[])
{

  VECGEOM_ASSERT(TestThetaCone());
  std::cout << "---------------------------------------\n";
  std::cout << "------ ThetaCone Test Passed ----------\n";
  std::cout << "---------------------------------------\n";

  return 0;
}
