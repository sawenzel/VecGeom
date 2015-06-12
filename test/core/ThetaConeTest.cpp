//
//
// TestBox
//             Ensure asserts are compiled in

#undef NDEBUG
#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/Box.h"
//#include "volumes/Orb.h"
#include "volumes/ThetaCone.h"
//#include "ApproxEqual.h"
#ifdef VECGEOM_USOLIDS
#include "UBox.hh"
//#include "UOrb.hh"
#include "UVector3.hh"
#endif

#include <cassert>
#include <cmath>

#define PI 3.14159265358979323846
using namespace vecgeom;
//template <class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >
bool TestThetaCone() {
    typedef Vector3D<Precision> Vec_t;
    //int verbose=0;

    vecgeom::Precision fR=9.;
    Vec_t pzero(0,0,0);
    Vec_t pbigx(100,0,0),pbigy(0,100,0),pbigz(0,0,100);
    Vec_t pbigmx(-100,0,0),pbigmy(0,-100,0),pbigmz(0,0,-100);
    Vec_t ponx(fR,0.,0.); // point on surface on X axis
    Vec_t ponmx(-fR,0.,0.); // point on surface on minus X axis
    Vec_t pony(0.,fR,0.); // point on surface on Y axis
    Vec_t ponmy(0.,-fR,0.); // point on surface on minus Y axis
    Vec_t ponz(0.,0.,fR); // point on surface on Z axis
    Vec_t ponmz(0.,0.,-fR); // point on surface on minus Z axis

    Vec_t ponxside(fR,0,0),ponyside(0,fR,0),ponzside(0,0,fR);
    Vec_t ponmxside(-fR,0,0),ponmyside(0,-fR,0),ponmzside(0,0,-fR);

    Vec_t vx(1,0,0),vy(0,1,0),vz(0,0,1);
    Vec_t vmx(-1,0,0),vmy(0,-1,0),vmz(0,0,-1);
    Vec_t vxy(1/std::sqrt(2.0),1/std::sqrt(2.0),0);
    Vec_t vmxy(-1/std::sqrt(2.0),1/std::sqrt(2.0),0);
    Vec_t vmxmy(-1/std::sqrt(2.0),-1/std::sqrt(2.0),0);
    Vec_t vxmy(1/std::sqrt(2.0),-1/std::sqrt(2.0),0);
    Vec_t vxmz(1/std::sqrt(2.0),0,-1/std::sqrt(2.0));

	ThetaCone tc(0,PI/3);
	//std::cout<<tc.Inside<kScalar>(vz)<<std::endl;
	assert(tc.Inside<kScalar>(vz)==vecgeom::EInside::kSurface); //Need a check only when fStheta=0 otherwise should be OK
	assert(tc.Inside<kScalar>(-vz)==vecgeom::EInside::kOutside);
    
	assert(tc.Inside<kScalar>(vx)==vecgeom::EInside::kOutside);
	assert(tc.Inside<kScalar>(vmx)==vecgeom::EInside::kOutside);
	assert(tc.Inside<kScalar>(vy)==vecgeom::EInside::kOutside);
	assert(tc.Inside<kScalar>(vmy)==vecgeom::EInside::kOutside);
	assert(tc.Inside<kScalar>(Vec_t(1,0,1))==vecgeom::EInside::kInside);
	assert(tc.Inside<kScalar>(Vec_t(0,1,1))==vecgeom::EInside::kInside);
	assert(tc.Inside<kScalar>(Vec_t(0,std::sqrt(3),1))==vecgeom::EInside::kSurface);
	assert(tc.Inside<kScalar>(Vec_t(-std::sqrt(3),0,1))==vecgeom::EInside::kSurface);
	assert(tc.Inside<kScalar>(Vec_t(-std::sqrt(3),0,-1))==vecgeom::EInside::kOutside);

	//Testing Contains function
	assert(!tc.Contains<kScalar>(vx));
	assert(!tc.Contains<kScalar>(vmx));
	assert(!tc.Contains<kScalar>(vy));
	assert(!tc.Contains<kScalar>(vmy));
	assert(tc.Contains<kScalar>(Vec_t(1,0,1)));
	assert(tc.Contains<kScalar>(Vec_t(-1,0,1)));
	assert(!tc.Contains<kScalar>(Vec_t(0,2,1)));
	assert(tc.Contains<kScalar>(Vec_t(0,-1,1)));



	ThetaCone tc2(2*PI/3,PI/3);

	assert(tc2.Inside<kScalar>(vz)==vecgeom::EInside::kOutside); //Need a check only when fStheta=0 otherwise should be OK
	assert(tc2.Inside<kScalar>(-vz)==vecgeom::EInside::kSurface);
    assert(tc2.Inside<kScalar>(vx)==vecgeom::EInside::kOutside);
	assert(tc2.Inside<kScalar>(vmx)==vecgeom::EInside::kOutside);
	assert(tc2.Inside<kScalar>(vy)==vecgeom::EInside::kOutside);
	assert(tc2.Inside<kScalar>(vmy)==vecgeom::EInside::kOutside);
	assert(tc2.Inside<kScalar>(Vec_t(1,0,-1))==vecgeom::EInside::kInside);
	assert(tc2.Inside<kScalar>(Vec_t(0,1,-1))==vecgeom::EInside::kInside);
	assert(tc2.Inside<kScalar>(Vec_t(0,std::sqrt(3),-1))==vecgeom::EInside::kSurface);
	assert(tc2.Inside<kScalar>(Vec_t(-std::sqrt(3),0,-1))==vecgeom::EInside::kSurface);
	assert(tc2.Inside<kScalar>(Vec_t(-std::sqrt(3),0,1))==vecgeom::EInside::kOutside);
        return true;
}

 int main(int argc, char *argv[]) {

   assert(TestThetaCone());
  /*
    if( argc < 2)
     {
       std::cerr << "need to give argument :--usolids or --vecgeom\n";
       return 1;
     }

     if( ! strcmp(argv[1], "--usolids") )
     {
 #ifdef VECGEOM_USOLIDS
   assert(TestOrb<UOrb>());
   std::cout << "UOrb passed\n";
       #else
       std::cerr << "VECGEOM_USOLIDS was not defined\n";
       return 2;
 #endif
     }
     else if( ! strcmp(argv[1], "--vecgeom") )
     {
   assert(TestOrb<vecgeom::SimpleOrb>());
   std::cout << "VecGeomOrb passed\n";
     }
     else
     {
       std::cerr << "need to give argument :--usolids or --vecgeom\n";
       return 1;
     }

 */
   return 0;
 }
