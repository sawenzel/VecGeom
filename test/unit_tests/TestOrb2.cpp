//
//
// TestBox
//             Ensure asserts are compiled in

#undef NDEBUG
#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/Box.h"
#include "volumes/Orb.h"
#include "ApproxEqual.h"
#ifdef VECGEOM_USOLIDS
#include "UBox.hh"
#include "UOrb.hh"
#include "UVector3.hh"
#endif

#include <cassert>
#include <cmath>
#include <iomanip>
#define PI 3.14159265358979323846

template <class Orb_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >
bool TestOrb() {
    
    int verbose=0;
    
    vecgeom::Precision fR=9.;
    

    //Check Surface Normal
    Vec_t normal;
    bool valid;
    
    double Dist;
    Vec_t norm;
    bool convex;
    convex = convex;
    
    Orb_t test("test_orb",50.);   
    Vec_t point(-816.03320712862443997, -183.11020255650146282, -528.65179293123389925);
    Vec_t dir(-0.077731580959048057755, 0.23815403417384209406, -0.96811180001502483705);
    //Dist = test.DistanceToIn(point,dir) ;
    Dist = test.DistanceToOut(point,dir,norm,convex);
    std::cout<<std::setprecision(15)<<"DistanceToOut : "<<Dist<<std::endl;
   
    
    return true;
}

int main() {
    
#ifdef VECGEOM_USOLIDS
  assert(TestOrb<UOrb>());
  std::cout << "UOrb passed\n";
#endif
  std::cout<<"------------------------------"<<std::endl;
  assert(TestOrb<vecgeom::SimpleOrb>());
  std::cout << "VecGeomOrb passed\n";
  return 0;
}

