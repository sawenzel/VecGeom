//
//
// TestTrd
//             Ensure asserts are compiled in

#undef NDEBUG

#include "base/Vector3D.h"
#include "volumes/Trd.h"
#include "ApproxEqual.h"
#ifdef VECGEOM_USOLIDS
#include "UTet.hh"
#include "UVector3.hh"
#endif

//#include <cassert>
#include <cmath>

template <class Tet_t,class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >

bool TestTet()
{

    Vec_t pzero(0,0,0);
    Vec_t pnt1(10.,0.,0.),pnt2(5.0,10.,0), pnt3(5.,5.,10.);
    Vec_t pt1(1.,0.,0.),pt2(0.0,1.,0), pt3(0.,0.,1.);
    Vec_t norm;

    bool  goodTet, valid, convex;
    Tet_t   t1( "Solid Tet #1", pzero, pnt1, pnt2, pnt3, &goodTet); 
    Tet_t   t2( "Solid Tet #2", pzero, pt1, pt2, pt3, &goodTet);

// Check name
    assert(t1.GetName()=="Solid Tet #1");

// Check  Cubic Volume
    double vol,volCheck;
    vol = t2.Capacity();
    volCheck = 1./6.;
    assert(ApproxEqual(vol,volCheck));

 // Check Surface area
    vol = t2.SurfaceArea();
    volCheck = 1.5 + 0.5*std::sqrt(3); 
    
    // std::cout<<trd1.SurfaceArea()<<std::endl;
    assert(ApproxEqual(vol,volCheck));

    Vec_t pntA( 1.0 , 1.0 , 1.0 ); 
    Vec_t pntB( 1.5 , 0.5 , 1.0 );  
    Vec_t pntBr023= (1.0/3.0) * (pzero + pnt2 + pnt3); 
    Vec_t pntC( 0.0,  5.0 , 1.5 );  

// Check Inside
    assert(t1.Inside(pntA)==vecgeom::EInside:: kInside);
    assert(t1.Inside(pntB)==vecgeom::EInside:: kSurface);
    assert(t1.Inside(pntBr023)==vecgeom::EInside:: kSurface);
    assert(t1.Inside(pntC)==vecgeom::EInside:: kOutside);

// Check Surface Normal
    Vec_t normal;
    Vec_t pntOnBotSurf012( 5.0, 5.0, 0.0); 
    Vec_t vmz(0,0,-1.0); 
    Vec_t vmx(-1.0,0,0.0); 
    Vec_t vmy(0,-1.0,0.0); 
    Vec_t vz(0,0,1.0); 
    Vec_t vx(1.0,0,0.0); 
    Vec_t vy(0,1.0,0.0); 
    valid=t1.Normal(pntOnBotSurf012,normal);
    assert(ApproxEqual(normal,vmz)&&valid);
    valid=t2.Normal(Vec_t(0.1,0.1,0.),normal);
    assert(ApproxEqual(normal,vmz));
    valid=t2.Normal(Vec_t(0.1,0,0.1),normal);
    assert(ApproxEqual(normal,vmy));
    valid=t2.Normal(Vec_t(0,0.1,0.1),normal);
    assert(ApproxEqual(normal,vmx));
    
// Check Normals on Edges
    
    Vec_t edgeXY(0.5, 0.0, 0.0); 
    Vec_t edgeXZ(0.0, 0.0, 0.5); 
    Vec_t edgeYZ(0.0, 0.5, 0.0); 
    double invSqrt2 = 1.0 / std::sqrt( 2.0); 
    double invSqrt3 = 1.0 / std::sqrt( 3.0); 
  
    valid= t2.Normal( edgeXY ,normal); 
    assert(ApproxEqual( normal, Vec_t( 0, -invSqrt2, -invSqrt2) )&&valid); 
    valid= t2.Normal( edgeYZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, 0, -invSqrt2) )&&valid); 
    valid= t2.Normal( edgeXZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, -invSqrt2, 0) )&&valid); 
    

// SafetyFromInside(P)

    double Dist;
    Dist=t2.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,0));
    Dist=t2.SafetyFromInside(-vmx);
    assert(ApproxEqual(Dist,0));
    Dist=t2.SafetyFromInside(Vec_t(0.1,0.1,0.1));
    assert(ApproxEqual(Dist,0.1));
 

// DistanceToOut(P,V)

    Dist=t2.DistanceToOut(pzero,vz,norm,convex);
    //std::cout<<Dist<< "   "<<norm<<std::endl;
    assert(ApproxEqual(Dist,1)&&ApproxEqual(norm,Vec_t(invSqrt3, invSqrt3, invSqrt3))&&convex);
    Dist=t2.DistanceToOut(pzero,vy,norm,convex);
    assert(ApproxEqual(Dist,1)&&ApproxEqual(norm,Vec_t(invSqrt3, invSqrt3, invSqrt3))&&convex);
    Dist=t2.DistanceToOut(pzero,vx,norm,convex);
    assert(ApproxEqual(Dist,1)&&ApproxEqual(norm,Vec_t(invSqrt3, invSqrt3, invSqrt3))&&convex);

    
// DistanceToIn(P,V)

    Vec_t pbig(0.1,0.1,-20);
    Dist=t2.DistanceToIn(pbig,vx);
    assert(ApproxEqual(Dist,UUtils::kInfinity));
    
    Dist=t2.DistanceToIn(pbig,vy);
    assert(ApproxEqual(Dist,UUtils::kInfinity));
    Dist=t2.DistanceToIn(pbig,vz);
    assert(ApproxEqual(Dist,20));
    
  // CalculateExtent
    Vec_t minExtent,maxExtent;
    t2.Extent(minExtent,maxExtent);
    //std::cout<<" min="<<minExtent<<" max="<<maxExtent<<std::endl;
    assert(ApproxEqual(minExtent,Vec_t(0,0,0)));
    assert(ApproxEqual(maxExtent,Vec_t( 1, 1, 1)));
   
   
    return true;
}


int main() {
#ifdef VECGEOM_USOLIDS
  assert(TestTet<UTet>());
  std::cout << "UTet passed\n";

#endif
  std::cout<< "VecGeomTet not included yet\n";
  
  return 0;
}
