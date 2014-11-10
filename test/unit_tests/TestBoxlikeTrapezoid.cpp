/// File: test/unit_test/TestBoxlikeTrapezoid.cpp
/// Author: Guilherme Lima (lima at fnal dot gov)
//
//  Note: the purpose of this (temporary?) file is to perform unit tests
//     where both VecGeom's and USolids' trapezoids pass the tests.
//     The USolids trapezoid fails more detailed tests in TestTrap.cpp

#undef NDEBUG

#include "base/Vector3D.h"
#include "volumes/Trapezoid.h"
#include "ApproxEqual.h"
#ifdef VECGEOM_USOLIDS
#include "UTrap.hh"
#include "UVector3.hh"
#endif

//#include <cassert>
#include <cmath>

template <class Trapezoid_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >
bool TestTrapezoid() {

    Vec_t pzero(0,0,0);
    Vec_t ponxside(20,0,0),ponyside(0,30,0),ponzside(0,0,40);
    Vec_t ponmxside(-20,0,0),ponmyside(0,-30,0),ponmzside(0,0,-40);
    Vec_t ponzsidey(0,25,40),ponmzsidey(0,25,-40);

    Vec_t pbigx(100,0,0),pbigy(0,100,0),pbigz(0,0,100);
    Vec_t pbigmx(-100,0,0),pbigmy(0,-100,0),pbigmz(0,0,-100);

    Vec_t vx(1,0,0),vy(0,1,0),vz(0,0,1);
    Vec_t vmx(-1,0,0),vmy(0,-1,0),vmz(0,0,-1);
    Vec_t vxy(1/std::sqrt(2.0),1/std::sqrt(2.0),0);
    Vec_t vmxy(-1/std::sqrt(2.0),1/std::sqrt(2.0),0);
    Vec_t vmxmy(-1/std::sqrt(2.0),-1/std::sqrt(2.0),0);
    Vec_t vxmy(1/std::sqrt(2.0),-1/std::sqrt(2.0),0);
    Vec_t vxmz(1/std::sqrt(2.0),0,-1/std::sqrt(2.0));

    double Dist;
    Vec_t norm;
    bool convex;
    convex = convex;
   
    Trapezoid_t trap1("Test Trapezoid #1",40.,0.,0.,30.,20.,20.,0.,30.,20.,20.,0.);
    Trapezoid_t trap2("Test Trapezoid #2",10.,0.,0.,10.,10.,10.,0.,10.,10.,10.,0.);
    Trapezoid_t trap3("Trapezoid #3",22.699999999999999,0,0,
                      24.707000000000001,0.14999999999999999, 0.14999999999999999, 0,
                      24.707000000000001,0.14999999999999999, 0.14999999999999999, 0);

// Check name
    assert(trap1.GetName()=="Test Trapezoid #1");

    // Check cubic volume

    std::cout<<"TestTrapezoid: trap2.Capacity()="<< trap2.Capacity()
             <<", trap1.Capacity()="<< trap1.Capacity() << std::endl;
    assert(trap2.Capacity() == 8000);    
    assert(trap1.Capacity() == 192000); 

    // Check Surface area
   
    assert(trap1.SurfaceArea() == 20800);    
    assert(trap2.SurfaceArea() == 6*20*20); 



// CalculateExtent
    
    Vec_t minExtent,maxExtent;
    trap1.Extent(minExtent,maxExtent);
    assert(ApproxEqual(minExtent,Vec_t(-20,-30,-40)));
    assert(ApproxEqual(maxExtent,Vec_t( 20, 30, 40)));
    trap2.Extent(minExtent,maxExtent);
    assert(ApproxEqual(minExtent,Vec_t(-10,-10,-10)));
    assert(ApproxEqual(maxExtent,Vec_t( 10, 10, 10)));

// Check Surface Normal
    Vec_t normal;
    bool valid;
    // Normals on Surface
    valid=trap1.Normal(ponxside,normal);
    assert(ApproxEqual(normal,Vec_t(1,0,0)));
    valid=trap1.Normal(ponmxside,normal);
    assert(ApproxEqual(normal,Vec_t(-1,0,0)));
    valid=trap1.Normal(ponyside,normal);
    assert(ApproxEqual(normal,Vec_t(0,1,0)));
    valid=trap1.Normal(ponmyside,normal);
    assert(ApproxEqual(normal,Vec_t(0,-1,0)));
    valid=trap1.Normal(ponzside,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,1)));
    valid=trap1.Normal(ponmzside,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,-1)));
    valid=trap1.Normal(ponzsidey,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,1)));
    valid=trap1.Normal(ponmzsidey,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,-1)));

    // Normals on Edges
    Vec_t edgeXY(    20.0,  30., 0.0); 
    Vec_t edgemXmY( -20.0, -30., 0.0); 
    Vec_t edgeXmY(   20.0, -30., 0.0); 
    Vec_t edgemXY(  -20.0,  30., 0.0); 
    Vec_t edgeXZ(    20.0, 0.0, 40.0); 
    Vec_t edgemXmZ( -20.0, 0.0, -40.0); 
    Vec_t edgeXmZ(   20.0, 0.0, -40.0); 
    Vec_t edgemXZ(  -20.0, 0.0, 40.0); 
    Vec_t edgeYZ(    0.0,  30.0,  40.0); 
    Vec_t edgemYmZ(  0.0, -30.0, -40.0); 
    Vec_t edgeYmZ(   0.0,  30.0, -40.0); 
    Vec_t edgemYZ(   0.0, -30.0,  40.0); 

    double invSqrt2 = 1.0 / std::sqrt( 2.0);
    double invSqrt3 = 1.0 / std::sqrt( 3.0);
  
    valid= trap1.Normal( edgeXY,normal );
    assert(valid==true);
    assert(ApproxEqual( normal, Vec_t( invSqrt2, invSqrt2, 0.0) )); 
    valid= trap1.Normal( edgemXmY,normal ); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, -invSqrt2, 0.0) )); 
    valid= trap1.Normal( edgeXmY,normal ); 
    assert(ApproxEqual( normal, Vec_t( invSqrt2, -invSqrt2, 0.0) )); 
    valid= trap1.Normal( edgemXY ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, invSqrt2, 0.0) )); 

    valid= trap1.Normal( edgeXZ,normal ); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt2, 0.0, invSqrt2) )); 
    valid= trap1.Normal( edgemXmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, 0.0, -invSqrt2) )); 
    valid= trap1.Normal( edgeXmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt2, 0.0, -invSqrt2) )); 
    valid= trap1.Normal( edgemXZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, 0.0, invSqrt2) )); 

    valid= trap1.Normal( edgeYZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( 0.0,  invSqrt2,  invSqrt2) )); 
    valid= trap1.Normal( edgemYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( 0.0, -invSqrt2, -invSqrt2) )); 
    valid= trap1.Normal( edgeYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( 0.0,  invSqrt2, -invSqrt2) )); 
    valid= trap1.Normal( edgemYZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( 0.0, -invSqrt2,  invSqrt2) )); 

    // Normals on corners
    Vec_t cornerXYZ(    20.0,  30., 40.0); 
    Vec_t cornermXYZ(  -20.0,  30., 40.0); 
    Vec_t cornerXmYZ(   20.0, -30., 40.0); 
    Vec_t cornermXmYZ( -20.0, -30., 40.0); 
    Vec_t cornerXYmZ(    20.0,  30., -40.0); 
    Vec_t cornermXYmZ(  -20.0,  30., -40.0); 
    Vec_t cornerXmYmZ(   20.0, -30., -40.0); 
    Vec_t cornermXmYmZ( -20.0, -30., -40.0); 
 
    valid= trap1.Normal( cornerXYZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3,  invSqrt3, invSqrt3) )); 
    valid= trap1.Normal( cornermXYZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3,  invSqrt3, invSqrt3) )); 
    valid= trap1.Normal( cornerXmYZ,normal ); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3, -invSqrt3, invSqrt3) )); 
    valid= trap1.Normal( cornermXmYZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3, -invSqrt3, invSqrt3) )); 
    valid= trap1.Normal( cornerXYmZ,normal ); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3,  invSqrt3, -invSqrt3) )); 
    valid= trap1.Normal( cornermXYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3,  invSqrt3, -invSqrt3) )); 
    valid= trap1.Normal( cornerXmYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3, -invSqrt3, -invSqrt3) )); 
    valid= trap1.Normal( cornermXmYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3, -invSqrt3, -invSqrt3) )); 
    
    // DistanceToOut(P,V) with asserts for norm and convex
     Dist=trap1.DistanceToOut(pzero,vx,norm,convex);
     assert(ApproxEqual(Dist,20)&&ApproxEqual(norm,vx)&&convex);
     Dist=trap1.DistanceToOut(pzero,vmx,norm,convex);
     assert(ApproxEqual(Dist,20)&&ApproxEqual(norm,vmx)&&convex);
     Dist=trap1.DistanceToOut(pzero,vy,norm,convex);
     assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,vy)&&convex);
     Dist=trap1.DistanceToOut(pzero,vmy,norm,convex);
     assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,vmy)&&convex);
     Dist=trap1.DistanceToOut(pzero,vz,norm,convex);
     assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vz)&&convex);
     Dist=trap1.DistanceToOut(pzero,vmz,norm,convex);
     assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vmz)&&convex);
     Dist=trap1.DistanceToOut(pzero,vxy,norm,convex);
     assert(ApproxEqual(Dist,std::sqrt(800.))&&convex);

     Dist=trap1.DistanceToOut(ponxside,vx,norm,convex);
     assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vx)&&convex);
     Dist=trap1.DistanceToOut(ponxside,vmx,norm,convex);
     assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vmx)&&convex);
     Dist=trap1.DistanceToOut(ponmxside,vmx,norm,convex);
     assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmx)&&convex);
     Dist=trap1.DistanceToOut(ponyside,vy,norm,convex);
     assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vy)&&convex);
     Dist=trap1.DistanceToOut(ponmyside,vmy,norm,convex);
     assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmy)&&convex);
     Dist=trap1.DistanceToOut(ponzside,vz,norm,convex);
     assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vz)&&convex);
     Dist=trap1.DistanceToOut(ponmzside,vmz,norm,convex);
     assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmz)&&convex);
//#endif

// Check Inside
    assert(trap1.Inside(pzero)==vecgeom::EInside::kInside);
    // assert(trap1.Inside(pzero)==VUSolid::eOutside);
    assert(trap1.Inside(pbigz)==vecgeom::EInside::kOutside);
    assert(trap1.Inside(ponxside)==vecgeom::EInside::kSurface);
    assert(trap1.Inside(ponyside)==vecgeom::EInside::kSurface);
    assert(trap1.Inside(ponzside)==vecgeom::EInside::kSurface);   

    assert(trap2.Inside(pzero)==vecgeom::EInside::kInside);
    assert(trap2.Inside(pbigz)==vecgeom::EInside::kOutside);
    assert(trap2.Inside(ponxside)==vecgeom::EInside::kOutside);
    assert(trap2.Inside(ponyside)==vecgeom::EInside::kOutside);
    assert(trap2.Inside(ponzside)==vecgeom::EInside::kOutside);
    assert(trap2.Inside(Vec_t(10,0,0))==vecgeom::EInside::kSurface);
    assert(trap2.Inside(Vec_t(0,10,0))==vecgeom::EInside::kSurface);
    assert(trap2.Inside(Vec_t(0,0,10))==vecgeom::EInside::kSurface);   

// SafetyFromInside(P)
    Dist=trap1.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,20));
    Dist=trap1.SafetyFromInside(vx);
    assert(ApproxEqual(Dist,19));
    Dist=trap1.SafetyFromInside(vy);
    assert(ApproxEqual(Dist,20));
    Dist=trap1.SafetyFromInside(vz);
    assert(ApproxEqual(Dist,20));

// Check DistanceToOut
    Dist=trap1.DistanceToOut(pzero,vx,norm,convex);
     assert(ApproxEqual(Dist,20));
     Dist=trap1.DistanceToOut(pzero,vmx,norm,convex);
     assert(ApproxEqual(Dist,20));
     Dist=trap1.DistanceToOut(pzero,vy,norm,convex);
     assert(ApproxEqual(Dist,30));
     Dist=trap1.DistanceToOut(pzero,vmy,norm,convex);
     assert(ApproxEqual(Dist,30));
     Dist=trap1.DistanceToOut(pzero,vz,norm,convex);
     assert(ApproxEqual(Dist,40));
     Dist=trap1.DistanceToOut(pzero,vmz,norm,convex);
     assert(ApproxEqual(Dist,40));
     Dist=trap1.DistanceToOut(pzero,vxy,norm,convex);
     assert(ApproxEqual(Dist,std::sqrt(800.)));

     Dist=trap1.DistanceToOut(ponxside,vx,norm,convex);
     assert(ApproxEqual(Dist,0));
     Dist=trap1.DistanceToOut(ponxside,vmx,norm,convex);
     assert(ApproxEqual(Dist,40));
     Dist=trap1.DistanceToOut(ponmxside,vmx,norm,convex);
     assert(ApproxEqual(Dist,0));
     Dist=trap1.DistanceToOut(ponyside,vy,norm,convex);
     assert(ApproxEqual(Dist,0));
     Dist=trap1.DistanceToOut(ponmyside,vmy,norm,convex);
     assert(ApproxEqual(Dist,0));
     Dist=trap1.DistanceToOut(ponzside,vz,norm,convex);
     assert(ApproxEqual(Dist,0));
     Dist=trap1.DistanceToOut(ponmzside,vmz,norm,convex);
     assert(ApproxEqual(Dist,0));

//SafetyFromOutside(P)
    Dist=trap1.SafetyFromOutside(pbigx);
    assert(ApproxEqual(Dist,80));
    Dist=trap1.SafetyFromOutside(pbigmx);
    assert(ApproxEqual(Dist,80));
    Dist=trap1.SafetyFromOutside(pbigy);
    assert(ApproxEqual(Dist,70));
    Dist=trap1.SafetyFromOutside(pbigmy);
    assert(ApproxEqual(Dist,70));
    Dist=trap1.SafetyFromOutside(pbigz);
    assert(ApproxEqual(Dist,60));
    Dist=trap1.SafetyFromOutside(pbigmz);
    assert(ApproxEqual(Dist,60));

// DistanceToIn(P,V)
    Dist=trap1.DistanceToIn(pbigx,vmx);
    assert(ApproxEqual(Dist,80));
    Dist=trap1.DistanceToIn(pbigmx,vx);
    assert(ApproxEqual(Dist,80));
    Dist=trap1.DistanceToIn(pbigy,vmy);
    assert(ApproxEqual(Dist,70));
    Dist=trap1.DistanceToIn(pbigmy,vy);
    assert(ApproxEqual(Dist,70));
    Dist=trap1.DistanceToIn(pbigz,vmz);
    assert(ApproxEqual(Dist,60));
    Dist=trap1.DistanceToIn(pbigmz,vz);
    assert(ApproxEqual(Dist,60));
    Dist=trap1.DistanceToIn(pbigx,vxy);
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));
    Dist=trap1.DistanceToIn(pbigmx,vxy);
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));

    Vec_t pJohnXZ(9,0,12);
    Dist = trap2.DistanceToIn(pJohnXZ,vxmz) ;
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));

    Vec_t pJohnXY(12,9,0);
    Dist = trap2.DistanceToIn(pJohnXY,vmxy) ;
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));

    Dist = trap2.DistanceToIn(pJohnXY,vmx) ;
    assert(ApproxEqual(Dist,2));

    Vec_t pMyXY(32,-11,0);
    Dist = trap2.DistanceToIn(pMyXY,vmxy) ;
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));

    Dist = trap1.DistanceToIn(Vec_t(-25,-35,0),vx) ;
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));

    Dist = trap1.DistanceToIn(Vec_t(-25,-35,0),vy) ;
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));
    

    Dist = trap2.DistanceToIn(pJohnXY,vmx) ;
    assert(ApproxEqual(Dist,2));

    Dist=trap3.DistanceToIn(Vec_t(  0.15000000000000185,
                                         -22.048743592955137,
                                           2.4268539333219472),
                           Vec_t(-0.76165597579890043,
                                          0.64364445891356026,
                                         -0.074515708658524193)) ;
    assert(ApproxEqual(Dist,0.0));
   

    /** testing tolerance of DistanceToIn **/
    Trapezoid_t b4("Trapezoid4",5.,0,0,5.,5.,5.,0,5.,5.,5.,0);
    // a point very slightly inside should return 0
    Dist = b4.DistanceToIn( Vec_t(-3.0087437277453119577,
                                  -4.9999999999999928946,
                                  4.8935648380409944025),
                            Vec_t(0.76315134679548990437,
                                  0.53698876104646497964,
                                  -0.35950395323836459305) );
    assert(ApproxEqual(Dist,0.0));

    /* **********************************************************
    */ /////////////////////////////////////////////////////

    return true;
}

int main() {

#ifdef VECGEOM_USOLIDS
  assert( TestTrapezoid<UTrap>() );
  std::cout << "USolids UTrap passed\n";
#endif

  assert( TestTrapezoid<vecgeom::SimpleTrapezoid>() );
  std::cout << "VecGeom Trapezoid passed\n";

  return 0;
}
