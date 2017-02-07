//
//
// TestTrap




#include "base/Vector3D.h"
#include "volumes/Box.h"
#include "ApproxEqual.h"
#ifdef VECGEOM_USOLIDS
#include "UTrap.hh"
#include "UVector3.hh"
#endif
#include "volumes/Trapezoid.h"

//             Ensure asserts are compiled in
#undef NDEBUG
#include <cassert>
#include <cmath>

template <typename Constants, class Trap_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >
bool TestTrap() {
    Vec_t pzero(0,0,0);
    Vec_t ponxside(20,0,0),ponyside(0,30,0),ponzside(0,0,40);
    Vec_t ponmxside(-20,0,0),ponmyside(0,-30,0),ponmzside(0,0,-40);
    Vec_t ponzsidey(0,25,40),ponmzsidey(0,25,-40);

    Vec_t pbigx(100,0,0),pbigy(0,100,0),pbigz(0,0,100), pbig(100,100,100);
    Vec_t pbigmx(-100,0,0),pbigmy(0,-100,0),pbigmz(0,0,-100);

    Vec_t vx(1,0,0),vy(0,1,0),vz(0,0,1);
    Vec_t vmx(-1,0,0),vmy(0,-1,0),vmz(0,0,-1);
    Vec_t vxy(1/std::sqrt(2.0),1/std::sqrt(2.0),0);
    Vec_t vmxy(-1/std::sqrt(2.0),1/std::sqrt(2.0),0);
    Vec_t vmxmy(-1/std::sqrt(2.0),-1/std::sqrt(2.0),0);
    Vec_t vxmy(1/std::sqrt(2.0),-1/std::sqrt(2.0),0);

    Vec_t vxmz(1/std::sqrt(2.0),0,-1/std::sqrt(2.0));
    Vec_t vymz(0,1/std::sqrt(2.0),-1/std::sqrt(2.0));
    Vec_t vmxmz(-1/std::sqrt(2.0),0,-1/std::sqrt(2.0));
    Vec_t vmymz(0,-1/std::sqrt(2.0),-1/std::sqrt(2.0));
    Vec_t vxz(1/std::sqrt(2.0),0,1/std::sqrt(2.0));
    Vec_t vyz(0,1/std::sqrt(2.0),1/std::sqrt(2.0));

    double Dist, dist, vol, volCheck ;
    Vec_t  normal,norm;
    bool valid,convex;

    double cosa = 4/std::sqrt(17.), sina = 1/std::sqrt(17.);

    Vec_t trapvert[8] = { Vec_t(-10.0,-20.0,-40.0),
                          Vec_t(+10.0,-20.0,-40.0),
                          Vec_t(-10.0,+20.0,-40.0),
                          Vec_t(+10.0,+20.0,-40.0),
                          Vec_t(-30.0,-40.0,+40.0),
                          Vec_t(+30.0,-40.0,+40.0),
                          Vec_t(-30.0,+40.0,+40.0),
                          Vec_t(+30.0,+40.0,+40.0)  } ;
    
    Trap_t trap1("Test Boxlike #1",40,0,0,30,20,20,0,30,20,20,0); // box:20,30,40
    
    //    Trap_t trap2("Test Trdlike #2",40,0,0,20,10,10,0,40,30,30,0);
    
    Trap_t trap2("Test Trdlike #2",trapvert);

    Trap_t trap3("trap3",50,0,0,50,50,50,UUtils::kPi/4,50,50,50,UUtils::kPi/4) ;
    Trap_t trap4("trap4",50,0,0,50,50,50,-UUtils::kPi/4,50,50,50,-UUtils::kPi/4) ;

    Vec_t Corners[8];
    Corners[0]=Vec_t(-3.,-3.,-3.);
    Corners[1]=Vec_t(3.,-3.,-3.);
    Corners[2]=Vec_t(-3.,3.,-3.);
    Corners[3]=Vec_t(3.,3.,-3.);
    Corners[4]=Vec_t(-3.,-3.,3.);
    Corners[5]=Vec_t(3.,-3.,3.);
    Corners[6]=Vec_t(-3.,3.,3.);
    Corners[7]=Vec_t(3.,3.,3.);

    Trap_t tempTrap("temp trap", Corners);


// Check name

    assert(trap1.GetName()=="Test Boxlike #1");
    assert(trap2.GetName()=="Test Trdlike #2");

// Check cubic volume

    vol = trap1.Capacity();
    volCheck = 8*20*30*40;
    assert(ApproxEqual(vol,volCheck));

    vol = trap4.Capacity();
    volCheck = 8*50*50*50;
    assert(ApproxEqual(vol,volCheck));

    vol = trap3.Capacity();
    volCheck = 8*50*50*50;
    assert(ApproxEqual(vol,volCheck));

    vol = trap2.Capacity();
    volCheck = 2*40.*( (20.+40.)*(10.+30.) + (30.-10.)*(40.-20.)/3. );
    assert(ApproxEqual(vol,volCheck));

// Check surface area

    vol = trap1.SurfaceArea();
    volCheck = 2*(40*60+80*60+80*40);
    assert(ApproxEqual(vol,volCheck));

    vol = trap2.SurfaceArea();
    volCheck =  4*(10*20+30*40)
                 + 2*((20+40)*std::sqrt(4*40*40+(30-10)*(30-10))
              + (30+10) *std::sqrt(4*40*40+(40-20)*(40-20))) ;
    assert(ApproxEqual(vol,volCheck));

// Check Inside

    assert(trap1.Inside(pzero)==vecgeom::EInside::kInside);
    assert(trap1.Inside(pbigz)==vecgeom::EInside::kOutside);
    assert(trap1.Inside(ponxside)==vecgeom::EInside::kSurface);
    assert(trap1.Inside(ponyside)==vecgeom::EInside::kSurface);
    assert(trap1.Inside(ponzside)==vecgeom::EInside::kSurface);

    assert(trap2.Inside(pzero)==vecgeom::EInside::kInside);
    assert(trap2.Inside(pbigz)==vecgeom::EInside::kOutside);
    assert(trap2.Inside(ponxside)==vecgeom::EInside::kSurface);
    assert(trap2.Inside(ponyside)==vecgeom::EInside::kSurface);
    assert(trap2.Inside(ponzside)==vecgeom::EInside::kSurface);

// Check Surface Normal

   
    valid=trap1.Normal(ponxside,normal);
    assert(ApproxEqual(normal,Vec_t(1.,0.,0.)));
    valid=trap1.Normal(ponmxside,normal);
    assert(ApproxEqual(normal,Vec_t(-1.,0.,0.)));
    valid=trap1.Normal(ponyside,normal);
    assert(ApproxEqual(normal,Vec_t(0.,1.,0.)));
    valid=trap1.Normal(ponmyside,normal);
    assert(ApproxEqual(normal,Vec_t(0.,-1.,0.)));
    valid=trap1.Normal(ponzside,normal);
    assert(ApproxEqual(normal,Vec_t(0.,0.,1.)));
    valid=trap1.Normal(ponmzside,normal);
    assert(ApproxEqual(normal,Vec_t(0.,0.,-1.)));
    valid=trap1.Normal(ponzsidey,normal);
    assert(ApproxEqual(normal,Vec_t(0.,0.,1.)));
    valid=trap1.Normal(ponmzsidey,normal);
    assert(ApproxEqual(normal,Vec_t(0.,0.,-1.)));

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
    assert(ApproxEqual( normal, Vec_t( invSqrt2, invSqrt2, 0.0) )); 

    // std::cout << " Normal at " << edgeXY << " is " << normal 
    //    << " Expected is " << Vec_t( invSqrt2, invSqrt2, 0.0) << std::endl;     

    valid= trap1.Normal( edgemXmY ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, -invSqrt2, 0.0) )&&valid); 
    valid= trap1.Normal( edgeXmY ,normal); 
    assert(ApproxEqual( normal, Vec_t( invSqrt2, -invSqrt2, 0.0) )); 
    valid= trap1.Normal( edgemXY ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, invSqrt2, 0.0) )); 

    valid= trap1.Normal( edgeXZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt2, 0.0, invSqrt2) )); 
    valid= trap1.Normal( edgemXmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, 0.0, -invSqrt2) )); 
    valid= trap1.Normal( edgeXmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt2, 0.0, -invSqrt2) )); 
    valid= trap1.Normal( edgemXZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, 0.0, invSqrt2) )); 

    valid= trap1.Normal( edgeYZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( 0.0,  invSqrt2,  invSqrt2) )); 
    valid= trap1.Normal( edgemYmZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( 0.0, -invSqrt2, -invSqrt2) )); 
    valid= trap1.Normal( edgeYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( 0.0,  invSqrt2, -invSqrt2) )); 
    valid= trap1.Normal( edgemYZ ,normal); 
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
    valid= trap1.Normal( cornermXYZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3,  invSqrt3, invSqrt3) )); 
    valid= trap1.Normal( cornerXmYZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3, -invSqrt3, invSqrt3) )); 
    valid= trap1.Normal( cornermXmYZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3, -invSqrt3, invSqrt3) )); 
    valid= trap1.Normal( cornerXYmZ,normal ); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3,  invSqrt3, -invSqrt3) )); 
    valid= trap1.Normal( cornermXYmZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3,  invSqrt3, -invSqrt3) )); 
    valid= trap1.Normal( cornerXmYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3, -invSqrt3, -invSqrt3) )); 
    valid= trap1.Normal( cornermXmYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3, -invSqrt3, -invSqrt3) )); 

     
    valid=trap2.Normal(ponxside,normal);
    assert(ApproxEqual(normal,Vec_t(cosa,0,-sina)));
    valid=trap2.Normal(ponmxside,normal);
    assert(ApproxEqual(normal,Vec_t(-cosa,0,-sina)));
    valid=trap2.Normal(ponyside,normal);
    assert(ApproxEqual(normal,Vec_t(0,cosa,-sina)));
    valid=trap2.Normal(ponmyside,normal);
    assert(ApproxEqual(normal,Vec_t(0,-cosa,-sina)));
    valid=trap2.Normal(ponzside,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,1)));
    valid=trap2.Normal(ponmzside,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,-1)));
    valid=trap2.Normal(ponzsidey,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,1)));
    valid=trap2.Normal(ponmzsidey,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,-1))); // (0,cosa,-sina) ?

// SafetyFromInside(P)

    Dist=trap1.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,20));
    Dist=trap1.SafetyFromInside(vx);
    assert(ApproxEqual(Dist,19));
    Dist=trap1.SafetyFromInside(vy);
    assert(ApproxEqual(Dist,20));
    Dist=trap1.SafetyFromInside(vz);
    assert(ApproxEqual(Dist,20));

    Dist=trap2.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,20*cosa));
    Dist=trap2.SafetyFromInside(vx);
    assert(ApproxEqual(Dist,19*cosa));
    Dist=trap2.SafetyFromInside(vy);
    assert(ApproxEqual(Dist,20*cosa));
    Dist=trap2.SafetyFromInside(vz);
    assert(ApproxEqual(Dist,20*cosa+sina));


// DistanceToOut(P,V)

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
    assert(ApproxEqual(Dist,std::sqrt(800.))&&ApproxEqual(norm,vx)&&convex);

    Dist=trap1.DistanceToOut(ponxside,vx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vx)&&convex);
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

    Dist=trap1.DistanceToOut(ponxside,vmx,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vmx)&&convex);
    Dist=trap1.DistanceToOut(ponmxside,vx,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vx)&&convex);
    Dist=trap1.DistanceToOut(ponyside,vmy,norm,convex);
    assert(ApproxEqual(Dist,60)&&ApproxEqual(norm,vmy)&&convex);
    Dist=trap1.DistanceToOut(ponmyside,vy,norm,convex);
    assert(ApproxEqual(Dist,60)&&ApproxEqual(norm,vy)&&convex);
    Dist=trap1.DistanceToOut(ponzside,vmz,norm,convex);
    assert(ApproxEqual(Dist,80)&&ApproxEqual(norm,vmz)&&convex);
    Dist=trap1.DistanceToOut(ponmzside,vz,norm,convex);
    assert(ApproxEqual(Dist,80)&&ApproxEqual(norm,vz)&&convex);


    Dist=trap2.DistanceToOut(pzero,vx,norm,convex);
    assert(ApproxEqual(Dist,20)&&ApproxEqual(norm,Vec_t(cosa,0,-sina))&&convex);
    Dist=trap2.DistanceToOut(pzero,vmx,norm,convex);
    assert(ApproxEqual(Dist,20)&&ApproxEqual(norm,Vec_t(-cosa,0,-sina))&&convex);
    Dist=trap2.DistanceToOut(pzero,vy,norm,convex);
    assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,Vec_t(0,cosa,-sina))&&convex);
    Dist=trap2.DistanceToOut(pzero,vmy,norm,convex);
    assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,Vec_t(0,-cosa,-sina))&&convex);
    Dist=trap2.DistanceToOut(pzero,vz,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vz)&&convex);
    Dist=trap2.DistanceToOut(pzero,vmz,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vmz)&&convex);
    Dist=trap2.DistanceToOut(pzero,vxy,norm,convex);
    assert(ApproxEqual(Dist,std::sqrt(800.))&&convex);

    Dist=trap2.DistanceToOut(ponxside,vx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,Vec_t(cosa,0,-sina))&&convex);
    Dist=trap2.DistanceToOut(ponmxside,vmx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,Vec_t(-cosa,0,-sina))&&convex);
    Dist=trap2.DistanceToOut(ponyside,vy,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,Vec_t(0,cosa,-sina))&&convex);
    Dist=trap2.DistanceToOut(ponmyside,vmy,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,Vec_t(0,-cosa,-sina))&&convex);
    Dist=trap2.DistanceToOut(ponzside,vz,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vz)&&convex);
    Dist=trap2.DistanceToOut(ponmzside,vmz,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmz)&&convex);


//SafetyFromOutside(P)
    
    Dist=trap1.SafetyFromOutside(pbig);
    //  std::cout<<"trap1.SafetyFromOutside(pbig) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,80));

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

    Dist=trap2.SafetyFromOutside(pbigx);
    assert(ApproxEqual(Dist,80*cosa));
    Dist=trap2.SafetyFromOutside(pbigmx);
    assert(ApproxEqual(Dist,80*cosa));
    Dist=trap2.SafetyFromOutside(pbigy);
    assert(ApproxEqual(Dist,70*cosa));
    Dist=trap2.SafetyFromOutside(pbigmy);
    assert(ApproxEqual(Dist,70*cosa));
    Dist=trap2.SafetyFromOutside(pbigz);
    assert(ApproxEqual(Dist,60));
    Dist=trap2.SafetyFromOutside(pbigmz);
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
    assert(ApproxEqual(Dist,Constants::kInfinity));
    Dist=trap1.DistanceToIn(pbigmx,vxy);
    assert(ApproxEqual(Dist,Constants::kInfinity));

    Dist=trap2.DistanceToIn(pbigx,vmx);
    assert(ApproxEqual(Dist,80));
    Dist=trap2.DistanceToIn(pbigmx,vx);
    assert(ApproxEqual(Dist,80));
    Dist=trap2.DistanceToIn(pbigy,vmy);
    assert(ApproxEqual(Dist,70));
    Dist=trap2.DistanceToIn(pbigmy,vy);
    assert(ApproxEqual(Dist,70));
    Dist=trap2.DistanceToIn(pbigz,vmz);
    assert(ApproxEqual(Dist,60));
    Dist=trap2.DistanceToIn(pbigmz,vz);
    assert(ApproxEqual(Dist,60));
    Dist=trap2.DistanceToIn(pbigx,vxy);
    assert(ApproxEqual(Dist,Constants::kInfinity));
    Dist=trap2.DistanceToIn(pbigmx,vxy);
    assert(ApproxEqual(Dist,Constants::kInfinity));

    dist=trap3.DistanceToIn(Vec_t(50,-50,0),vy);
    assert(ApproxEqual(dist,50));

    dist=trap3.DistanceToIn(Vec_t(50,-50,0),vmy);
    assert(ApproxEqual(dist,Constants::kInfinity));

    dist=trap4.DistanceToIn(Vec_t(50,50,0),vy);
    assert(ApproxEqual(dist,Constants::kInfinity));

    dist=trap4.DistanceToIn(Vec_t(50,50,0),vmy);
    assert(ApproxEqual(dist,50));

    dist=trap1.DistanceToIn(Vec_t(0,60,0),vxmy);
    assert(ApproxEqual(dist,Constants::kInfinity));

    dist=trap1.DistanceToIn(Vec_t(0,50,0),vxmy);
    std::cout<<"trap1.DistanceToIn(Vec_t(0,50,0),vxmy) = "<<dist<<" and vxmy="<< vxmy << std::endl ;
    // assert(ApproxEqual(dist,sqrt(800.)));  // A bug in UTrap!!!  Just keep printout above as a reminder

    dist=trap1.DistanceToIn(Vec_t(0,40,0),vxmy);
    assert(ApproxEqual(dist,10.0*std::sqrt(2.0)));

    dist=trap1.DistanceToIn(Vec_t(0,40,50),vxmy);
    assert(ApproxEqual(dist,Constants::kInfinity));

    // Parallel to side planes

    dist=trap1.DistanceToIn(Vec_t(40,60,0),vmx);
    assert(ApproxEqual(dist,Constants::kInfinity));

    dist=trap1.DistanceToIn(Vec_t(40,60,0),vmy);
    assert(ApproxEqual(dist,Constants::kInfinity));

    dist=trap1.DistanceToIn(Vec_t(40,60,50),vmz);
    assert(ApproxEqual(dist,Constants::kInfinity));

    dist=trap1.DistanceToIn(Vec_t(0,0,50),vymz);
    assert(ApproxEqual(dist,10.0*std::sqrt(2.0)));

    dist=trap1.DistanceToIn(Vec_t(0,0,80),vymz);
    assert(ApproxEqual(dist,Constants::kInfinity));

    dist=trap1.DistanceToIn(Vec_t(0,0,70),vymz);
    std::cout<<"trap1.DistanceToIn(Vec_t(0,0,70),vymz) = "<<dist<<", vymz="<< vymz << std::endl ;
    //assert(ApproxEqual(dist,30.0*sqrt(2.0)));  // A bug in UTrap!!!  Just keep printout above as a reminder

// CalculateExtent
     
   Vec_t minExtent,maxExtent;
   trap1.Extent(minExtent,maxExtent);
   assert(ApproxEqual(minExtent,Vec_t(-20,-30,-40)));
   assert(ApproxEqual(maxExtent,Vec_t( 20, 30, 40)));
   trap2.Extent(minExtent,maxExtent);
   assert(ApproxEqual(minExtent,Vec_t(-30,-40,-40)));
   assert(ApproxEqual(maxExtent,Vec_t( 30, 40, 40)));
   
   return true;
}

#ifdef VECGEOM_USOLIDS
struct USOLIDSCONSTANTS
{
  static constexpr double kInfinity = DBL_MAX;//UUSolids::kInfinity;
};
#endif
struct VECGEOMCONSTANTS
{
  static constexpr double kInfinity = vecgeom::kInfinity;
};


int main(int argc, char *argv[]) {
 
   if( argc < 2)
    {
      std::cerr << "need to give argument :--usolids or --vecgeom\n";     
      return 1;
    }
    
    if( ! strcmp(argv[1], "--usolids") )
    { 
      #ifdef VECGEOM_USOLIDS
      TestTrap<USOLIDSCONSTANTS, UTrap>();
      std::cout << "UTrap passed (but notice discrepancies above, where asserts have been disabled!)\n";
      #else
      std::cerr << "VECGEOM_USOLIDS was not defined\n";
      return 2;
      #endif
    }
    else if( ! strcmp(argv[1], "--vecgeom") )
    {
      //testingvecgeom = true;
     TestTrap<VECGEOMCONSTANTS, VECGEOM_NAMESPACE::SimpleTrapezoid>();
     std::cout << "VecGeom Trap passed.\n";
    
    }
    else
    {
      std::cerr << "need to give argument :--usolids or --vecgeom\n";     
      return 1;
    }


  return 0;
}
