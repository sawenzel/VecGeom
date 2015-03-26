//
//
// TestBox
//             Ensure asserts are compiled in

#undef NDEBUG

#include "base/Vector3D.h"
#include "volumes/Tube.h"
#include "ApproxEqual.h"
#ifdef VECGEOM_USOLIDS
#include "UTubs.hh"
#include "UVector3.hh"
#endif

//#include <cassert>
#include <cmath>

#define PI 3.14159265358979323846


template <class Tube_t,class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >

bool TestTubs()
{
    std::cout.precision(16) ;
    VUSolid::EnumInside side;
    Vec_t pzero(0,0,0);
	Vec_t ptS(0,0,0);

    double kCarTolerance = VUSolid::Tolerance();

    Vec_t pbigx(100,0,0),pbigy(0,100,0),pbigz(0,0,100);
    Vec_t pbigmx(-100,0,0),pbigmy(0,-100,0),pbigmz(0,0,-100);

    Vec_t ponxside(50,0,0);
    Vec_t ponyside(0,50,0);
    Vec_t ponzside( 0,0,50);

    Vec_t vx(1,0,0),vy(0,1,0),vz(0,0,1);
    Vec_t vmx(-1,0,0),vmy(0,-1,0),vmz(0,0,-1);
    Vec_t vxy(1/std::sqrt(2.0),1/std::sqrt(2.0),0);
    Vec_t vmxy(-1/std::sqrt(2.0),1/std::sqrt(2.0),0);
    Vec_t vmxmy(-1/std::sqrt(2.0),-1/std::sqrt(2.0),0);
    Vec_t vxmy(1/std::sqrt(2.0),-1/std::sqrt(2.0),0);


    double Dist, vol,volCheck;
      
    Tube_t t1("Solid Tube #1",0,50,50,0,2*UUtils::kPi);

    Tube_t t1a("Solid Tube #1",0,50,50,0,0.5*UUtils::kPi);

    Tube_t t2("Hole Tube #2",45,50 ,50 ,0,2*UUtils::kPi);

    Tube_t t2a("Hole Tube #2",5 ,50 ,50 ,0,2*UUtils::kPi);

    Tube_t t2b("Hole Tube #2",15 ,50 ,50 ,0,2*UUtils::kPi);

    Tube_t t2c("Hole Tube #2",25 ,50 ,50 ,0,2*UUtils::kPi);

    Tube_t t2d("Hole Tube #2",35 ,50 ,50 ,0,2*UUtils::kPi);

    Tube_t t3("Solid Sector #3",0,50 ,50 ,0.5*UUtils::kPi,0.5*UUtils::kPi);

    Tube_t t4("Hole Sector #4",45 ,50 ,50 ,0.5*UUtils::kPi,0.5*UUtils::kPi);

    Tube_t t5("Hole Sector #5",50 ,100 ,50 ,0.0,1.5*UUtils::kPi);
    
    Tube_t t6("Solid Sector #3",0,50 ,50 ,0.5*UUtils::kPi,1.5*UUtils::kPi);

  Tube_t tube6("tube6",750,760,350,0.31415926535897931,5.6548667764616276);

  Tube_t tube7("tube7",2200,3200,2500,-0.68977164349384879,3.831364227270472);

  Tube_t tube8("tube8",2550,2580,2000,0,2*UUtils::kPi);

  Tube_t tube9("tube9",1150,1180,2000,0,2*UUtils::kPi);

  Tube_t tube10("tube10",400 ,405 ,400 ,0,2*UUtils::kPi) ;

  Tube_t* clad =
      new Tube_t("clad",90.,110.,105,0.,UUtils::kPi);    // external

  Tube_t* core =
      new Tube_t("core",95.,105.,100,0.,UUtils::kPi); // internal
  


    std::cout.precision(20);

// Check name
   assert(t1.GetName()=="Solid Tube #1");

  // Check cubic volume
  vol = t1.Capacity();
  volCheck = 50*2*UUtils::kPi*50*50;
  assert(ApproxEqual(vol,volCheck));
 
  // Check Surface area
  // vol = t2.SurfaceArea();
  // volCheck = 2*UUtils::kPi*(45+50)*(50-45+2*50);
  // assert(ApproxEqual(vol,volCheck));


  	Tube_t myClad("myClad", 90.0, 110.0, 105.0, 0.0, PI);    // TEST MINE
	
	long int totPointOnSurface = 2000000;
	std::cout << " Generating " << totPointOnSurface << " points on the surface of the tube ." << std::endl;
	long int totInSurf = 0, totOutSurf = 0, totTop = 0, totBot = 0, totL = 0, totR = 0, added = 0;
  
	for (long int i = 0; i < totPointOnSurface; i++) {
	  ptS = myClad.GetPointOnSurface(); 
	  double x2y2 = sqrt(ptS[0]*ptS[0]+ptS[1]*ptS[1]);
	  double fival = sqrt(atan2(ptS[1], ptS[0]));  //std::cout << " fival = " << fival*180.0/3.14 << std::endl;
	  double fi1 = 0.0;
	  double fi2 = PI;
	  double zz = ptS[2];
	  double eps = 1.0e-8;
	  bool surfaceIn = (abs(x2y2 - 90.0) <= eps);
	  bool surfaceOut = (abs(x2y2 - 110.0) <= eps);
	  bool topSurface = (abs(zz - 105) <= eps); // (x2y2 >= 90.0) && (x2y2 <= 110.0) && (abs(zz - 105) <= eps);
	  bool botSurface = (abs(zz + 105) <= eps); // (x2y2 >= 90.0) && (x2y2 <= 110.0) && (abs(zz + 105) <= eps);
	  bool fiLeft = (abs(fival - fi1) <= eps);
	  bool fiRight = (abs(fival - fi2) <= eps);
	  bool Tagged = 0;
	  //std::cout << i;
	  if (fiLeft && !Tagged) {
		  totL++;  added++; Tagged++; //std::cout << " FiL "; 
	  }
	  if (fiRight && !Tagged) {
		  totR++;  added++; Tagged++; //std::cout << " FiR ";
	  }
	  if (surfaceIn && !Tagged) {
		  totInSurf++; added++; Tagged++;  //std::cout << " In  ";
	  }
	  if (surfaceOut && !Tagged) {
		  totOutSurf++; added++; Tagged++;  //std::cout << " Out "; 
	  }
	  if (topSurface && !Tagged) {
		  totTop++; added++; Tagged++; // std::cout << " Top "; 
	  }
	  if (botSurface && !Tagged) {
		  totBot++;  added++; Tagged++; //std::cout << " Bot ";
	  }
	  //std::cout << std::endl;
	  if (!Tagged && !fiLeft && !fiRight && !surfaceIn && !surfaceOut && !topSurface && !botSurface && fival > 3.0 && fival < 3.16) std::cout << ptS << std::endl;
  }
	std::cout << std::endl;
  std::cout << " Total Inner surface = " << totInSurf << " (" << ((totInSurf*1.0)/totPointOnSurface)*100.0 << "%)  Total Outer surface = " << totOutSurf << " (" << ((totOutSurf*1.0)/totPointOnSurface)*100.0 << "%) Total Top surface = " << totTop << " (" << ((totTop*1.0)/totPointOnSurface)*100.0 << "%) Total Bottom Surface = " << totBot << " (" << ((totBot*1.0)/totPointOnSurface)*100.0 << "%) " << " Total Left Surface = " << totL << " (" << ((totL*1.0)/totPointOnSurface)*100.0 << "%) Total Right Surface = " << totR << " (" << ((totR*1.0)/totPointOnSurface)*100.0 << "%) " << std::endl;

  std::cout << "Total point = " << totPointOnSurface << std::endl <<"total added = " << added << std::endl;

// Check Inside
    assert(t1.Inside(pzero)==vecgeom::EInside::kInside);
    assert(t1.Inside(pbigz)==vecgeom::EInside::kOutside);
    assert(t1.Inside(ponxside)==vecgeom::EInside::kSurface);
    assert(t1.Inside(ponyside)==vecgeom::EInside::kSurface);
    assert(t1.Inside(ponzside)==vecgeom::EInside::kSurface);
    assert(t1a.Inside(pzero)==vecgeom::EInside::kSurface);
    assert(t1a.Inside(pbigz)==vecgeom::EInside::kOutside);
    assert(t1a.Inside(ponxside)==vecgeom::EInside::kSurface);
    assert(t1a.Inside(ponyside)==vecgeom::EInside::kSurface);
    assert(t1a.Inside(ponzside)==vecgeom::EInside::kSurface);
    assert(t2.Inside(pzero)==vecgeom::EInside::kOutside);
    assert(t2.Inside(pbigz)==vecgeom::EInside::kOutside);
    assert(t2.Inside(ponxside)==vecgeom::EInside::kSurface);
    assert(t2.Inside(ponyside)==vecgeom::EInside::kSurface);
    assert(t2.Inside(ponzside)==vecgeom::EInside::kOutside);
    assert(t2a.Inside(pzero)==vecgeom::EInside::kOutside);
    assert(t2a.Inside(pbigz)==vecgeom::EInside::kOutside);
    assert(t2a.Inside(ponxside)==vecgeom::EInside::kSurface);
    assert(t2a.Inside(ponyside)==vecgeom::EInside::kSurface);
    assert(t2a.Inside(ponzside)==vecgeom::EInside::kOutside);

// Check Surface Normal
    Vec_t normal;
    bool valid;
    Vec_t norm;
    double p2=1./std::sqrt(2.),p3=1./std::sqrt(3.);
    valid=t1.Normal(ponxside,normal);
    assert(ApproxEqual(normal,vx));

    valid=t4.Normal(Vec_t(0.,50.,0.),normal);
    assert(ApproxEqual(normal,Vec_t(p2,p2,0.))&&valid);
    valid=t4.Normal(Vec_t(0.,45.,0.),normal);
    assert(ApproxEqual(normal,Vec_t(p2,-p2,0.)));
    valid=t4.Normal(Vec_t(0.,45.,50.),normal);
    assert(ApproxEqual(normal,Vec_t(p3,-p3,p3)));
    valid=t4.Normal(Vec_t(0.,45.,-50.),normal);
    assert(ApproxEqual(normal,Vec_t(p3,-p3,-p3)));
    valid=t4.Normal(Vec_t(-50.,0.,-50.),normal);
    assert(ApproxEqual(normal,Vec_t(-p3,-p3,-p3)));
    valid=t4.Normal(Vec_t(-50.,0.,0.),normal);
    assert(ApproxEqual(normal,Vec_t(-p2,-p2,0.)));
    valid=t6.Normal(Vec_t(0.,0.,0.),normal);
    assert(ApproxEqual(normal,Vec_t(p2,p2,0.)));

// SafetyFromInside(P)
    Dist=t1.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,50));

// DistanceToOut(P,V)
    bool convex;
    Dist=t1.DistanceToOut(pzero,vx,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vx)&&convex);
    Dist=t1.DistanceToOut(pzero,vmx,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vmx)&&convex);
    Dist=t1.DistanceToOut(pzero,vy,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vy)&&convex);
    Dist=t1.DistanceToOut(pzero,vmy,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vmy)&&convex);
    Dist=t1.DistanceToOut(pzero,vz,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vz)&&convex);
    Dist=t1.DistanceToOut(pzero,vmz,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vmz)&&convex);
    Dist=t1.DistanceToOut(pzero,vxy,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vxy)&&convex);

    Dist=t2.DistanceToOut(pzero,vxy,norm,convex);
    //  std::cout<<"Dist=t2.DistanceToOut(pzero,vxy) = "<<Dist<<std::endl;

    Dist=t2.DistanceToOut(ponxside,vmx,norm,convex);
    //  std::cout<<"Dist=t2.DistanceToOut(ponxside,vmx) = "<<Dist<<std::endl;

    Dist=t2.DistanceToOut(ponxside,vmxmy,norm,convex);
    //  std::cout<<"Dist=t2.DistanceToOut(ponxside,vmxmy) = "<<Dist<<std::endl;

    Dist=t2.DistanceToOut(ponxside,vz,norm,convex);
    //  std::cout<<"Dist=t2.DistanceToOut(ponxside,vz) = "<<Dist<<std::endl;

    Dist=t2.DistanceToOut(pbigx,vx,norm,convex);
    //   std::cout<<"Dist=t2.DistanceToOut(pbigx,vx) = "<<Dist<<std::endl;

    Dist=t2.DistanceToOut(pbigx,vxy,norm,convex);
    //   std::cout<<"Dist=t2.DistanceToOut(pbigx,vxy) = "<<Dist<<std::endl;

    Dist=t2.DistanceToOut(pbigx,vz,norm,convex);
    //   std::cout<<"Dist=t2.DistanceToOut(pbigx,vz) = "<<Dist<<std::endl;

    Dist=t2.DistanceToOut(Vec_t(45.5,0,0),vx,norm,convex);
    //  std::cout<<"Dist=t2.DistanceToOut((45.5,0,0),vx) = "<<Dist<<std::endl;

    Dist=t2.DistanceToOut(Vec_t(49.5,0,0),vx,norm,convex);
    //  std::cout<<"Dist=t2.DistanceToOut((49.5,0,0),vx) = "<<Dist<<std::endl;


    Dist=t3.DistanceToOut(Vec_t(0,10,0),vx,norm,convex);
    // std::cout<<"Dist=t3.DistanceToOut((0,10,0),vx) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,0));

    Dist=t3.DistanceToOut(Vec_t(0.5,10,0),vx,norm,convex);
    // std::cout<<"Dist=t3.DistanceToOut((0.5,10,0),vx) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,48.489795));

    Dist=t3.DistanceToOut(Vec_t(-0.5,9,0),vx,norm,convex);
    // std::cout<<"Dist=t3.DistanceToOut((-0.5,9,0),vx) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,0.5));

    Dist=t3.DistanceToOut(Vec_t(-5,9.5,0),vx,norm,convex);
    // std::cout<<"Dist=t3.DistanceToOut((-5,9.5,0),vx) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,5));

    Dist=t3.DistanceToOut(Vec_t(-5,9.5,0),vmy,norm,convex);
    // std::cout<<"Dist=t3.DistanceToOut((-5,9.5,0),vmy) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,9.5));

    Dist=t3.DistanceToOut(Vec_t(-5,9,0),vxmy,norm,convex);
    // std::cout<<"Dist=t3.DistanceToOut((-5,9,0),vxmy) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,7.0710678));

       
//SafetyFromOutside(P)

    Dist=t1.SafetyFromOutside(pbigx);
    assert(ApproxEqual(Dist,50));
    Dist=t1.SafetyFromOutside(pbigmx);
    assert(ApproxEqual(Dist,50));
    Dist=t1.SafetyFromOutside(pbigy);
    assert(ApproxEqual(Dist,50));
    Dist=t1.SafetyFromOutside(pbigmy);
    assert(ApproxEqual(Dist,50));
    Dist=t1.SafetyFromOutside(pbigz);
    assert(ApproxEqual(Dist,50));
    Dist=t1.SafetyFromOutside(pbigmz);
    assert(ApproxEqual(Dist,50));

// DistanceToIn(P,V)

    Dist=t1.DistanceToIn(pbigx,vmx);
    assert(ApproxEqual(Dist,50));
    Dist=t1.DistanceToIn(pbigmx,vx);
    assert(ApproxEqual(Dist,50));
    Dist=t1.DistanceToIn(pbigy,vmy);
    assert(ApproxEqual(Dist,50));
    Dist=t1.DistanceToIn(pbigmy,vy);
    assert(ApproxEqual(Dist,50));
    Dist=t1.DistanceToIn(pbigz,vmz);
    assert(ApproxEqual(Dist,50));
    Dist=t1.DistanceToIn(pbigmz,vz);
    assert(ApproxEqual(Dist,50));
    Dist=t1.DistanceToIn(pbigx,vxy);
    assert(ApproxEqual(Dist,UUtils::kInfinity));

    Dist=t1a.DistanceToIn(pbigz,vmz);
    assert(ApproxEqual(Dist,50));

    Dist=t2.DistanceToIn(Vec_t(45.5,0,0),vx);
    //  std::cout<<"Dist=t2.DistanceToIn((45.5,0,0),vx) = "<<Dist<<std::endl;
   
    Dist=t2.DistanceToIn(Vec_t(45.5,0,0),vmx);
    //  std::cout<<"Dist=t2.DistanceToIn((45.5,0,0),vmx) = "<<Dist<<std::endl;
   
    Dist=t2.DistanceToIn(Vec_t(49.5,0,0),vmx);
    //  std::cout<<"Dist=t2.DistanceToIn((49.5,0,0),vmx) = "<<Dist<<std::endl;
   
    Dist=t2.DistanceToIn(Vec_t(49.5,0,0),vx);
    //   std::cout<<"Dist=t2.DistanceToIn((49.5,0,0),vx) = "<<Dist<<std::endl;
   
    Dist=t3.DistanceToIn(Vec_t(49.5,0,0),vmx);
    //  std::cout<<"Dist=t2.DistanceToIn((49.5,0,0),vmx) = "<<Dist<<std::endl;
   
    Dist=t3.DistanceToIn(Vec_t(49.5,5,0),vmx);
    //  std::cout<<"Dist=t2.DistanceToIn((49.5,5,0),vmx) = "<<Dist<<std::endl;
   
    Dist=t3.DistanceToIn(Vec_t(49.5,-0.5,0),vmx);
    //  std::cout<<"Dist=t2.DistanceToIn((49.5,-0.5,0),vmx) = "<<Dist<<std::endl;
   
    Dist=t5.DistanceToIn(Vec_t(30.0,-20.0,0),vxy);
    // std::cout<<"Dist=t5.DistanceToIn((30.0,-20.0,0),vxy) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,28.284271));
   
    Dist=t5.DistanceToIn(Vec_t(30.0,-70.0,0),vxy);
    // std::cout<<"Dist=t5.DistanceToIn((30.0,-70.0,0),vxy) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,UUtils::kInfinity));
   
    Dist=t5.DistanceToIn(Vec_t(30.0,-20.0,0),vmxmy);
    //  std::cout<<"Dist=t5.DistanceToIn((30.0,-20.0,0),vmxmy) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,42.426407));
   
    Dist=t5.DistanceToIn(Vec_t(30.0,-70.0,0),vmxmy);
    // std::cout<<"Dist=t5.DistanceToIn((30.0,-70.0,0),vmxmy) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,UUtils::kInfinity));
   
    Dist=t5.DistanceToIn(Vec_t(50.0,-20.0,0),vy);
    // std::cout<<"Dist=t5.DistanceToIn((50.0,-20.0,0),vy) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,20));

    Dist=t5.DistanceToIn(Vec_t(100.0,-20.0,0),vy);
    // std::cout<<"Dist=t5.DistanceToIn((100.0,-20.0,0),vy) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,UUtils::kInfinity));
   
    Dist=t5.DistanceToIn(Vec_t(30.0,-50.0,0),vmx);
    //  std::cout<<"Dist=t5.DistanceToIn((30.0,-50.0,0),vmx) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,30));
   
    Dist=t5.DistanceToIn(Vec_t(30.0,-100.0,0),vmx);
    //  std::cout<<"Dist=t5.DistanceToIn((30.0,-100.0,0),vmx) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,UUtils::kInfinity));
   
    /* ********************************
       ************************************ */
    //Tubs from Problem reports

       //
	// Make a tub
	//
	Tube_t *arc = new Tube_t( "outer", 1000, 1100, 10, -UUtils::kPi/12., UUtils::kPi/6. );
	
	//
	// First issue: 
	//   A point on the start phi surface just beyond the
	//   start angle but still well within tolerance 
	//   is found to be "outside" by Tube_t::Inside
	//
	//   pt1 = exactly on phi surface (within precision)
	//   pt2 = t1 but slightly higher, and still on tolerant surface
	//   pt3 = t1 but slightly lower, and still on tolerant surface
	//
	Vec_t pt1( 1050*std::cos(-UUtils::kPi/12.),
	                   1050*std::sin(-UUtils::kPi/12.),
			      0 );
 			  
        Vec_t pt2 = pt1 + Vec_t(0,0.001*kCarTolerance,0) ;
        Vec_t pt3 = pt1 - Vec_t(0,0.001*kCarTolerance,0) ;
	
	VUSolid::EnumInside a1 = arc->Inside(pt1);
	VUSolid::EnumInside a2 = arc->Inside(pt2);
	VUSolid::EnumInside a3 = arc->Inside(pt3);
	
	//std::cout << "Point pt1 is " << OutputInside(a1) << std::endl;
        assert(a1==vecgeom::EInside::kSurface);
	//std::cout << "Point pt2 is " << OutputInside(a2) << std::endl;
        assert(a2==vecgeom::EInside::kSurface);
	//std::cout << "Point pt3 is " << OutputInside(a3) << std::endl;
	assert(a3==vecgeom::EInside::kSurface);


    assert(t1.Inside(pzero)==vecgeom::EInside::kInside);
    assert(t1.Inside(pbigx)==vecgeom::EInside::kOutside);

    VUSolid::EnumInside in = t5.Inside(Vec_t(60,-0.001*kCarTolerance,0)) ;
    assert(in == vecgeom::EInside::kSurface);
    //    std::cout<<"t5.Inside(Vec_t(60,-0.001*kCarTolerance,0)) = "
    //     <<OutputInside(in)<<std::endl;
    in = tube10.Inside(Vec_t(-114.8213313833317 ,
					   382.7843220719649 ,
                                           -32.20788536438663 )) ;
    assert(in == vecgeom::EInside::kOutside);
    // std::cout<<"tube10.Inside(Vec_t(-114.821...)) = "<<OutputInside(in)<<std::endl;

     
       // bug #76
    Dist=tube6.DistanceToOut(
    Vec_t(-388.20504321896431,-641.71398957741451,332.85995254027955),
    Vec_t(-0.47312863350457468,-0.782046391443315, 0.40565100491504164),
    norm,convex);
    // std::cout<<"Dist=tube6.DistanceToOut(p,v) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,10.940583));

      // bug #91
    Dist=tube7.DistanceToOut(
    Vec_t(-2460,1030,-2500),
    Vec_t(-0.086580540180167642,0.070084247882560638,0.9937766390194761),
    norm,convex);
    // std::cout<<"Dist=tube7.DistanceToOut(p,v) = "<<Dist<<std::endl;
    // assert(ApproxEqual(Dist,4950.348576972614));

    Dist=tube8.DistanceToOut(
 Vec_t(6.71645645882942,2579.415860329989,-1.519530725281157),
 Vec_t(-0.6305220496340839,-0.07780451841562354,0.7722618738739774),
 norm,convex);
    // std::cout<<"Dist=tube8.DistanceToOut(p,v) = "<<Dist<<std::endl;
    // assert(ApproxEqual(Dist,4950.348576972614));

    Dist=tube9.DistanceToOut(
 Vec_t(2.267347771505638,1170.164934028592,4.820317321984064),
 Vec_t(-0.1443054266272111,-0.01508874701037938,0.9894181489944458),
    norm,convex);
    // std::cout<<"Dist=tube9.DistanceToOut(p,v) = "<<Dist<<std::endl;
    // assert(ApproxEqual(Dist,4950.348576972614));

    Dist=t1a.DistanceToOut(Vec_t(0.,0.,50.),vx,norm,convex);
    //std::cout<<"Dist=t1a.DistanceToOut((0,0,50),vx) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,50));

    Dist=t1a.DistanceToOut(Vec_t(0.,5.,50.),vmy,norm,convex);
    //std::cout<<"Dist=t1a.DistanceToOut((0,5,50),vmy) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,5));

    std::cout<<std::endl ;

    // Bug 810

    Vec_t pTmp(0.,0.,0.);

    Dist = clad->DistanceToIn(pTmp,vy);   
    pTmp += Dist*vy;
    //std::cout<<"pTmpX = "<<pTmp.x<<";  pTmpY = "<<pTmp.y<<";  pTmpZ = "<<pTmp.z<<std::endl;
    side=core->Inside(pTmp);    
    assert(side==vecgeom::EInside::kOutside);
    //std::cout<<"core->Inside(pTmp) = "<<OutputInside(side)<<std::endl;
    side=clad->Inside(pTmp);  
    assert(side==vecgeom::EInside::kSurface);  
    //std::cout<<"clad->Inside(pTmp) = "<<OutputInside(side)<<std::endl;

    Dist = core->DistanceToIn(pTmp,vy);   
    pTmp += Dist*vy;
    //std::cout<<"pTmpX = "<<pTmp.x<<";  pTmpY = "<<pTmp.y<<";  pTmpZ = "<<pTmp.z<<std::endl;
    side=core->Inside(pTmp);   
    assert(side==vecgeom::EInside::kSurface);   
    //std::cout<<"core->Inside(pTmp) = "<<OutputInside(side)<<std::endl;
    side=clad->Inside(pTmp);  
    assert(side==vecgeom::EInside::kInside);    
    //std::cout<<"clad->Inside(pTmp) = "<<OutputInside(side)<<std::endl;
    Dist = core->DistanceToOut(pTmp,vy,norm,convex);   
    pTmp += Dist*vy;
    //std::cout<<"pTmpX = "<<pTmp.x<<";  pTmpY = "<<pTmp.y<<";  pTmpZ = "<<pTmp.z<<std::endl;
    side=core->Inside(pTmp); 
    assert(side==vecgeom::EInside::kSurface);     
    //std::cout<<"core->Inside(pTmp) = "<<OutputInside(side)<<std::endl;
    side=clad->Inside(pTmp);  
    assert(side==vecgeom::EInside::kInside);    
    //std::cout<<"clad->Inside(pTmp) = "<<OutputInside(side)<<std::endl;

    Dist = clad->DistanceToOut(pTmp,vy,norm,convex);   
    pTmp += Dist*vy;
    //std::cout<<"pTmpX = "<<pTmp.x<<";  pTmpY = "<<pTmp.y<<";  pTmpZ = "<<pTmp.z<<std::endl;
    side=core->Inside(pTmp); 
    assert(side==vecgeom::EInside::kOutside);     
    //std::cout<<"core->Inside(pTmp) = "<<OutputInside(side)<<std::endl;
    side=clad->Inside(pTmp);  
    assert(side==vecgeom::EInside::kSurface);   
    //std::cout<<"clad->Inside(pTmp) = "<<OutputInside(side)<<std::endl;
     
    Vec_t pSN1 = Vec_t( 33.315052227388207, 37.284142675357259, 33.366096020078537);
    Tube_t t4SN("Hole Sector #4",45 ,50 ,50 ,UUtils::kPi/4.,UUtils::kPi/8.);

    in = t4SN.Inside(pSN1);
    assert(in == vecgeom::EInside::kSurface);
    valid = t4SN.Normal(pSN1,normal);
    

// CalculateExtent
    Vec_t minExtent,maxExtent;
    t1.Extent(minExtent,maxExtent);
    //std::cout<<" min="<<minExtent<<" max="<<maxExtent<<std::endl;
    assert(ApproxEqual(minExtent,Vec_t(-50,-50,-50)));
    assert(ApproxEqual(maxExtent,Vec_t( 50, 50, 50)));
    t2.Extent(minExtent,maxExtent);
    // std::cout<<" min="<<minExtent<<" max="<<maxExtent<<std::endl;
    assert(ApproxEqual(minExtent,Vec_t(-50,-50,-50)));
    assert(ApproxEqual(maxExtent,Vec_t( 50, 50, 50)));
  

    /* ********************************
    ************************************ */

    return true;
}




int main() {
#ifdef VECGEOM_USOLIDS
  assert(TestTubs<UTubs>());
  std::cout << "UTube passed\n";

#endif
  
   std::cout<< "VecGeomTube not included yet\n";
  return 0;
}
