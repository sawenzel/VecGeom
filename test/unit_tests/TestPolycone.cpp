//
//
// TestPolycone



#include "base/Vector3D.h"
#include "volumes/Polycone.h"
#include "volumes/Tube.h"
#include "volumes/Cone.h"
#include "volumes/LogicalVolume.h"
#include "volumes/PlacedVolume.h"
#include "ApproxEqual.h"
#ifdef VECGEOM_USOLIDS
#include "UPolycone.hh"
#include "UGenericPolycone.hh"
#include "UVector3.hh"
#endif
#include <cmath>

//             ensure asserts are compiled in
#undef NDEBUG
#include <cassert>

using namespace vecgeom;

bool testingvecgeom=false;



template <class Polycone_t,class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >

bool TestPolycone()
{
 double RMINVec[8];
  RMINVec[0] = 30;
  RMINVec[1] = 30;
  RMINVec[2] =  0;
  RMINVec[3] =  0;
  RMINVec[4] =  0;  
  RMINVec[5] =  0;
  RMINVec[6] = 40;
  RMINVec[7] = 40;  

  double RMAXVec[8];
  RMAXVec[0] = 70;
  RMAXVec[1] = 70;
  RMAXVec[2] = 70;
  RMAXVec[3] = 40;
  RMAXVec[4] = 40;
  RMAXVec[5] = 80;
  RMAXVec[6] = 80;
  RMAXVec[7] = 60; 

  double Z_Values[8];
  Z_Values[0] =-20;
  Z_Values[1] =-10;
  Z_Values[2] =-10;
  Z_Values[3] =  0;
  Z_Values[4] = 10;
  Z_Values[5] = 20;
  Z_Values[6] = 30;
  Z_Values[7] = 40;

  double Phi_Values[2];
  Phi_Values[0]=-10.*UUtils::kPi/180.;
  Phi_Values[1]=10.*UUtils::kPi/180.;
  Polycone_t *MyPCone = new Polycone_t ("MyPCone",
						    Phi_Values[0],
						    Phi_Values[1],
						    8        ,
						    Z_Values ,
						    RMINVec  ,
						    RMAXVec   );
  double RMIN[3];
  RMIN[0] = 0;
  RMIN[1] = 0;
  RMIN[2] = 0;
  double RMAX[3];
  RMAX[0] = 70;
  RMAX[1] = 70;
  RMAX[2] = 80;
  double Z_Val2[3];
  Z_Val2[0] =-10;
  Z_Val2[1] =-0;
  Z_Val2[2] = 10;
 
 Polycone_t Simple("SimpleTube+Cone",
		                      0,
		                      360.*UUtils::kPi/180.,
         	                      3      ,
		                      Z_Val2 ,
		                      RMIN ,
		                      RMAX );

if(testingvecgeom){

    int Nz = 4;
    // a tube and two cones
    double rmin[] = { 0.1, 0.0, 0.0 , 0.4 };
    double rmax[] = { 1., 2., 2. , 1.5 };
    double z[] = { -1, -0.5, 0.5, 2 };


            UnplacedPolycone poly1( 0.,    /* initial phi starting angle */
				360.*UUtils::kPi/180.,  //kTwoPi,    /* total phi angle */
            Nz,        /* number corners in r,z space */
	    z,       /* z coordinates */
            rmin,   /* r coordinate of these corners */
            rmax);

    poly1.Print();

    // lets make external separate tubes and cones representing the sections
    UnplacedCone section0(rmin[0], rmax[0], rmin[1], rmax[1], (z[1] - z[0])/2., 0, kTwoPi);
    UnplacedCone section1(rmin[1], rmax[1], rmin[2], rmax[2], (z[2] - z[1])/2., 0, kTwoPi);
    UnplacedCone section2(rmin[2], rmax[2], rmin[3], rmax[3], (z[3] - z[2])/2., 0, kTwoPi);


    assert( poly1.GetNz() == 4 );
    assert( poly1.GetNSections() == 3 );
    assert( poly1.GetSectionIndex( -0.8 ) == 0 );
    assert( poly1.GetSectionIndex( 0.51 ) == 2 );
    assert( poly1.GetSectionIndex( 0. ) == 1 );
    assert( poly1.GetSectionIndex( -2. ) == -1 );
    assert( poly1.GetSectionIndex( 3. ) == -2 );
    assert( poly1.GetStartPhi() == 0. );
    assert( (std::fabs(poly1.GetDeltaPhi()-kTwoPi))<1e-10 );

    assert(  poly1.fZs[0] == z[0] );
    assert(  poly1.fZs[poly1.GetNSections()] == z[Nz-1] );
    assert( poly1.Capacity() > 0 );
    assert( std::fabs(poly1.Capacity() - ( section0.Capacity() + section1.Capacity() + section2.Capacity() ))< 1e-6);

    // create a place version
    VPlacedVolume const * placedpoly1 = (new LogicalVolume("poly1", &poly1))->Place( new Transformation3D( ) );

    // test contains/inside
    assert( placedpoly1->Contains( Vec_t(0.,0.,0.) ) == true );
    assert( placedpoly1->Contains( Vec_t(0.,0.,-2.) ) == false );
    assert( placedpoly1->Contains( Vec_t(0.,0.,-0.8) ) == false );
    assert( placedpoly1->Contains( Vec_t(0.,0.,-1.8) ) == false );
    assert( placedpoly1->Contains( Vec_t(0.,0., 10) ) == false );
    assert( placedpoly1->Contains( Vec_t(0.,0., 1.8) ) == false );

     // test DistanceToIn
    assert( placedpoly1-> DistanceToIn( Vec_t(0.,0.,-3.) , Vec_t(0.,0.,1.)) == 2.5 );
    assert( placedpoly1-> DistanceToIn( Vec_t(0.,0.,-2.) , Vec_t(0.,0.,-1.)) == vecgeom::kInfinity );
    assert( placedpoly1-> DistanceToIn( Vec_t(0.,0.,3) , Vec_t(0.,0.,-1.)) == 2.5 );
    assert( placedpoly1-> DistanceToIn( Vec_t(0.,0.,3) , Vec_t(0.,0.,1.)) == vecgeom::kInfinity );
    assert( placedpoly1-> DistanceToIn( Vec_t(3.,0.,0) , Vec_t(-1.,0.,0.)) == 1 );
    assert( std::fabs(placedpoly1-> DistanceToIn( Vec_t(0.,0., 1.999999999) , Vec_t(1.,0.,0.)) -0.4)<1000.*kTolerance );

    // test SafetyToIn
    assert( placedpoly1-> SafetyToIn( Vec_t(0.,0.,-3.)) == 2. );
    assert( placedpoly1-> SafetyToIn( Vec_t(0.5,0.,-1.)) == 0. );
    assert( placedpoly1-> SafetyToIn( Vec_t(0.,0.,3) ) == 1 );
    assert( placedpoly1-> SafetyToIn( Vec_t(2.,0.,0.1) ) == 0 );
   
    // test SafetyToOut
   
    assert( placedpoly1-> SafetyToOut( Vec_t(0.,0.,0.)) == 0.5 );
    assert( placedpoly1-> SafetyToOut( Vec_t(0.,0.,0.5)) == 0. );
    assert( std::fabs(placedpoly1-> SafetyToOut( Vec_t(1.9,0.,0.0) ) - 0.1 )<1000.*kTolerance);
    assert( placedpoly1-> SafetyToOut( Vec_t(0.2,0.,-1) ) == 0. );
    assert( placedpoly1-> SafetyToOut( Vec_t(1.4,0.,2) ) == 0. );
    
    // test DistanceToOut
    assert( placedpoly1-> DistanceToOut( Vec_t(0.,0.,0.) , Vec_t(0.,0.,1.)) == 0.5 );
    assert( placedpoly1-> DistanceToOut( Vec_t(0.,0.,0.) , Vec_t(0.,0.,-1.)) == 0.5 );
    assert( placedpoly1-> DistanceToOut( Vec_t(2.,0.,0.) , Vec_t(1.,0.,0.)) == 0. );
    assert( placedpoly1-> DistanceToOut( Vec_t(2.,0.,0.) , Vec_t(-1.,0.,0.)) == 4. );
     
    assert( placedpoly1-> DistanceToOut( Vec_t(1.,0.,2) , Vec_t(0.,0.,1.)) == 0. );
    assert( placedpoly1-> DistanceToOut( Vec_t(0.5,0., -1) , Vec_t(0.,0.,-1.)) == 0. );
    assert( placedpoly1-> DistanceToOut( Vec_t(0.5,0., -1) , Vec_t(0.,0., 1.)) == 3. );

   
}


// Check name
    assert(MyPCone->GetName()=="MyPCone");
    assert(Simple.GetName()=="SimpleTube+Cone");

// Check Cubic volume
    double vol,volCheck;
    vol =  Simple.Capacity();
    volCheck = UUtils::kPi*(70*70*10+10*(70*70+80*80+70*80)/3.);
    assert(ApproxEqual(vol,volCheck));
   
 // Check Surface area
    vol = Simple.SurfaceArea();
    volCheck = UUtils::kPi*(70*70+80*80+(70+80)*std::sqrt(10*10+10*10)+10*2*70);
    assert(ApproxEqual(vol,volCheck));  
  

// Check Inside
  
    Vec_t pzero(0,0,0); 
    Vec_t ponxside(70,0,-5 ),ponyside(0,70,-5),ponzside(70,0,0);
    Vec_t ponmxside(-70,0,-3),ponmyside(0,-70,-10),ponmzside(70,0,-9);
    Vec_t ponzsidey(0,25,0),ponmzsidey(4,25,0);

    Vec_t pbigx(100,0,0),pbigy(0,100,0),pbigz(0,0,100);
    Vec_t pbigmx(-100,0,0),pbigmy(0,-100,0),pbigmz(0,0,-100);

    Vec_t vx(1,0,0),vy(0,1,0),vz(0,0,1);
    Vec_t vmx(-1,0,0),vmy(0,-1,0),vmz(0,0,-1);
    Vec_t vxy(1/std::sqrt(2.0),1/std::sqrt(2.0),0);
    Vec_t vmxy(-1/std::sqrt(2.0),1/std::sqrt(2.0),0);
    Vec_t vmxmy(-1/std::sqrt(2.0),-1/std::sqrt(2.0),0);
    Vec_t vxmy(1/std::sqrt(2.0),-1/std::sqrt(2.0),0);

    double Dist;
    Vec_t normal,norm;
    bool valid,convex;
    assert(Simple.Inside(pzero)==vecgeom::EInside::kInside);
    assert(Simple.Inside(pbigz)==vecgeom::EInside::kOutside);
    assert(Simple.Inside(pbigx)==vecgeom::EInside::kOutside);
    assert(Simple.Inside(pbigy)==vecgeom::EInside::kOutside);
    assert(Simple.Inside(ponxside)==vecgeom::EInside::kSurface);
    assert(Simple.Inside(ponyside)==vecgeom::EInside::kSurface);
    assert(Simple.Inside(ponzside)==vecgeom::EInside::kSurface);

    assert(Simple.Inside(ponmxside)==vecgeom::EInside::kSurface);
    assert(Simple.Inside(ponmyside)==vecgeom::EInside::kSurface);
    assert(Simple.Inside(ponmzside)==vecgeom::EInside::kSurface);
    assert(Simple.Inside(ponzsidey)==vecgeom::EInside::kInside);
    assert(Simple.Inside(ponmzsidey)==vecgeom::EInside::kInside);

// Check Surface Normal
 

    valid=Simple.Normal(ponxside,normal);
    assert(ApproxEqual(normal,Vec_t(1,0,0))&&valid);
    valid=Simple.Normal(ponmxside,normal);
    assert(ApproxEqual(normal,Vec_t(-1,0,0)));
    valid=Simple.Normal(ponyside,normal);
    assert(ApproxEqual(normal,Vec_t(0,1,0)));
    valid=Simple.Normal(Vec_t(0,0,10),normal);
    assert(ApproxEqual(normal,Vec_t(0,0,1)));
    valid=Simple.Normal(Vec_t(0,0,-10),normal);
    assert(ApproxEqual(normal,Vec_t(0,0,-1)));

    // Normals on Edges
   
    Vec_t edgeXZ(    80.0, 0.0, 10.0); 
    Vec_t edgemXmZ( -70.0, 0.0, -10.0); 
    /*    Vec_t edgeXmZ(   20.0, 0.0, -40.0); 
    Vec_t edgemXZ(  -20.0, 0.0, 40.0); 
    Vec_t edgeYZ(    0.0,  30.0,  40.0); 
    Vec_t edgemYmZ(  0.0, -30.0, -40.0); 
    Vec_t edgeYmZ(   0.0,  30.0, -40.0); 
    Vec_t edgemYZ(   0.0, -30.0,  40.0); 

    double invSqrt2 = 1.0 / std::sqrt( 2.0); 
    double invSqrt3 = 1.0 / std::sqrt( 3.0); 
    */
  
    // std::cout << " Normal at " << edgeXY << " is " << normal 
    //    << " Expected is " << Vec_t( invSqrt2, invSqrt2, 0.0) << std::ensl;     
    valid= Simple.Normal( edgeXZ ,normal); 
    //assert(ApproxEqual( normal, Vec_t(  invSqrt2, 0.0, invSqrt2) )); 
    valid= Simple.Normal( edgemXmZ ,normal);     
    //assert(ApproxEqual( normal, Vec_t( -invSqrt2, 0.0, -invSqrt2) )); 

// SafetyFromInside(P)

    Dist=Simple.SafetyFromInside(Vec_t(5,5,-5));
    //std::cout<<Dist<<std::endl;
    assert(ApproxEqual(Dist,5));
    Dist=Simple.SafetyFromInside(Vec_t(5,5,7));
    //std::cout<<Dist<<std::endl;
    assert(ApproxEqual(Dist,3));
    Dist=Simple.SafetyFromInside(Vec_t(69,0,-5));
    //std::cout<<Dist<<std::endl;
    assert(ApproxEqual(Dist,1));
    Dist=Simple.SafetyFromInside(Vec_t(-3,-3,8));
    //std::cout<<Dist<<std::endl;
    assert(ApproxEqual(Dist,2));

   
// DistanceToOut(P,V)

    Dist=Simple.DistanceToOut(pzero,vx,norm,convex);
    //std::cout<<Dist<<" "<<norm<<std::endl;
    assert(ApproxEqual(Dist,70) && ApproxEqual(norm,vx) && (!convex));
    Dist=Simple.DistanceToOut(pzero,vmx,norm,convex);
    //std::cout<<Dist<<" "<<norm<<std::endl;
    assert(ApproxEqual(Dist,70) && ApproxEqual(norm,vmx) && (!convex));
    //std::cout<<Dist<<std::endl;
    Dist=Simple.DistanceToOut(pzero,vy,norm,convex);
    //std::cout<<Dist<<std::endl;
    assert(ApproxEqual(Dist,70) && ApproxEqual(norm,vy) && (!convex));
    //std::cout<<Dist<<std::endl;
    Dist=Simple.DistanceToOut(pzero,vmy,norm,convex);
    //std::cout<<Dist<<std::endl;
    assert(ApproxEqual(Dist,70) && ApproxEqual(norm,vmy)&& (!convex));
    Dist=Simple.DistanceToOut(pzero,vz,norm,convex);
    // std::cout<<Dist<< " " <<norm<<std::endl; 
    assert(ApproxEqual(Dist,10)&&ApproxEqual(norm,vz));
   
    Dist=Simple.DistanceToOut(Vec_t(70,0,-10),vx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vx));
    Dist=Simple.DistanceToOut(Vec_t(-70,0,-1),vmx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmx));
    Dist=Simple.DistanceToOut(Vec_t(0,70,-10),vy,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vy));
    Dist=Simple.DistanceToOut(Vec_t(0,-70,-1),vmy,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmy));
   
//SafetyFromOutside(P)

    Dist=Simple.SafetyFromOutside(pbigx);
    //std::cout<<Dist<<std::endl;
    assert(ApproxEqual(Dist,20));
    Dist=Simple.SafetyFromOutside(pbigmx);
    //std::cout<<Dist<<std::endl; 
    assert(ApproxEqual(Dist,20));
    Dist=Simple.SafetyFromOutside(pbigy);
    //std::cout<<Dist<<std::endl; 
    assert(ApproxEqual(Dist,20));
    Dist=Simple.SafetyFromOutside(pbigmy);
    //std::cout<<Dist<<std::endl; 
    assert(ApproxEqual(Dist,20));
    Dist=Simple.SafetyFromOutside(pbigz);
    //std::cout<<Dist<<std::endl; 
    assert(ApproxEqual(Dist,80));
    Dist=Simple.SafetyFromOutside(pbigmz);
    //std::cout<<Dist<<std::endl;
    assert(ApproxEqual(Dist,80));

// DistanceToIn(P,V)

    Dist=Simple.DistanceToIn(Vec_t(100,0,-1),vmx);
    //std::cout<<Dist<<std::endl; 
    assert(ApproxEqual(Dist,30));
    Dist=Simple.DistanceToIn(Vec_t(-100,0,-1),vx);
    //std::cout<<Dist<<std::endl; 
    assert(ApproxEqual(Dist,30));
    Dist=Simple.DistanceToIn(Vec_t(0,100,-5),vmy);
    // std::cout<<Dist<<std::endl; 
    assert(ApproxEqual(Dist,30));
    Dist=Simple.DistanceToIn(Vec_t(0,-100,-5),vy);
    //std::cout<<Dist<<std::endl; 
    assert(ApproxEqual(Dist,30));
    Dist=Simple.DistanceToIn(pbigz,vmz);
    //std::cout<<Dist<<std::endl; 
    assert(ApproxEqual(Dist,90));
    Dist=Simple.DistanceToIn(pbigmz,vz);
    //std::cout<<Dist<<std::endl;
    assert(ApproxEqual(Dist,90));
    Dist=Simple.DistanceToIn(pbigx,vxy);
    //std::cout<<Dist<<std::endl; 
    assert(ApproxEqual(Dist,UUtils::kInfinity));
    Dist=Simple.DistanceToIn(pbigmx,vmxy);
    //std::cout<<Dist<<std::endl; 
    assert(ApproxEqual(Dist,UUtils::kInfinity));

   
  // CalculateExtent
    Vec_t minExtent,maxExtent;
    Simple.Extent(minExtent,maxExtent);
    //std::cout<<" min="<<minExtent<<" max="<<maxExtent<<std::endl;
    assert(ApproxEqual(minExtent,Vec_t(-80,-80,-10)));
    assert(ApproxEqual(maxExtent,Vec_t( 80, 80, 10)));
    MyPCone->Extent(minExtent,maxExtent);
    //std::cout<<" min="<<minExtent<<" max="<<maxExtent<<std::endl;
    assert(ApproxEqual(minExtent,Vec_t(-80,-80,-20)));
    assert(ApproxEqual(maxExtent,Vec_t( 80, 80, 40)));
  
   

#ifdef SCAN_SOLID
  
  std::cout << "\n=======     Polycone SCAN test      ========";
  std::cout << "\n\nPCone created ! "<<std::endl;
  // -> Check methods :
  //  - Inside
  //  - DistanceToIn
  //  - DistanceToOut

  
  VUSolid::EnumInside in;
  
  std::cout<<"\n\n==================================================";
  Vec_t pt(0, -100, 24);
  int y;
  for (y = -100; y<=100; y+=10)
  {
    //pt.setY(y);
    pt.Set(0,y,24);
    in = MyPCone->Inside(pt);
    
    std::cout << "\nx=" << pt.x() << "  y=" << pt.y() << "  z=" << pt.z();
    
    if( in == vecgeom::EInside::kInside )
      std::cout <<" is inside";
    else
      if( in == vecgeom::EInside::kOutside )
	std::cout <<" is outside";
      else
	std::cout <<" is on the surface";
  }

  std::cout<<"\n\n==================================================";
  Vec_t start( 0, 0, -30);
  Vec_t dir(1./std::sqrt(2.), 1./std::sqrt(2.), 0),normal;
  double   d;
  int z;
  bool convex;
  
  std::cout<<"\nPdep is (0, 0, z)";
  std::cout<<"\nDir is (1, 1, 0)\n";

  for(z=-30; z<=50; z+=5)
  {
    //start.setZ(z);
    start.Set(0,0,z);

    in = MyPCone->Inside(start);
    std::cout<< "x=" << start.x() << "  y=" << start.y() << "  z=" << start.z();
    
    if( in == vecgeom::EInside::kInside )
    {
      std::cout <<" is inside";

      d = MyPCone->DistanceToOut(start, dir,normal,convex);
      std::cout<<"  distance to out="<<d;
      d = MyPCone->SafetyFromInside(start);
      std::cout<<"  closest distance to out="<<d<<std::endl;
    }
    else if( in == vecgeom::EInside::kOutside ) 
    {
      std::cout <<" is outside";

      d = MyPCone->DistanceToIn(start, dir);
      std::cout<<"  distance to in="<<d;
      d = MyPCone->SafetyFromOutside(start);
      std::cout<<"  closest distance to in="<<d<<std::endl;
    }
    else
      std::cout <<" is on the surface"<<std::endl;

  }

  std::cout<<"\n\n==================================================";
  Vec_t start2( 0, -100, -30);
  Vec_t dir2(0, 1, 0);
  double   d2;

  std::cout<<"\nPdep is (0, -100, z)";
  std::cout<<"\nDir is (0, 1, 0)\n";

  for(z=-30; z<=50; z+=5)
  {
    std::cout<<"  z="<<z;
    //start2.setZ(z);
    start2.Set(0,-100,z);
    d2 = MyPCone->DistanceToIn(start2, dir2);
    std::cout<<"  distance to in="<<d2;
    d2 = MyPCone->SafetyFromOutside(start2);
    std::cout<<"  distance to in="<<d2<<std::endl;
  }

  std::cout<<"\n\n==================================================";
  Vec_t start3( 0, 0, -50);
  Vec_t dir3(0, 0, 1);
  double   d3;

  std::cout<<"\nPdep is (0, y, -50)";
  std::cout<<"\nDir is (0, 0, 1)\n";

  for(y=-0; y<=90; y+=5)
  {
    std::cout<<"  y="<<y;
    //start3.setY(y);
    start3.Set(0,y,-50);
    d3 = MyPCone->DistanceToIn(start3, dir3);
    std::cout<<"  distance to in="<<d3<<std::endl;
  }
  //
  // Add checks in Phi direction
  // Point move in Phi direction for differents Z
  //
   std::cout<<"\n\n==================================================";
   Vec_t start4; 
 for(z=-10; z<=50; z+=5)
   {
     std::cout<<"\n\n===================Z="<<z<<"==============================";
     //Vec_t start4( 0, 0, z-0.00001);
     //Vec_t start4( 0, 0, z);
     start4.Set(0,0,z);
  //G4double phi=pi/180.*rad;
  //  G4double phi=0.0000000001*pi/180.*rad;
   double phi=-UUtils::kPi/180.*UUtils::kPi/180.;
  Vec_t dir4(std::cos(phi), std::sin(phi), 0);
  double   d4;

  std::cout<<"\nPdep is (0<<R<<50, phi, z)";
  std::cout<<"\nDir is (std::cos(phi), std::sin(phi), 0)\n";
  std::cout<<"Ndirection is="<<dir4 <<std::endl;

  for(y=-0; y<=50; y+=5)
  {
    
    //start4.setX(y*std::cos(phi));
    //start4.setY(y*std::sin(phi));
    start4.Set(y*std::cos(phi),y*std::sin(phi),z);
    std::cout<<"  R="<<y<<" with Start"<<start4;
    in = MyPCone->Inside(start4);
    if( in == vecgeom::EInside::kInside )
      {
       std::cout <<" is inside";
       d4 = MyPCone->DistanceToOut(start4, dir4,normal,convex);
         std::cout<<"  distance to out="<<d4;
         d4 = MyPCone->SafetyFromInside(start4);
         std::cout<<" closest distance to out="<<d4<<std::endl;
	}
    else
      if( in == vecgeom::EInside::kOutside )
	{
         std::cout <<" is outside";
          d4 = MyPCone->DistanceToIn(start4, dir4);
         std::cout<<"  distance to in="<<d4;
         d4 = MyPCone->SafetyFromOutside(start4);
         std::cout<<" closest distance to in="<<d4<<std::endl;
	}
      else
	{std::cout <<" is on the surface";
         d4 = MyPCone->DistanceToIn(start4, dir4);
         std::cout<<"  distance to in="<<d4;
         d4 = MyPCone->SafetyFromOutside(start4);
         std::cout<<" closest distance to in="<<d4<<std::endl;
	}
    
  }
   }
 //
 // Add checks in Phi direction
 // Point move in X direction for differents Z
 // and 'schoot' on rhi edge
   std::cout<<"\n\n==================================================";
   Vec_t start5;
 for(z=-10; z<=50; z+=5)
   {
     std::cout<<"\n\n===================Z="<<z<<"==============================";
     // Vec_t start5( 0., 0.000000000001, z);
     // Vec_t start5( 0., 1, z);
     start5.Set(0,1,z);
  Vec_t dir5(0,-1, 0);
  double   d5;

  std::cout<<"\nPdep is (0<<X<<50, 1, z)";
  std::cout<<"\nDir is (0, -1, 0)\n";
  std::cout<<"Ndirection is="<<dir5 <<std::endl;

  for(y=-0; y<=50; y+=5)
  {
    
    //start5.setX(y);
    start5.Set(0,y,z);
    std::cout<<" Start"<<start5;
    in = MyPCone->Inside(start5);
    if( in ==  vecgeom::EInside::kInside )
      {
       std::cout <<" is inside";
       d5 = MyPCone->DistanceToOut(start5, dir5,normal,convex);
       std::cout<<"  distance to out="<<d5;
       d5 = MyPCone->SafetyFromInside(start5);
       std::cout<<" closest distance to out="<<d5<<std::endl;
      }
    else
      if( in ==  vecgeom::EInside::kOutside )
        {
	 std::cout <<" is outside";
         d5 = MyPCone->DistanceToIn(start5, dir5);
         std::cout<<"  distance to in="<<d5;
         d5 = MyPCone->SafetyFromOutside(start5);
         std::cout<<" closest distance to in="<<d5<<std::endl;
        }
      else
        {
	 std::cout <<" is on the surface";
         d5 = MyPCone->DistanceToIn(start5, dir5);
         std::cout<<"  distance to in="<<d5;
         d5 = MyPCone->SafetyFromOutside(start5);
         std::cout<<" closest distance to in="<<d5<<std::endl;
        }
    
  }
   }

 #endif



    return true;
}

/*

int main() {
#ifdef VECGEOM_USOLIDS
  assert(TestPolycone<UPolycone>());
  //assert(TestPolycone<UGenericPolycone>());
  std::cout << "UPolycone passed\n";

#endif
  std::cout<< "VecGeom Polycone not included yet\n";
  
  return 0;
}
*/

//bool testingvecgeom=false;
int main(int argc, char *argv[]) {
 
   if( argc < 2)
    {
      std::cerr << "need to give argument :--usolids or --vecgeom\n";     
      return 1;
    }
    
    if( ! strcmp(argv[1], "--usolids") )
    { 
      #ifdef VECGEOM_USOLIDS
      assert(TestPolycone<UPolycone>());
      std::cout << "UPolycone passed\n";
      #else
      std::cerr << "VECGEOM_USOLIDS was not defined\n";
      return 2;
      #endif
    }
    else if( ! strcmp(argv[1], "--vecgeom") )
    {
        testingvecgeom = true;
        assert(TestPolycone<vecgeom::SimplePolycone>());
     std::cout << "VecGeomPolycone passed\n";
    }
    else
    {
      std::cerr << "need to give argument :--usolids or --vecgeom\n";     
      return 1;
    }


  return 0;
}
