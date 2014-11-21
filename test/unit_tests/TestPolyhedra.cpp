//
//
// TestTrd
//             Ensure asserts are compiled in

#undef NDEBUG

#include "base/Vector3D.h"
#include "volumes/Trd.h"
#include "ApproxEqual.h"
#ifdef VECGEOM_USOLIDS
#include "UPolyhedra.hh"
#include "UVector3.hh"
#endif

//#include <cassert>
#include <cmath>

template <class Polyhedra_t,class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >

bool TestPolyhedra()
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
  Z_Values[0] =-30;
  Z_Values[1] =-20;
  Z_Values[2] =-10;
  Z_Values[3] =  0;
  Z_Values[4] = 10;
  Z_Values[5] = 20;
  Z_Values[6] = 30;
  Z_Values[7] = 40;


  double Phi_Values[2];
  Phi_Values[0]=0.;
  Phi_Values[1]=45.*UUtils::kPi/180.;
  //Phi_Values[1]=2*UUtils::kPi;
  
  Polyhedra_t *MyPGon = new Polyhedra_t ("MyPGon",
                                                    Phi_Values[0],
						    Phi_Values[1],
						    5        ,
						    8        ,
						    Z_Values ,
						    RMINVec  ,
						    RMAXVec   );

// Check name
    assert(MyPGon->GetName()=="MyPGon");
    
// Check Cubic volume
    double vol;
    vol = MyPGon->Capacity();
    //std::cout.precision(20);
    //std::cout<<MyPGon->Capacity()<<std::endl;
    assert(ApproxEqual(vol,155138.6874225));

// Check Surface area
    vol=MyPGon->SurfaceArea();
    assert(ApproxEqual(vol,1284298.5697));    
    //std::cout<<MyPGon->SurfaceArea()<<std::endl;

// Asserts
 Vec_t p1,p2,p3,p4,p5,p6,dirx,diry,dirz;
 p1=Vec_t(0,0,-5); 
 p2=Vec_t(50,0,40);
 p3=Vec_t(5,1,20 ); 
 p4=Vec_t(45,5,30);
 p5=Vec_t(0,0,30); 
 p6=Vec_t(41,0,10);

 dirx=Vec_t(1,0,0);
 diry=Vec_t(0,1,0);
 dirz=Vec_t(0,0,1);

 //Check Inside
  assert(MyPGon->Inside(p1) ==  vecgeom::EInside::kSurface);
  assert(MyPGon->Inside(p2) ==  vecgeom::EInside::kSurface);
  assert(MyPGon->Inside(p3) ==  vecgeom::EInside::kInside);
  assert(MyPGon->Inside(p4) ==  vecgeom::EInside::kInside);
  assert(MyPGon->Inside(p5) ==  vecgeom::EInside::kOutside);
  //assert(MyPGon->Inside(p6) ==  vecgeom::EInside::kOutside);
 //Check DistanceToIn
 
  double tolerance = 1e-9;
  assert(std::fabs((MyPGon->DistanceToIn(p1,dirx))) < tolerance);
  assert(std::fabs((MyPGon->DistanceToIn(p1,-diry)))< tolerance);
  assert(std::fabs((MyPGon->DistanceToIn(p2,diry))) < tolerance);
  assert(std::fabs((MyPGon->DistanceToIn(p5,dirx)  -40.12368793931)) < tolerance);
  assert(std::fabs((MyPGon->DistanceToIn(p6,-dirx)  -0.87631206069)) < tolerance);
  assert(std::fabs((MyPGon->DistanceToIn(p6,dirz)   -0.218402670765))< tolerance);
 //Check DistanceToOut
  Vec_t normal;
  bool convex;
  assert(std::fabs((MyPGon->DistanceToOut (p1,-dirx,normal,convex)))               < tolerance);
  assert(std::fabs((MyPGon->DistanceToOut (p3,-diry,normal,convex) -1.) )          < tolerance);
  assert(std::fabs((MyPGon->DistanceToOut (p3,dirz,normal,convex)  -1.27382374146))< tolerance);
  assert(std::fabs((MyPGon->DistanceToOut(p4,dirz,normal,convex)  -10.))           < tolerance);
  assert(std::fabs((MyPGon->DistanceToOut(p4,dirx,normal,convex)  -34.8538673445)) < tolerance);
  assert(std::fabs((MyPGon->DistanceToOut(p4,diry,normal,convex)  -40.))           < tolerance);

// CalculateExtent
    Vec_t minExtent,maxExtent;
    MyPGon->Extent(minExtent,maxExtent);
    // std::cout<<" min="<<minExtent<<" max="<<maxExtent<<std::endl;
    assert(ApproxEqual(minExtent,Vec_t(-80.247375,-80.247375,-30)));
    assert(ApproxEqual(maxExtent,Vec_t( 80.247375, 80.247375, 40)));


 #ifdef SCAN_SOLID 
  std::cout << "\n=======     Polyhedra SCAN test      ========";
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
    in = MyPGon->Inside(pt);
    
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

    in = MyPGon->Inside(start);
    std::cout<< "x=" << start.x() << "  y=" << start.y() << "  z=" << start.z();
    
    if( in == vecgeom::EInside::kInside )
    {
      std::cout <<" is inside";

      d = MyPGon->DistanceToOut(start, dir,normal,convex);
      std::cout<<"  distance to out="<<d;
      d = MyPGon->SafetyFromInside(start);
      std::cout<<"  closest distance to out="<<d<<std::endl;
    }
    else if( in == vecgeom::EInside::kOutside ) 
    {
      std::cout <<" is outside";

      d = MyPGon->DistanceToIn(start, dir);
      std::cout<<"  distance to in="<<d;
      d = MyPGon->SafetyFromOutside(start);
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
    d2 = MyPGon->DistanceToIn(start2, dir2);
    std::cout<<"  distance to in="<<d2;
    d2 = MyPGon->SafetyFromOutside(start2);
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
    d3 = MyPGon->DistanceToIn(start3, dir3);
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
    in = MyPGon->Inside(start4);
    if( in == vecgeom::EInside::kInside )
      {
       std::cout <<" is inside";
       d4 = MyPGon->DistanceToOut(start4, dir4,normal,convex);
         std::cout<<"  distance to out="<<d4;
         d4 = MyPGon->SafetyFromInside(start4);
         std::cout<<" closest distance to out="<<d4<<std::endl;
	}
    else
      if( in == vecgeom::EInside::kOutside )
	{
         std::cout <<" is outside";
          d4 = MyPGon->DistanceToIn(start4, dir4);
         std::cout<<"  distance to in="<<d4;
         d4 = MyPGon->SafetyFromOutside(start4);
         std::cout<<" closest distance to in="<<d4<<std::endl;
	}
      else
	{std::cout <<" is on the surface";
         d4 = MyPGon->DistanceToIn(start4, dir4);
         std::cout<<"  distance to in="<<d4;
         d4 = MyPGon->SafetyFromOutside(start4);
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
    in = MyPGon->Inside(start5);
    if( in ==  vecgeom::EInside::kInside )
      {
       std::cout <<" is inside";
       d5 = MyPGon->DistanceToOut(start5, dir5,normal,convex);
       std::cout<<"  distance to out="<<d5;
       d5 = MyPGon->SafetyFromInside(start5);
       std::cout<<" closest distance to out="<<d5<<std::endl;
      }
    else
      if( in ==  vecgeom::EInside::kOutside )
        {
	 std::cout <<" is outside";
         d5 = MyPGon->DistanceToIn(start5, dir5);
         std::cout<<"  distance to in="<<d5;
         d5 = MyPGon->SafetyFromOutside(start5);
         std::cout<<" closest distance to in="<<d5<<std::endl;
        }
      else
        {
	 std::cout <<" is on the surface";
         d5 = MyPGon->DistanceToIn(start5, dir5);
         std::cout<<"  distance to in="<<d5;
         d5 = MyPGon->SafetyFromOutside(start5);
         std::cout<<" closest distance to in="<<d5<<std::endl;
        }
    
  }
   }
#endif

    
    return true;
}


int main() {
#ifdef VECGEOM_USOLIDS
  assert(TestPolyhedra<UPolyhedra>());
  std::cout << "UPolyhedra passed\n";

#endif
  std::cout<< "VecGeom Polyhedra not included yet\n";
  
  return 0;
}
