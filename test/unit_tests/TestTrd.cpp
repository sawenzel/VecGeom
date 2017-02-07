//
//
// TestTrd
//             Ensure asserts are compiled in



#include "base/Vector3D.h"
#include "volumes/Trd.h"
#include "ApproxEqual.h"
#ifdef VECGEOM_USOLIDS
#include "UTrd.hh"
#include "UVector3.hh"
#endif
#include <cmath>

#undef NDEBUG
#include <cassert>

template <class Trd_t,class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >

bool TestTrd()
{

 VUSolid::EnumInside inside ;
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

    double Dist, vol, volCheck;
    Vec_t normal,norm;
    bool valid,convex;


    Trd_t trd1("Test Box #1",20,20,30,30,40);
    Trd_t trd2("Test Trd",10,30,20,40,40);
    Trd_t trd3("BABAR Trd",0.14999999999999999,0.14999999999999999, 
                           24.707000000000001, 24.707000000000001, 
	                   22.699999999999999) ;


// Check name
    assert(trd1.GetName()=="Test Box #1");
    assert(trd2.GetName()=="Test Trd");

// check Cubic volume

    vol = trd1.Capacity();
    volCheck = 8*20*30*40;
    assert(ApproxEqual(vol,volCheck));

 // Check Surface area    
    std::cout<<"Trd Surface Area : " << trd1.SurfaceArea()<<std::endl;
    //assert(trd1.SurfaceArea() == 20800);

// Check Inside

    assert(trd1.Inside(pzero)==vecgeom::EInside::kInside);
    assert(trd1.Inside(pbigz)==vecgeom::EInside::kOutside);
    assert(trd1.Inside(ponxside)==vecgeom::EInside::kSurface);
    assert(trd1.Inside(ponyside)==vecgeom::EInside::kSurface);
    assert(trd1.Inside(ponzside)==vecgeom::EInside::kSurface);

    inside = trd1.Inside(Vec_t(20,30,40)) ;
    //  std::cout << "trd1.Inside((20,30,40)) = " << OutputInside(inside) << std::ensl ;
    assert(inside == vecgeom::EInside::kSurface);

    inside = trd1.Inside(Vec_t(-20,30,40)) ;
    // std::cout << "trd1.Inside((-20,30,40)) = " << OutputInside(inside) << std::ensl ;
    assert(inside == vecgeom::EInside::kSurface);

    inside = trd1.Inside(Vec_t(20,-30,40)) ;
    //  std::cout << "trd1.Inside((20,-30,40)) = " << OutputInside(inside) << std::ensl ;
    assert(inside == vecgeom::EInside::kSurface);

    inside = trd1.Inside(Vec_t(20,30,-40)) ;
    // std::cout << "trd1.Inside((20,30,-40)) = " << OutputInside(inside) << std::ensl ;
    assert(inside == vecgeom::EInside::kSurface);

    inside = trd1.Inside(Vec_t(20,30,0)) ;
    // std::cout << "trd1.Inside((20,30,0)) = " << OutputInside(inside) << std::ensl ;
    assert(inside == vecgeom::EInside::kSurface);

    inside = trd1.Inside(Vec_t(0,30,40)) ;
    // std::cout << "trd1.Inside((0,30,40)) = " << OutputInside(inside) << std::ensl ;
    assert(inside == vecgeom::EInside::kSurface);

    inside = trd1.Inside(Vec_t(20,0,40)) ;
    // std::cout << "trd1.Inside((20,0,40)) = " << OutputInside(inside) << std::ensl ;
    assert(inside == vecgeom::EInside::kSurface);

    inside = trd1.Inside(Vec_t(-20,-30,-40)) ;
    // std::cout << "trd1.Inside((-20,-30,-40)) = " << OutputInside(inside) << std::ensl ;
    assert(inside == vecgeom::EInside::kSurface);

    assert(trd2.Inside(pzero)==vecgeom::EInside::kInside);
    assert(trd2.Inside(pbigz)==vecgeom::EInside::kOutside);
    assert(trd2.Inside(ponxside)==vecgeom::EInside::kSurface);
    assert(trd2.Inside(ponyside)==vecgeom::EInside::kSurface);
    assert(trd2.Inside(ponzside)==vecgeom::EInside::kSurface);

// Check Surface Normal
 

    valid=trd1.Normal(ponxside,normal);
    assert(ApproxEqual(normal,Vec_t(1,0,0)));
    valid=trd1.Normal(ponmxside,normal);
    assert(ApproxEqual(normal,Vec_t(-1,0,0)));
    valid=trd1.Normal(ponyside,normal);
    assert(ApproxEqual(normal,Vec_t(0,1,0)));
    valid=trd1.Normal(ponmyside,normal);
    assert(ApproxEqual(normal,Vec_t(0,-1,0)));
    valid=trd1.Normal(ponzside,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,1)));
    valid=trd1.Normal(ponmzside,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,-1)));
    valid=trd1.Normal(ponzsidey,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,1)));
    valid=trd1.Normal(ponmzsidey,normal);
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

    valid= trd1.Normal( edgeXY ,normal); 
    assert(ApproxEqual( normal, Vec_t( invSqrt2, invSqrt2, 0.0) )&&valid); 

    // std::cout << " Normal at " << edgeXY << " is " << normal 
    //    << " Expected is " << Vec_t( invSqrt2, invSqrt2, 0.0) << std::ensl;     

    valid= trd1.Normal( edgemXmY,normal ); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, -invSqrt2, 0.0) )); 
    valid= trd1.Normal( edgeXmY ,normal); 
    assert(ApproxEqual( normal, Vec_t( invSqrt2, -invSqrt2, 0.0) )); 
    valid= trd1.Normal( edgemXY ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, invSqrt2, 0.0) )); 

    valid= trd1.Normal( edgeXZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt2, 0.0, invSqrt2) )); 
    valid= trd1.Normal( edgemXmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, 0.0, -invSqrt2) )); 
    valid= trd1.Normal( edgeXmZ,normal ); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt2, 0.0, -invSqrt2) )); 
    valid= trd1.Normal( edgemXZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt2, 0.0, invSqrt2) )); 

    valid= trd1.Normal( edgeYZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( 0.0,  invSqrt2,  invSqrt2) )); 
    valid= trd1.Normal( edgemYmZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( 0.0, -invSqrt2, -invSqrt2) )); 
    valid= trd1.Normal( edgeYmZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( 0.0,  invSqrt2, -invSqrt2) )); 
    valid= trd1.Normal( edgemYZ,normal ); 
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
 
    valid= trd1.Normal( cornerXYZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3,  invSqrt3, invSqrt3) )); 
    valid= trd1.Normal( cornermXYZ,normal ); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3,  invSqrt3, invSqrt3) )); 
    valid= trd1.Normal( cornerXmYZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3, -invSqrt3, invSqrt3) )); 
    valid= trd1.Normal( cornermXmYZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3, -invSqrt3, invSqrt3) )); 
    valid= trd1.Normal( cornerXYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3,  invSqrt3, -invSqrt3) )); 
    valid= trd1.Normal( cornermXYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3,  invSqrt3, -invSqrt3) )); 
    valid= trd1.Normal( cornerXmYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t(  invSqrt3, -invSqrt3, -invSqrt3) )); 
    valid= trd1.Normal( cornermXmYmZ ,normal); 
    assert(ApproxEqual( normal, Vec_t( -invSqrt3, -invSqrt3, -invSqrt3) )); 


   double cosa = 4/std::sqrt(17.), sina = 1/std::sqrt(17.);
    
    valid=trd2.Normal(ponxside,normal);
    assert(ApproxEqual(normal,Vec_t(cosa,0,-sina)));
    valid=trd2.Normal(ponmxside,normal);
    assert(ApproxEqual(normal,Vec_t(-cosa,0,-sina)));
    valid=trd2.Normal(ponyside,normal);
    assert(ApproxEqual(normal,Vec_t(0,cosa,-sina)));
    valid=trd2.Normal(ponmyside,normal);
    assert(ApproxEqual(normal,Vec_t(0,-cosa,-sina)));
    valid=trd2.Normal(ponzside,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,1)));
    valid=trd2.Normal(ponmzside,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,-1)));
    valid=trd2.Normal(ponzsidey,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,1)));
    valid=trd2.Normal(ponmzsidey,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,-1))); // (0,cosa,-sina) ?

// SafetyFromInside(P)

    Dist=trd1.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,20));
    Dist=trd1.SafetyFromInside(vx);
    assert(ApproxEqual(Dist,19));
    Dist=trd1.SafetyFromInside(vy);
    assert(ApproxEqual(Dist,20));
    Dist=trd1.SafetyFromInside(vz);
    assert(ApproxEqual(Dist,20));

    Dist=trd2.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,20*cosa));
    Dist=trd2.SafetyFromInside(vx);
    assert(ApproxEqual(Dist,19*cosa));
    Dist=trd2.SafetyFromInside(vy);
    assert(ApproxEqual(Dist,20*cosa));
    Dist=trd2.SafetyFromInside(vz);
    assert(ApproxEqual(Dist,20*cosa+sina));


// DistanceToOut(P,V)

    Dist=trd1.DistanceToOut(pzero,vx,norm,convex);
    assert(ApproxEqual(Dist,20)&&ApproxEqual(norm,vx)&&convex);
    Dist=trd1.DistanceToOut(pzero,vmx,norm,convex);
    assert(ApproxEqual(Dist,20)&&ApproxEqual(norm,vmx)&&convex);
    Dist=trd1.DistanceToOut(pzero,vy,norm,convex);
    assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,vy)&&convex);
    Dist=trd1.DistanceToOut(pzero,vmy,norm,convex);
    assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,vmy)&&convex);
    Dist=trd1.DistanceToOut(pzero,vz,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vz)&&convex);
    Dist=trd1.DistanceToOut(pzero,vmz,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vmz)&&convex);
    Dist=trd1.DistanceToOut(pzero,vxy,norm,convex);
    assert(ApproxEqual(Dist,std::sqrt(800.))&&convex);

    Dist=trd1.DistanceToOut(ponxside,vx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vx)&&convex);
    Dist=trd1.DistanceToOut(ponmxside,vmx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmx)&&convex);
    Dist=trd1.DistanceToOut(ponyside,vy,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vy)&&convex);
    Dist=trd1.DistanceToOut(ponmyside,vmy,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmy)&&convex);
    Dist=trd1.DistanceToOut(ponzside,vz,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vz)&&convex);
    Dist=trd1.DistanceToOut(ponmzside,vmz,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmz)&&convex);

    Dist=trd2.DistanceToOut(pzero,vx,norm,convex);
    assert(ApproxEqual(Dist,20)&&ApproxEqual(norm,Vec_t(cosa,0,-sina))&&convex);
    Dist=trd2.DistanceToOut(pzero,vmx,norm,convex);
    assert(ApproxEqual(Dist,20)&&ApproxEqual(norm,Vec_t(-cosa,0,-sina))&&convex);
    Dist=trd2.DistanceToOut(pzero,vy,norm,convex);
    assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,Vec_t(0,cosa,-sina))&&convex);
    Dist=trd2.DistanceToOut(pzero,vmy,norm,convex);
    assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,Vec_t(0,-cosa,-sina))&&convex);
    Dist=trd2.DistanceToOut(pzero,vz,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vz)&&convex);
    Dist=trd2.DistanceToOut(pzero,vmz,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vmz)&&convex);
    Dist=trd2.DistanceToOut(pzero,vxy,norm,convex);
    assert(ApproxEqual(Dist,std::sqrt(800.))&&convex);

    Dist=trd2.DistanceToOut(ponxside,vx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,Vec_t(cosa,0,-sina))&&convex);
    Dist=trd2.DistanceToOut(ponmxside,vmx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,Vec_t(-cosa,0,-sina))&&convex);
    Dist=trd2.DistanceToOut(ponyside,vy,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,Vec_t(0,cosa,-sina))&&convex);
    Dist=trd2.DistanceToOut(ponmyside,vmy,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,Vec_t(0,-cosa,-sina))&&convex);
    Dist=trd2.DistanceToOut(ponzside,vz,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vz)&&convex);
    Dist=trd2.DistanceToOut(ponmzside,vmz,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmz)&&convex);


//SafetyFromOutside(P)

    Dist=trd1.SafetyFromOutside(pbigx);
    assert(ApproxEqual(Dist,80));
    Dist=trd1.SafetyFromOutside(pbigmx);
    assert(ApproxEqual(Dist,80));
    Dist=trd1.SafetyFromOutside(pbigy);
    assert(ApproxEqual(Dist,70));
    Dist=trd1.SafetyFromOutside(pbigmy);
    assert(ApproxEqual(Dist,70));
    Dist=trd1.SafetyFromOutside(pbigz);
    assert(ApproxEqual(Dist,60));
    Dist=trd1.SafetyFromOutside(pbigmz);
    assert(ApproxEqual(Dist,60));

    Dist=trd2.SafetyFromOutside(pbigx);
    assert(ApproxEqual(Dist,80*cosa));
    Dist=trd2.SafetyFromOutside(pbigmx);
    assert(ApproxEqual(Dist,80*cosa));
    Dist=trd2.SafetyFromOutside(pbigy);
    assert(ApproxEqual(Dist,70*cosa));
    Dist=trd2.SafetyFromOutside(pbigmy);
    assert(ApproxEqual(Dist,70*cosa));
    Dist=trd2.SafetyFromOutside(pbigz);
    assert(ApproxEqual(Dist,60));
    Dist=trd2.SafetyFromOutside(pbigmz);
    assert(ApproxEqual(Dist,60));


// DistanceToIn(P,V)

    Dist=trd1.DistanceToIn(pbigx,vmx);
    assert(ApproxEqual(Dist,80));
    Dist=trd1.DistanceToIn(pbigmx,vx);
    assert(ApproxEqual(Dist,80));
    Dist=trd1.DistanceToIn(pbigy,vmy);
    assert(ApproxEqual(Dist,70));
    Dist=trd1.DistanceToIn(pbigmy,vy);
    assert(ApproxEqual(Dist,70));
    Dist=trd1.DistanceToIn(pbigz,vmz);
    assert(ApproxEqual(Dist,60));
    Dist=trd1.DistanceToIn(pbigmz,vz);
    assert(ApproxEqual(Dist,60));
    Dist=trd1.DistanceToIn(pbigx,vxy);
    assert(ApproxEqual(Dist,UUtils::kInfinity));
    Dist=trd1.DistanceToIn(pbigmx,vxy);
    assert(ApproxEqual(Dist,UUtils::kInfinity));

    Dist=trd2.DistanceToIn(pbigx,vmx);
    assert(ApproxEqual(Dist,80));
    Dist=trd2.DistanceToIn(pbigmx,vx);
    assert(ApproxEqual(Dist,80));
    Dist=trd2.DistanceToIn(pbigy,vmy);
    assert(ApproxEqual(Dist,70));
    Dist=trd2.DistanceToIn(pbigmy,vy);
    assert(ApproxEqual(Dist,70));
    Dist=trd2.DistanceToIn(pbigz,vmz);
    assert(ApproxEqual(Dist,60));
    Dist=trd2.DistanceToIn(pbigmz,vz);
    assert(ApproxEqual(Dist,60));
    Dist=trd2.DistanceToIn(pbigx,vxy);
    assert(ApproxEqual(Dist,UUtils::kInfinity));
    Dist=trd2.DistanceToIn(pbigmx,vxy);
    assert(ApproxEqual(Dist,UUtils::kInfinity));

    Dist=trd3.DistanceToIn(Vec_t(  0.15000000000000185,
                                         -22.048743592955137,
                                           2.4268539333219472),
                           Vec_t(-0.76165597579890043,
                                          0.64364445891356026,
                                         -0.074515708658524193)) ;

    //    std::cout<<"BABAR trd distance = "<<Dist<<std::ensl ;
    assert(ApproxEqual(Dist,0.0));

   // return-value = 2.4415531753644804e-15




  // CalculateExtent
    Vec_t minExtent,maxExtent;
    trd1.Extent(minExtent,maxExtent);
    //std::cout<<" min="<<minExtent<<" max="<<maxExtent<<std::endl;
    assert(ApproxEqual(minExtent,Vec_t(-20,-30,-40)));
    assert(ApproxEqual(maxExtent,Vec_t( 20, 30, 40)));
    trd2.Extent(minExtent,maxExtent);
    //std::cout<<" min="<<minExtent<<" max="<<maxExtent<<std::endl;
     assert(ApproxEqual(minExtent,Vec_t(-30,-40,-40)));
     assert(ApproxEqual(maxExtent,Vec_t( 30, 40, 40)));
  

   
    return true;
}


int main(int argc, char *argv[]) {
 
   if( argc < 2)
    {
      std::cerr << "need to give argument :--usolids or --vecgeom\n";     
      return 1;
    }
    
    if( ! strcmp(argv[1], "--usolids") )
    { 
      #ifdef VECGEOM_USOLIDS
      assert(TestTrd<UTrd>());
      std::cout << "UTrd passed\n";
      #else
      std::cerr << "VECGEOM_USOLIDS was not defined\n";
      return 2;
      #endif
    }
    else if( ! strcmp(argv[1], "--vecgeom") )
    {
      //testingvecgeom = true;
        assert(TestTrd<vecgeom::SimpleTrd>());
     std::cout << "VecGeomTrd passed\n";
    }
    else
    {
      std::cerr << "need to give argument :--usolids or --vecgeom\n";     
      return 1;
    }


  return 0;
}
