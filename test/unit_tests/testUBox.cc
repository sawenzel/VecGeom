//
//
// testUBox
//             Ensure asserts are compiled in

#undef NDEBUG
#include <assert.h>
#include <cmath>

#include "ApproxEqual.hh"

#include "UVector3.hh"
#include "UBox.hh"
#include "VUSolid.hh"

bool testUBox()
{
    UVector3 pzero(0,0,0);
    UVector3 ponxside(20,0,0),ponyside(0,30,0),ponzside(0,0,40);
    UVector3 ponmxside(-20,0,0),ponmyside(0,-30,0),ponmzside(0,0,-40);
    UVector3 ponzsidey(0,25,40),ponmzsidey(0,25,-40);

    UVector3 pbigx(100,0,0),pbigy(0,100,0),pbigz(0,0,100);
    UVector3 pbigmx(-100,0,0),pbigmy(0,-100,0),pbigmz(0,0,-100);

    UVector3 vx(1,0,0),vy(0,1,0),vz(0,0,1);
    UVector3 vmx(-1,0,0),vmy(0,-1,0),vmz(0,0,-1);
    UVector3 vxy(1/std::sqrt(2.0),1/std::sqrt(2.0),0);
    UVector3 vmxy(-1/std::sqrt(2.0),1/std::sqrt(2.0),0);
    UVector3 vmxmy(-1/std::sqrt(2.0),-1/std::sqrt(2.0),0);
    UVector3 vxmy(1/std::sqrt(2.0),-1/std::sqrt(2.0),0);
    UVector3 vxmz(1/std::sqrt(2.0),0,-1/std::sqrt(2.0));

    double Dist;
    UVector3 norm;
    bool convex;

   
    UBox b1("Test Box #1",20,30,40);
    UBox b2("Test Box #2",10,10,10);
    UBox box3("BABAR Box",0.14999999999999999, 
                           24.707000000000001,  
	                   22.699999999999999) ;

// Check name
    assert(b1.GetName()=="Test Box #1");

    // Check cubic volume

    assert(b2.Capacity() == 8000);    
    assert(b1.Capacity() == 192000); 

    // Check Surface area
   
    assert(b1.SurfaceArea() == 20800);    
    assert(b2.SurfaceArea() == 6*20*20); 

// Check Inside
    assert(b1.Inside(pzero)==vecgeom::EInside::kInside);
    // assert(b1.Inside(pzero)==VUSolid::eOutside);
    assert(b1.Inside(pbigz)==vecgeom::EInside::kOutside);
    assert(b1.Inside(ponxside)==vecgeom::EInside::kSurface);
    assert(b1.Inside(ponyside)==vecgeom::EInside::kSurface);
    assert(b1.Inside(ponzside)==vecgeom::EInside::kSurface);

// Check Surface Normal
    UVector3 normal;
    bool valid;
    // Normals on Surface 
    valid=b1.Normal(ponxside,normal);
    //normal=b1.SurfaceNormal(ponxside);
    assert(ApproxEqual(normal,UVector3(1,0,0)));
    valid=b1.Normal(ponmxside,normal);
    assert(ApproxEqual(normal,UVector3(-1,0,0)));
    valid=b1.Normal(ponyside,normal);
    assert(ApproxEqual(normal,UVector3(0,1,0)));
    valid=b1.Normal(ponmyside,normal);
    assert(ApproxEqual(normal,UVector3(0,-1,0)));
    valid=b1.Normal(ponzside,normal);
    assert(ApproxEqual(normal,UVector3(0,0,1)));
    valid=b1.Normal(ponmzside,normal);
    assert(ApproxEqual(normal,UVector3(0,0,-1)));
    valid=b1.Normal(ponzsidey,normal);
    assert(ApproxEqual(normal,UVector3(0,0,1)));
    valid=b1.Normal(ponmzsidey,normal);
    assert(ApproxEqual(normal,UVector3(0,0,-1)));

    // Normals on Edges
    UVector3 edgeXY(    20.0,  30., 0.0); 
    UVector3 edgemXmY( -20.0, -30., 0.0); 
    UVector3 edgeXmY(   20.0, -30., 0.0); 
    UVector3 edgemXY(  -20.0,  30., 0.0); 
    UVector3 edgeXZ(    20.0, 0.0, 40.0); 
    UVector3 edgemXmZ( -20.0, 0.0, -40.0); 
    UVector3 edgeXmZ(   20.0, 0.0, -40.0); 
    UVector3 edgemXZ(  -20.0, 0.0, 40.0); 
    UVector3 edgeYZ(    0.0,  30.0,  40.0); 
    UVector3 edgemYmZ(  0.0, -30.0, -40.0); 
    UVector3 edgeYmZ(   0.0,  30.0, -40.0); 
    UVector3 edgemYZ(   0.0, -30.0,  40.0); 

    double invSqrt2 = 1.0 / std::sqrt( 2.0);
    double invSqrt3 = 1.0 / std::sqrt( 3.0);
  
    valid= b1.Normal( edgeXY,normal );
    assert(ApproxEqual( normal, UVector3( invSqrt2, invSqrt2, 0.0) )); 
    // G4cout << " Normal at " << edgeXY << " is " << normal 
    //    << " Expected is " << UVector3( invSqrt2, invSqrt2, 0.0) << G4endl;     
    valid= b1.Normal( edgemXmY,normal ); 
    assert(ApproxEqual( normal, UVector3( -invSqrt2, -invSqrt2, 0.0) )); 
    valid= b1.Normal( edgeXmY,normal ); 
    assert(ApproxEqual( normal, UVector3( invSqrt2, -invSqrt2, 0.0) )); 
    valid= b1.Normal( edgemXY ,normal); 
    assert(ApproxEqual( normal, UVector3( -invSqrt2, invSqrt2, 0.0) )); 

    valid= b1.Normal( edgeXZ,normal ); 
    assert(ApproxEqual( normal, UVector3(  invSqrt2, 0.0, invSqrt2) )); 
    valid= b1.Normal( edgemXmZ ,normal); 
    assert(ApproxEqual( normal, UVector3( -invSqrt2, 0.0, -invSqrt2) )); 
    valid= b1.Normal( edgeXmZ ,normal); 
    assert(ApproxEqual( normal, UVector3(  invSqrt2, 0.0, -invSqrt2) )); 
    valid= b1.Normal( edgemXZ ,normal); 
    assert(ApproxEqual( normal, UVector3( -invSqrt2, 0.0, invSqrt2) )); 

    valid= b1.Normal( edgeYZ ,normal); 
    assert(ApproxEqual( normal, UVector3( 0.0,  invSqrt2,  invSqrt2) )); 
    valid= b1.Normal( edgemYmZ ,normal); 
    assert(ApproxEqual( normal, UVector3( 0.0, -invSqrt2, -invSqrt2) )); 
    valid= b1.Normal( edgeYmZ ,normal); 
    assert(ApproxEqual( normal, UVector3( 0.0,  invSqrt2, -invSqrt2) )); 
    valid= b1.Normal( edgemYZ,normal ); 
    assert(ApproxEqual( normal, UVector3( 0.0, -invSqrt2,  invSqrt2) )); 

    // Normals on corners
    UVector3 cornerXYZ(    20.0,  30., 40.0); 
    UVector3 cornermXYZ(  -20.0,  30., 40.0); 
    UVector3 cornerXmYZ(   20.0, -30., 40.0); 
    UVector3 cornermXmYZ( -20.0, -30., 40.0); 
    UVector3 cornerXYmZ(    20.0,  30., -40.0); 
    UVector3 cornermXYmZ(  -20.0,  30., -40.0); 
    UVector3 cornerXmYmZ(   20.0, -30., -40.0); 
    UVector3 cornermXmYmZ( -20.0, -30., -40.0); 
 
    valid= b1.Normal( cornerXYZ ,normal); 
    assert(ApproxEqual( normal, UVector3(  invSqrt3,  invSqrt3, invSqrt3) )); 
    valid= b1.Normal( cornermXYZ,normal ); 
    assert(ApproxEqual( normal, UVector3( -invSqrt3,  invSqrt3, invSqrt3) )); 
    valid= b1.Normal( cornerXmYZ,normal ); 
    assert(ApproxEqual( normal, UVector3(  invSqrt3, -invSqrt3, invSqrt3) )); 
    valid= b1.Normal( cornermXmYZ,normal ); 
    assert(ApproxEqual( normal, UVector3( -invSqrt3, -invSqrt3, invSqrt3) )); 
    valid= b1.Normal( cornerXYmZ,normal ); 
    assert(ApproxEqual( normal, UVector3(  invSqrt3,  invSqrt3, -invSqrt3) )); 
    valid= b1.Normal( cornermXYmZ ,normal); 
    assert(ApproxEqual( normal, UVector3( -invSqrt3,  invSqrt3, -invSqrt3) )); 
    valid= b1.Normal( cornerXmYmZ ,normal); 
    assert(ApproxEqual( normal, UVector3(  invSqrt3, -invSqrt3, -invSqrt3) )); 
    valid= b1.Normal( cornermXmYmZ ,normal); 
    assert(ApproxEqual( normal, UVector3( -invSqrt3, -invSqrt3, -invSqrt3) )); 
    
// SafetyFromInside(P)
    Dist=b1.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,20));
    Dist=b1.SafetyFromInside(vx);
    assert(ApproxEqual(Dist,19));
    Dist=b1.SafetyFromInside(vy);
    assert(ApproxEqual(Dist,20));
    Dist=b1.SafetyFromInside(vz);
    assert(ApproxEqual(Dist,20));

// DistanceToOut(P,V)
    Dist=b1.DistanceToOut(pzero,vx,norm,convex);
    assert(ApproxEqual(Dist,20)&&ApproxEqual(norm,vx)&&convex);
    Dist=b1.DistanceToOut(pzero,vmx,norm,convex);
    assert(ApproxEqual(Dist,20)&&ApproxEqual(norm,vmx)&&convex);
    Dist=b1.DistanceToOut(pzero,vy,norm,convex);
    assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,vy)&&convex);
    Dist=b1.DistanceToOut(pzero,vmy,norm,convex);
    assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,vmy)&&convex);
    Dist=b1.DistanceToOut(pzero,vz,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vz)&&convex);
    Dist=b1.DistanceToOut(pzero,vmz,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vmz)&&convex);
    Dist=b1.DistanceToOut(pzero,vxy,norm,convex);
    assert(ApproxEqual(Dist,std::sqrt(800.))&&convex);

    Dist=b1.DistanceToOut(ponxside,vx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vx)&&convex);
    Dist=b1.DistanceToOut(ponxside,vmx,norm,convex);
    assert(ApproxEqual(Dist,40)&&ApproxEqual(norm,vmx)&&convex);
    Dist=b1.DistanceToOut(pbigx,vy,norm,convex);
    assert(ApproxEqual(Dist,30)&&ApproxEqual(norm,vy)&&convex);
    Dist=b1.DistanceToOut(ponmxside,vmx,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmx)&&convex);
    Dist=b1.DistanceToOut(ponyside,vy,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vy)&&convex);
    Dist=b1.DistanceToOut(ponmyside,vmy,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmy)&&convex);
    Dist=b1.DistanceToOut(ponzside,vz,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vz)&&convex);
    Dist=b1.DistanceToOut(ponmzside,vmz,norm,convex);
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmz)&&convex);
    

//SafetyFromOutside(P)
    Dist=b1.SafetyFromOutside(pbigx);
    assert(ApproxEqual(Dist,80));
    Dist=b1.SafetyFromOutside(pbigmx);
    assert(ApproxEqual(Dist,80));
    Dist=b1.SafetyFromOutside(pbigy);
    assert(ApproxEqual(Dist,70));
    Dist=b1.SafetyFromOutside(pbigmy);
    assert(ApproxEqual(Dist,70));
    Dist=b1.SafetyFromOutside(pbigz);
    assert(ApproxEqual(Dist,60));
    Dist=b1.SafetyFromOutside(pbigmz);
    assert(ApproxEqual(Dist,60));

// DistanceToIn(P,V)
    Dist=b1.DistanceToIn(pbigx,vmx);
    assert(ApproxEqual(Dist,80));
    Dist=b1.DistanceToIn(pbigmx,vx);
    assert(ApproxEqual(Dist,80));
    Dist=b1.DistanceToIn(pbigy,vmy);
    assert(ApproxEqual(Dist,70));
    Dist=b1.DistanceToIn(pbigmy,vy);
    assert(ApproxEqual(Dist,70));
    Dist=b1.DistanceToIn(pbigz,vmz);
    assert(ApproxEqual(Dist,60));
    Dist=b1.DistanceToIn(pbigmz,vz);
    assert(ApproxEqual(Dist,60));
    Dist=b1.DistanceToIn(pbigx,vxy);
    assert(ApproxEqual(Dist,UUtils::kInfinity));
    Dist=b1.DistanceToIn(pbigmx,vxy);
    assert(ApproxEqual(Dist,UUtils::kInfinity));

    UVector3 pJohnXZ(9,0,12);
    Dist = b2.DistanceToIn(pJohnXZ,vxmz) ;
    //    G4cout<<"b2.DistanceToIn(pJohnXZ,vxmz) = "<<Dist<<G4endl ;
     assert(ApproxEqual(Dist,UUtils::kInfinity));

    UVector3 pJohnXY(12,9,0);
    Dist = b2.DistanceToIn(pJohnXY,vmxy) ;
    //    G4cout<<"b2.DistanceToIn(pJohnXY,vmxy) = "<<Dist<<G4endl ;
    assert(ApproxEqual(Dist,UUtils::kInfinity));

    Dist = b2.DistanceToIn(pJohnXY,vmx) ;
    //    G4cout<<"b2.DistanceToIn(pJohnXY,vmx) = "<<Dist<<G4endl ;
    assert(ApproxEqual(Dist,2));

    UVector3 pMyXY(32,-11,0);
    Dist = b2.DistanceToIn(pMyXY,vmxy) ;
    //   G4cout<<"b2.DistanceToIn(pMyXY,vmxy) = "<<Dist<<G4endl ;
    assert(ApproxEqual(Dist,UUtils::kInfinity));

    Dist = b1.DistanceToIn(UVector3(-25,-35,0),vx) ;
    assert(ApproxEqual(Dist,UUtils::kInfinity));

    Dist = b1.DistanceToIn(UVector3(-25,-35,0),vy) ;
    assert(ApproxEqual(Dist,UUtils::kInfinity));
    

    Dist = b2.DistanceToIn(pJohnXY,vmx) ;
    //    G4cout<<"b2.DistanceToIn(pJohnXY,vmx) = "<<Dist<<G4endl ;
    assert(ApproxEqual(Dist,2));

    Dist=box3.DistanceToIn(UVector3(  0.15000000000000185,
                                         -22.048743592955137,
                                           2.4268539333219472),
                           UVector3(-0.76165597579890043,
                                          0.64364445891356026,
                                         -0.074515708658524193)) ;
    assert(ApproxEqual(Dist,0.0));
   
// CalculateExtent
    
    UVector3 minExtent,maxExtent;
    b1.Extent(minExtent,maxExtent);
    assert(ApproxEqual(minExtent,UVector3(-20,-30,-40)));
    assert(ApproxEqual(maxExtent,UVector3( 20, 30, 40)));
    b2.Extent(minExtent,maxExtent);
    assert(ApproxEqual(minExtent,UVector3(-10,-10,-10)));
    assert(ApproxEqual(maxExtent,UVector3( 10, 10, 10)));


    /* **********************************************************
    */ /////////////////////////////////////////////////////

    return true;
}

int main()
{

    assert(testUBox());
    return 0;
}

