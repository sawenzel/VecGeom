//
//
// TestCones




#include "base/Vector3D.h"
#include "ApproxEqual.h"
#include "volumes/Cone.h"
#include "base/Global.h"
#ifdef VECGEOM_USOLIDS
#include "UCons.hh"
#include "UVector3.hh"
#endif
#include <cmath>

//  Ensure asserts are compiled in
#undef NDEBUG
#include <cassert>


#define DELTA 0.0001

bool testingvecgeom=false;

// Returns false if actual is within wanted+/- DELTA
//         true if error
bool OutRange(double actual,double wanted)
{
    bool rng = false ;
    if (actual < wanted-DELTA || actual > wanted + DELTA ) rng = true ;
    return rng ;
}
bool OutRange(UVector3 actual,UVector3 wanted)
{
    bool rng = false ;
    if (OutRange(actual.x(),wanted.x())
    ||OutRange(actual.y(),wanted.y())
    ||OutRange(actual.z(),wanted.z())  ) rng = true ;
    return rng ;
}

template <typename Constants, class Cone_t,class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >
bool TestCons()
{
        Vec_t   pzero(0,0,0);

        Vec_t   pplx(120,0,0),pply(0,120,0),pplz(0,0,120);

        Vec_t   pmix(-120,0,0),pmiy(0,-120,0),pmiz(0,0,-120);

        Vec_t   ponmiz(0,75,-50),ponplz(0,75,50);

        Vec_t   ponr1(std::sqrt(50*50/2.0),std::sqrt(50*50/2.0),0);

        Vec_t   ponr2(std::sqrt(100*100/2.0),std::sqrt(100*100/2.0),0);

        Vec_t   ponphi1(60*std::cos(VECGEOM_NAMESPACE::kPi/6),-60*std::sin(VECGEOM_NAMESPACE::kPi/6),0);

        Vec_t   ponphi2(60*std::cos(VECGEOM_NAMESPACE::kPi/6),60*std::sin(VECGEOM_NAMESPACE::kPi/6),0);

        Vec_t   ponr2b(150,0,0);

        Vec_t pnearplz(45,45,45),pnearmiz(45,45,-45);
        Vec_t pydx(60,150,0),pbigx(500,0,0);

        Vec_t proot1(0,125,-1000),proot2(0,75,-1000);

        Vec_t pparr1(0,25,-150);   // Test case parallel to both rs of c8
        Vec_t pparr2(0,75,-50),pparr3(0,125,50);
        Vec_t vparr(0,1./std::sqrt(5.),2./std::sqrt(5.));

        Vec_t vnphi1(-std::sin(VECGEOM_NAMESPACE::kPi/6),-std::cos(VECGEOM_NAMESPACE::kPi/6),0),
              vnphi2(-std::sin(VECGEOM_NAMESPACE::kPi/6),std::cos(VECGEOM_NAMESPACE::kPi/6),0);

        Vec_t vx(1,0,0),vy(0,1,0),vz(0,0,1),
              vmx(-1,0,0),vmy(0,-1,0),vmz(0,0,-1),
              vxy(1./std::sqrt(2.),1./std::sqrt(2.),0),
              vxmy(1./std::sqrt(2.),-1./std::sqrt(2.),0),
                vmxmy(-1./std::sqrt(2.),-1./std::sqrt(2.),0),
                vmxy(-1./std::sqrt(2.),1./std::sqrt(2.),0),
                    vx2mz(1.0/std::sqrt(5.0),0,-2.0/std::sqrt(5.0)),
                vxmz(1./std::sqrt(2.),0,-1./std::sqrt(2.));

  Cone_t c1("Hollow Full Tube",50,100,50,100,50,0,2*VECGEOM_NAMESPACE::kPi),
         cn1("cn1",45.,50.,45.,50.,50,0.5*VECGEOM_NAMESPACE::kPi,0.5*VECGEOM_NAMESPACE::kPi),
         cn2("cn1",45.,50.,45.,50.,50,0.5*VECGEOM_NAMESPACE::kPi,1.5*VECGEOM_NAMESPACE::kPi),
         c2("Hollow Full Cone",50,100,50,200,50,-1,2*VECGEOM_NAMESPACE::kPi),
         c3("Hollow Cut Tube",50,100,50,100,50,-VECGEOM_NAMESPACE::kPi/6,VECGEOM_NAMESPACE::kPi/3),
         c4("Hollow Cut Cone",50,100,50,200,50,-VECGEOM_NAMESPACE::kPi/6,VECGEOM_NAMESPACE::kPi/3),
         c5("Hollow Cut Cone",25,50,75,150,50,0,1.5*VECGEOM_NAMESPACE::kPi),
         c6("Solid Full Tube",0,150,0,150,50,0,2*VECGEOM_NAMESPACE::kPi),
         c7("Thin Tube",95,100,95,100,50,0,2*VECGEOM_NAMESPACE::kPi),
         c8a("Solid Full Cone2",0,100,0,150,50,0,2*VECGEOM_NAMESPACE::kPi),
         c8b("Hollow Full Cone2",50,100,100,150,50,0,2*VECGEOM_NAMESPACE::kPi),
         c8c("Hollow Full Cone2inv",100,150,50,100,50,0,2*VECGEOM_NAMESPACE::kPi),
         c9("Exotic Cone",50,60,
            0,           // 1.0e-7,   500*kRadTolerance,
                           10,50,0,2*VECGEOM_NAMESPACE::kPi),
         cms("cms cone",0.0,70.0,0.0,157.8,2949.0,0.0,6.283185307179586);

   Cone_t cms2("RearAirCone",401.0,1450.0,
                            1020.0,1450.0,175.0,0.0,6.283185307179586) ;
      Cone_t   ctest10( "aCone", 20., 60., 80., 140.,
                           100., 10*VECGEOM_NAMESPACE::kPi/180., 300*VECGEOM_NAMESPACE::kPi/180. );

  Vec_t pct10(60,0,0);
  Vec_t pct10mx(-50,0,0);
  Vec_t pct10phi1(60*std::cos(10.*VECGEOM_NAMESPACE::kPi/180.),60*std::sin(10*VECGEOM_NAMESPACE::kPi/180.),0);
  Vec_t pct10phi2(60*std::cos(50.*VECGEOM_NAMESPACE::kPi/180.),-60*std::sin(50*VECGEOM_NAMESPACE::kPi/180.),0);

  Vec_t pct10e1(-691-500,174, 404 );

  Vec_t pct10e2( 400-500, 20.9, 5.89 );

  Vec_t pct10e3( 456-500, 13, -14.7 );

  Vec_t pct10e4( 537-500, 1.67, -44.1 );
  // point P is outside
  Vec_t pct10e5(537, 1.67, -44.1);

  Vec_t pct10e6(1e+03, -63.5, -213 );
  double tolerance=vecgeom::kTolerance*1000;

  double a1,a2,a3,am;

  a1=pct10e2.x()-pct10e1.x();
  a2=pct10e2.y()-pct10e1.y();
  a3=pct10e2.z()-pct10e1.z();
  am=std::sqrt(a1*a1+a2*a2+a3*a3);
  Vec_t  d1(a1/am,a2/am,a3/am);
  //std::cout<<d1.x()<<"\t"<<d1.y()<<"\t"<<d1.z()<<std::endl;

  a1=pct10e3.x()-pct10e2.x();
  a2=pct10e3.y()-pct10e2.y();
  a3=pct10e3.z()-pct10e2.z();
  am=std::sqrt(a1*a1+a2*a2+a3*a3);
  Vec_t  d2(a1/am,a2/am,a3/am);
  //std::cout<<d2.x()<<"\t"<<d2.y()<<"\t"<<d2.z()<<std::endl;

  a1=pct10e4.x()-pct10e3.x();
  a2=pct10e4.y()-pct10e3.y();
  a3=pct10e4.z()-pct10e3.z();
  am=std::sqrt(a1*a1+a2*a2+a3*a3);
  Vec_t  d3(a1/am,a2/am,a3/am);
  //std::cout<<d3.x()<<"\t"<<d3.y()<<"\t"<<d3.z()<<std::endl;


  // 19.01.04 modified test10 info:

  Vec_t  pt10s1(  6.454731216775542,
            -90.42080754048007,
                        100.                 );

  Vec_t  pt10s2( 22.65282328600368,
                        -69.34877585931267,
                         76.51600623610082 );

  Vec_t  pt10s3( 51.28206938732319,
            -32.10510677306267,
                         35.00932544708616 );

  Vec_t    vt10d( 0.4567090876640433 ,
                          0.5941309830320264,
                         -0.6621368319663807 );


  Vec_t norm;

  // Check name

  assert(c1.GetName()=="Hollow Full Tube");

  // Check Cubic volume
  double vol,volCheck;
  vol = c1.Capacity();
  volCheck = 2*VECGEOM_NAMESPACE::kPi*50*(100*100-50*50);
  assert(ApproxEqual(vol,volCheck));

  vol = c6.Capacity();
  volCheck = 2*VECGEOM_NAMESPACE::kPi*50*(150*150);
  assert(ApproxEqual(vol,volCheck));

  // Check Surface area
  vol = c1.SurfaceArea();
  volCheck = 2*VECGEOM_NAMESPACE::kPi*(50*2*50+100*2*50+100*100-50*50);
  assert(ApproxEqual(vol,volCheck));

  // Check Inside
  VUSolid::EnumInside in;
  std::cout.precision(16) ;
  //std::cout << "Testing Cone_t::Inside...\n";

  in = ctest10.Inside(pct10e1);
  //std::cout << "ctest10.Inside(pct10e1) = " <<in<< std::endl;
  assert(in==vecgeom::EInside::kOutside);

  in = ctest10.Inside(pct10e2);
  //std::cout << "ctest10.Inside(pct10e2) = " <<in<< std::endl;
  assert(in==vecgeom::EInside::kInside);

  in = ctest10.Inside(pct10e3);
  //std::cout << "ctest10.Inside(pct10e3) = " <<in<< std::endl;
  assert(in==vecgeom::EInside::kInside);

  in = ctest10.Inside(pct10e4);
  //std::cout << "ctest10.Inside(pct10e4) = " <<in<< std::endl;
  assert(in==vecgeom::EInside::kOutside);

  in = ctest10.Inside(pct10e5);
  //std::cout << "ctest10.Inside(pct10e5) = " <<in<< std::endl;
  assert(in==vecgeom::EInside::kOutside);

  in = ctest10.Inside(pct10e6);
  //std::cout << "ctest10.Inside(pct10e6) = " <<in<< std::endl;
  assert(in==vecgeom::EInside::kOutside);

  in = ctest10.Inside(pct10mx);
  //std::cout << "ctest10.Inside(pct10mx) = " <<in<< std::endl;
  assert(in==vecgeom::EInside::kSurface);

  in = ctest10.Inside(pt10s1);
  //std::cout << "ctest10.Inside(pt10s1) = " <<in<< std::endl;
  assert(in==vecgeom::EInside::kSurface);

  in = ctest10.Inside(pt10s2);
  //std::cout << "ctest10.Inside(pt10s2) = " <<in<< std::endl;
  assert(in==vecgeom::EInside::kSurface);

  in = ctest10.Inside(pt10s3);
  //std::cout << "ctest10.Inside(pt10s3) = " <<in<< std::endl;
  assert(in==vecgeom::EInside::kOutside);


      if (c1.Inside(pzero)!=vecgeom::EInside::kOutside)
            std::cout << "Error A" << std::endl;
        if (c6.Inside(pzero)!=vecgeom::EInside::kInside)
            std::cout << "Error A2" << std::endl;
        if (c1.Inside(pplx)!=vecgeom::EInside::kOutside)
            std::cout << "Error B1" << std::endl;
        if (c2.Inside(pplx)!=vecgeom::EInside::kInside)
            std::cout << "Error B2" << std::endl;
        if (c3.Inside(pplx)!=vecgeom::EInside::kOutside)
            std::cout << "Error B3" << std::endl;
        if (c4.Inside(pplx)!=vecgeom::EInside::kInside)
            std::cout << "Error B4" << std::endl;
        if (c1.Inside(ponmiz)!=vecgeom::EInside::kSurface)
            std::cout << "Error C" << std::endl;
        if (c1.Inside(ponplz)!=vecgeom::EInside::kSurface)
            std::cout << "Error D" << std::endl;
        if (c1.Inside(ponr1)!=vecgeom::EInside::kSurface)
            std::cout << "Error E" << std::endl;
        if (c1.Inside(ponr2)!=vecgeom::EInside::kSurface)
            std::cout << "Error F" << std::endl;
        if (c3.Inside(ponphi1)!=vecgeom::EInside::kSurface)
            std::cout << "Error G" << std::endl;
        if (c3.Inside(ponphi2)!=vecgeom::EInside::kSurface)
            std::cout << "Error H" << std::endl;

        if (c5.Inside(Vec_t(70,1,0))!=vecgeom::EInside::kInside)
            std::cout << "Error I" << std::endl;
        if (c5.Inside(Vec_t(50,-50,0))!=vecgeom::EInside::kOutside)
            std::cout << "Error I2" << std::endl;
        if (c5.Inside(Vec_t(70,0,0))!=vecgeom::EInside::kSurface)
            std::cout << "Error I3" << std::endl;
    // on tolerant r, inside z, within phi
        if (c5.Inside(Vec_t(100,0,0))!=vecgeom::EInside::kSurface)
            std::cout << "Error I4" << std::endl;
        if (c3.Inside(Vec_t(100,0,0))!=vecgeom::EInside::kSurface)
            std::cout << "Error I5" << std::endl;
    // on tolerant r, tolerant z, within phi
        if (c5.Inside(Vec_t(100,0,50))!=vecgeom::EInside::kSurface)
            std::cout << "Error I4" << std::endl;
        if (c3.Inside(Vec_t(100,0,50))!=vecgeom::EInside::kSurface)
            std::cout << "Error I5" << std::endl;


    //std::cout << "Testing Cone_t::SurfaceNormal...\n";

    Vec_t normal;
    double p2=1./std::sqrt(2.),p3=1./std::sqrt(3.);
    bool valid,convex;

    valid=cn1.Normal(Vec_t(0.,50.,0.),normal);
    assert(ApproxEqual(normal,Vec_t(p2,p2,0.))&&valid);
    valid=cn1.Normal(Vec_t(0.,45.,0.),normal);
    assert(ApproxEqual(normal,Vec_t(p2,-p2,0.)));
    valid=cn1.Normal(Vec_t(0.,45.,50.),normal);
    assert(ApproxEqual(normal,Vec_t(p3,-p3,p3)));
    valid=cn1.Normal(Vec_t(0.,45.,-50.),normal);
    assert(ApproxEqual(normal,Vec_t(p3,-p3,-p3)));
    valid=cn1.Normal(Vec_t(-50.,0.,-50.),normal);
    assert(ApproxEqual(normal,Vec_t(-p3,-p3,-p3)));
    valid=cn1.Normal(Vec_t(-50.,0.,0.),normal);
    assert(ApproxEqual(normal,Vec_t(-p2,-p2,0.)));
    valid=cn2.Normal(Vec_t(50.,0.,0.),normal);
    assert(ApproxEqual(normal,Vec_t(p2,p2,0.)));
    valid=c6.Normal(Vec_t(0.,0.,50.),normal);
    assert(ApproxEqual(normal,Vec_t(0.,0.,1.)));

    valid=c1.Normal(ponplz,norm);
    if (OutRange(norm,Vec_t(0,0,1)))
        std::cout << "Error A " << norm << std::endl;
    valid=c1.Normal(ponmiz,norm);
    if (OutRange(norm,Vec_t(0,0,-1)))
        std::cout << "Error B " << norm << std::endl;
    valid=c1.Normal(ponr1,norm);
    if (OutRange(norm,Vec_t(-1.0/std::sqrt(2.0),-1.0/std::sqrt(2.0),0)))
        std::cout << "Error C " << norm << std::endl;
    valid=c1.Normal(ponr2,norm);
    if (OutRange(norm,Vec_t(1.0/std::sqrt(2.0),1.0/std::sqrt(2.0),0)))
        std::cout << "Error D " << norm << std::endl;
    valid=c3.Normal(ponphi1,norm);
    if (OutRange(norm,vnphi1))
        std::cout << "Error E " << norm << std::endl;
    valid=c3.Normal(ponphi2,norm);
    if (OutRange(norm,vnphi2))
        std::cout << "Error F " << norm << std::endl;
    valid=c4.Normal(ponr2b,norm);
    if (OutRange(norm,vxmz))
        std::cout << "Error G " << norm << std::endl;

    valid=c5.Normal(Vec_t(51,0,-50),norm);
    if (OutRange(norm,Vec_t(0.,-p2,-p2)))
        std::cout << "Errot H " << norm << std::endl;

    //std::cout << "Testing Cone_t::DistanceToOut...\n";
        double dist;

    dist=c4.SafetyFromInside(ponphi1);
    if (OutRange(dist,0))
        std::cout << "Error A " << dist << std::endl;

    dist=c1.SafetyFromInside(ponphi1);
    if (OutRange(dist,10))
        std::cout << "Error B " << dist << std::endl;

    dist=c1.SafetyFromInside(pnearplz);
    if (OutRange(dist,5))
        std::cout << "Error C " << dist << std::endl;
    dist=c1.SafetyFromInside(pnearmiz);
    if (OutRange(dist,5))
        std::cout << "Error D " << dist << std::endl;

    dist=c1.SafetyFromInside(ponr1);
    if (OutRange(dist,0))
        std::cout << "Error E " << dist << std::endl;
    dist=c1.SafetyFromInside(ponr2);
    if (OutRange(dist,0))
        std::cout << "Error F " << dist << std::endl;

    dist=c6.SafetyFromInside(pzero);
    if (OutRange(dist,50))
        std::cout << "Error G " << dist << std::endl;

    dist=c5.SafetyFromInside(Vec_t(0,-70,0));
    if (OutRange(dist,0))
        std::cout << "Error H " << dist << std::endl;

        //std::cout << "Testing Cone_t::DistanceToOut...\n";

    dist=c4.DistanceToOut(pplx,vx,norm,convex);
    if (OutRange(dist,30)||OutRange(norm,vxmz)||!convex)
        std::cout << "Error Rmax1 " << dist << std::endl;

    dist=c2.DistanceToOut(pplx,vx,norm,convex);
    if (OutRange(dist,30)||OutRange(norm,vxmz)||!convex)
        std::cout << "Error Rmax2 " << dist << std::endl;

    dist=c4.DistanceToOut(pplx,vmx,norm,convex);
    if(testingvecgeom){
    if (OutRange(dist,70))
        std::cout << "Error Rmin1 " << dist << std::endl;


    dist=c2.DistanceToOut(pplx,vmx,norm,convex);
    if (OutRange(dist,70))
        std::cout << "Error Rmin2 " << dist << std::endl;
    }else{

    if (OutRange(dist,70)||convex)
        std::cout << "Error Rmin1 " << dist << std::endl;
    dist=c2.DistanceToOut(pplx,vmx,norm,convex);
    if (OutRange(dist,70)||convex)
        std::cout << "Error Rmin2 " << dist << std::endl;

    }
    dist=c3.DistanceToOut(ponphi1,vmy,norm,convex);
    if (OutRange(dist,0)||
        OutRange(norm,vnphi1)||
        !convex)
        std::cout << "Error PhiS 1" << dist << std::endl;
    dist=c3.DistanceToOut(ponphi1,vy,norm,convex);
    //norm=pNorm->unit();
    if (OutRange(dist,2*60*std::sin(VECGEOM_NAMESPACE::kPi/6))||
        OutRange(norm,vnphi2)||
        !convex)
        std::cout << "Error PhiS 2" << dist << std::endl;

    dist=c3.DistanceToOut(ponphi2,vy,norm,convex);
    if (OutRange(dist,0)||
        OutRange(norm,vnphi2)||
        !convex)
        std::cout << "Error PhiE 1" << dist << std::endl;
    dist=c3.DistanceToOut(ponphi2,vmy,norm,convex);
    if (OutRange(dist,2*60*std::sin(VECGEOM_NAMESPACE::kPi/6))||
        OutRange(norm,vnphi1)||
        !convex)
        std::cout << "Error PhiS 2" << dist << std::endl;


    dist=c6.DistanceToOut(ponplz,vmz,norm,convex);
    if (OutRange(dist,100)||
        OutRange(norm,vmz)||
        !convex)
        std::cout << "Error Top Z 1" << dist << std::endl;
    dist=c6.DistanceToOut(ponplz,vz,norm,convex);
    if (OutRange(dist,0)||
        OutRange(norm,vz)||
        !convex)
        std::cout << "Error Top Z 2" << dist << std::endl;

    dist=c6.DistanceToOut(ponmiz,vz,norm,convex);
    if (OutRange(dist,100)||
        OutRange(norm,vz)||
        !convex)
        std::cout << "Error Lower Z 1" << dist << std::endl;
    dist=c6.DistanceToOut(ponmiz,vmz,norm,convex);
    if (OutRange(dist,0)||
        OutRange(norm,vmz)||
        !convex)
        std::cout << "Error Lower Z 2" << dist << std::endl;

// Test case for rmax root bug
    dist=c7.DistanceToOut(ponr2,vmx,norm,convex);
    if(testingvecgeom){
    if (OutRange(dist,100/std::sqrt(2.)-std::sqrt(95*95-100*100/2.)))
        std::cout << "Error rmax root bug" << dist << std::endl;
    }else{
     if (OutRange(dist,100/std::sqrt(2.)-std::sqrt(95*95-100*100/2.))||convex)
        std::cout << "Error rmax root bug" << dist << std::endl;
    }

// Parallel radii test cases
    dist=c8a.DistanceToOut(pparr2,vparr,norm,convex);
    if (OutRange(dist,100.*std::sqrt(5.)/2.)||
                     !convex||
                     OutRange(norm,vz))
        std::cout << "Error solid parr2a " <<dist << std::endl;
    dist=c8a.DistanceToOut(pparr2,-vparr,norm,convex);
    if (OutRange(dist,0)||
        !convex||
        OutRange(norm,vmz))
        std::cout << "Error solid parr2b " <<dist << std::endl;

    dist=c8a.DistanceToOut(pparr2,vz,norm,convex);
    if (OutRange(dist,100)||
        !convex||
        OutRange(norm,vz))
        std::cout << "Error solid parr2c " <<dist << std::endl;
    dist=c8a.DistanceToOut(pparr2,vmz,norm,convex);
    if (OutRange(dist,0)||
        !convex||
        OutRange(norm,vmz))
        std::cout << "Error solid parr2d " <<dist << std::endl;

    dist=c8a.DistanceToOut(pparr3,vparr,norm,convex);
    if (OutRange(dist,0)||
        !convex||
        OutRange(norm,vz))
        std::cout << "Error solid parr3a " <<dist << std::endl;

    dist=c8a.DistanceToOut(pparr3,-vparr,norm,convex);
    if (OutRange(dist,100*std::sqrt(5.)/2.)||
        !convex||
        OutRange(norm,vmz))
        std::cout << "Error solid parr3b " <<dist << std::endl;
    dist=c8a.DistanceToOut(pparr3,vz,norm,convex);
    if (OutRange(dist,0)||
        !convex||
        OutRange(norm,vz))
        std::cout << "Error solid parr3c " <<dist << std::endl;

    dist=c8a.DistanceToOut(pparr3,vmz,norm,convex);
    if (OutRange(dist,50)||
        !convex||
        OutRange(norm,Vec_t(0,2./std::sqrt(5.0),-1./std::sqrt(5.0))))
        std::cout << "Error solid parr3d " <<dist << std::endl;


    dist=c8b.DistanceToOut(pparr2,vparr,norm,convex);
    if (OutRange(dist,100*std::sqrt(5.)/2.)||
                     !convex||
                     OutRange(norm,vz))
        std::cout << "Error hollow parr2a " <<dist << std::endl;
    dist=c8b.DistanceToOut(pparr2,-vparr,norm,convex);
    if (OutRange(dist,0)||
        !convex||
        OutRange(norm,vmz))
        std::cout << "Error hollow parr2b " <<dist << std::endl;

    dist=c8b.DistanceToOut(pparr2,vz,norm,convex);
    if(testingvecgeom)
      {
        if (OutRange(dist,50))
        std::cout << "Error hollow parr2c " <<dist << std::endl;
      }
    else
      {
        if (OutRange(dist,50)||convex)
        std::cout << "Error hollow parr2c " <<dist << std::endl;
      }
    dist=c8b.DistanceToOut(pparr2,vmz,norm,convex);
    if (OutRange(dist,0)||
        !convex||
        OutRange(norm,vmz))
        std::cout << "Error hollow parr2d " <<dist << std::endl;


    dist=c8b.DistanceToOut(pparr3,vparr,norm,convex);
    if (OutRange(dist,0)||
        !convex||
        OutRange(norm,vz))
        std::cout << "Error hollow parr3a " <<dist << std::endl;
    dist=c8b.DistanceToOut(pparr3,-vparr,norm,convex);
    if (OutRange(dist,100.*std::sqrt(5.)/2.)||
        !convex||
        OutRange(norm,vmz))
        std::cout << "Error hollow parr3b " <<dist << std::endl;
    dist=c8b.DistanceToOut(pparr3,vz,norm,convex);
    if (OutRange(dist,0)||
        !convex||
        OutRange(norm,vz))
        std::cout << "Error hollow parr3c " <<dist << std::endl;

    dist=c8b.DistanceToOut(pparr3,vmz,norm,convex);
    if (OutRange(dist,50)||
        !convex||
        OutRange(norm,Vec_t(0,2./std::sqrt(5.),-1.0/std::sqrt(5.))))
        std::cout << "Error hollow parr3d " <<dist << std::endl;

    dist=c9.DistanceToOut(Vec_t(1e3*tolerance,0,50),
                              vx2mz,norm,convex);
    if(testingvecgeom)
      {
      if (OutRange(dist,111.8033988))
std::cout<<"Error:c9.Out((1e3*tolerance,0,50),vx2mz,...) = " <<dist << std::endl;
      }else{
       if (OutRange(dist,111.8033988)||
        !convex||
        OutRange(norm,Vec_t(0,0,-1.0)))
std::cout<<"Error:c9.Out((1e3*tolerance,0,50),vx2mz,...) = " <<dist << std::endl;

    }
    dist=c9.DistanceToOut(Vec_t(5,0,50),
                              vx2mz,norm,convex);
    if (OutRange(dist,111.8033988)||
        !convex||
        OutRange(norm,Vec_t(0,0,-1.0)))
        std::cout << "Error:c9.Out((5,0,50),vx2mz,...) = " <<dist << std::endl;

    dist=c9.DistanceToOut(Vec_t(10,0,50),
                              vx2mz,norm,convex);
    if(testingvecgeom)
      {
      if (OutRange(dist,111.8033988))
        std::cout << "Error:c9.Out((10,0,50),vx2mz,...) = " <<dist << std::endl;
      }
    else
      {
        if (OutRange(dist,111.8033988)||
        !convex||
        OutRange(norm,Vec_t(0,0,-1.0)))
        std::cout << "Error:c9.Out((10,0,50),vx2mz,...) = " <<dist << std::endl;

      }
    dist=cms.DistanceToOut(
        Vec_t(0.28628920024909,-0.43438111004815,-2949.0),
        Vec_t(6.0886686196674e-05,-9.2382200635766e-05,0.99999999387917),
        norm,convex);
    if (OutRange(dist,5898.0))
    std::cout << "Error:cms.DistToOut() =  " <<dist << std::endl;

    dist=cms.DistanceToOut(
        Vec_t(0.28628920024909,-0.43438111004815,
                     -2949.0 + tolerance*0.25),
        Vec_t(6.0886686196674e-05,-9.2382200635766e-05,0.99999999387917),
        norm,convex);
    if (OutRange(dist,5898.0))
    std::cout << "Error:cms.DistToOut(+) =  " <<dist << std::endl;

    dist=cms.DistanceToOut(Vec_t(0.28628920024909,
                                            -0.43438111004815,
                                            -2949.0 - tolerance*0.25),
                               Vec_t(6.0886686196674e-05,
                                            -9.2382200635766e-05,
                                             0.99999999387917),
        norm,convex);
    if (OutRange(dist,5898.0))
    std::cout << "Error:cms.DistToOut(-) =  " <<dist << std::endl;

    dist=cms2.DistanceToOut(Vec_t(-344.13684353113,
                                       258.98049377272,
                                              -158.20772167926),
                                Vec_t(-0.30372024336672,
                                              -0.5581146924652,
                                               0.77218003329776),
                                norm,convex);
    if (OutRange(dist,0.))
 std::cout<<"cms2.DistanceToOut(Vec_t(-344.13684 ... = "<<dist<<std::endl;

    dist=ctest10.DistanceToOut(pct10e2,
                              d1,norm,convex);
    //norm=pNorm->unit();
    if (OutRange(dist,111.8033988)||
        !convex||
        OutRange(norm,Vec_t(0,0,-1.0)))
      std::cout << "ctest10.DistanceToOut(pct10e2,d1,...) = " <<dist <<" p is on "<<ctest10.Inside(pct10e2)<<" p="<<pct10e2<< std::endl;
    dist=ctest10.DistanceToOut(pct10e3,
                              d1,norm,convex);
    //norm=pNorm->unit();
    if (OutRange(dist,111.8033988)||
        !convex||
        OutRange(norm,Vec_t(0,0,-1.0)))
       std::cout << "ctest10.DistanceToOut(pct10e3,d1,...) = " <<dist << std::endl;

    /////////////////////////////////////////////
    //

    //std::cout << "Testing Cone_t::DistanceToIn(p) ...\n";


    dist=c1.SafetyFromOutside(pzero);
    if (OutRange(dist,50))
      std::cout << "Error A " << dist << std::endl;

    dist=c1.SafetyFromOutside(pplx);
    if (OutRange(dist,20))
      std::cout << "Error B " << dist << std::endl;

    dist=c1.SafetyFromOutside(pply);
    if (OutRange(dist,20))
      std::cout << "Error C " << dist << std::endl;

    dist=c4.SafetyFromOutside(pply);
    if (OutRange(dist,120*std::sin(VECGEOM_NAMESPACE::kPi/3)))
      std::cout << "Error D " << dist << std::endl;

    dist=c4.SafetyFromOutside(pmiy);
    if (OutRange(dist,120*std::sin(VECGEOM_NAMESPACE::kPi/3)))
      std::cout << "Error D " << dist << std::endl;

    dist=c1.SafetyFromOutside(pplz);
    if (OutRange(dist,70))
        std::cout << "Error E " << dist << std::endl;
// Check with both rmins=0
    dist=c5.SafetyFromOutside(pplx);
    if (OutRange(dist,20./std::sqrt(2.)))
      std::cout << "Error F " << dist << std::endl;

    /////////////////////////////////////////////////////
    //

    //std::cout << "Testing Cone_t::DistanceToIn(p,v,...) ...\n";

    dist=c1.DistanceToIn(pplz,vmz);
    if (OutRange(dist,Constants::kInfinity))
      std::cout << "Error A " << dist << std::endl;

    dist=c2.DistanceToIn(pplz,vmz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c2.DistanceToIn(pplz,vmz) = " << dist << std::endl;

    dist=c3.DistanceToIn(pplz,vmz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c3.DistanceToIn(pplz,vmz) = " << dist << std::endl;

    dist=c4.DistanceToIn(pplz,vmz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c4.DistanceToIn(pplz,vmz) = " << dist << std::endl;

    dist=c5.DistanceToIn(pplz,vmz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c5.DistanceToIn(pplz,vmz) = " << dist << std::endl;

    dist=c6.DistanceToIn(pplz,vmz);
    if (OutRange(dist,70.0))
    std::cout << "Error:c6.DistanceToIn(pplz,vmz) = " << dist << std::endl;

    dist=c7.DistanceToIn(pplz,vmz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c7.DistanceToIn(pplz,vmz) = " << dist << std::endl;

    dist=c8a.DistanceToIn(pplz,vmz);
    if (OutRange(dist,70.0))
    std::cout << "Error:c8a.DistanceToIn(pplz,vmz) = " << dist << std::endl;

    dist=c8b.DistanceToIn(pplz,vmz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c8b.DistanceToIn(pplz,vmz) = " << dist << std::endl;

    dist=c8c.DistanceToIn(pplz,vmz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c8c.DistanceToIn(pplz,vmz) = " << dist << std::endl;

    if(testingvecgeom)
    //Cone is with Rmin=Rmax=0 at +dz
    //this is creating a very small cut at dz in USolids implementation(at construction) 
      {
      dist=c9.DistanceToIn(pplz,vmz);
      if (OutRange(dist,70.0))
      std::cout << "Error:c9.DistanceToIn(pplz,vmz) = " << dist << std::endl;

      dist=c9.DistanceToIn(Vec_t(0,0,50),vmz);
      if (OutRange(dist,0.0))
      std::cout << "Error:c9.DistanceToIn((0,0,50),vmz) = " << dist << std::endl;
      }
    else{
      dist=c9.DistanceToIn(pplz,vmz);
      if (OutRange(dist,Constants::kInfinity))
      std::cout << "Error:c9.DistanceToIn(pplz,vmz) = " << dist << std::endl;

      dist=c9.DistanceToIn(Vec_t(0,0,50),vmz);
      if (OutRange(dist,Constants::kInfinity))
      std::cout << "Error:c9.DistanceToIn((0,0,50),vmz) = " << dist << std::endl;
    }

    ///////////////

    dist=c1.DistanceToIn(pmiz,vz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error A " << dist << std::endl;

    dist=c2.DistanceToIn(pmiz,vz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c2.DistanceToIn(pmiz,vz) = " << dist << std::endl;

    dist=c3.DistanceToIn(pmiz,vz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c3.DistanceToIn(pmiz,vz) = " << dist << std::endl;

    dist=c4.DistanceToIn(pmiz,vz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c4.DistanceToIn(pmiz,vz) = " << dist << std::endl;

    dist=c5.DistanceToIn(pmiz,vz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c5.DistanceToIn(pmiz,vz) = " << dist << std::endl;

    dist=c6.DistanceToIn(pmiz,vz);
    if (OutRange(dist,70.0))
    std::cout << "Error:c6.DistanceToIn(pmiz,vz) = " << dist << std::endl;

    dist=c7.DistanceToIn(pmiz,vz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c7.DistanceToIn(pmiz,vz) = " << dist << std::endl;

    dist=c8a.DistanceToIn(pmiz,vz);
    if (OutRange(dist,70.0))
    std::cout << "Error:c8a.DistanceToIn(pmiz,vz) = " << dist << std::endl;

    dist=c8b.DistanceToIn(pmiz,vz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c8b.DistanceToIn(pmiz,vz) = " << dist << std::endl;

    dist=c8c.DistanceToIn(pmiz,vz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c8c.DistanceToIn(pmiz,vz) = " << dist << std::endl;

    dist=c9.DistanceToIn(pmiz,vz);
    if (OutRange(dist,Constants::kInfinity))
    std::cout << "Error:c9.DistanceToIn(pmiz,vz) = " << dist << std::endl;

    //////////////

    dist=c1.DistanceToIn(pplx,vmx);
    if (OutRange(dist,20))
      std::cout << "Error B " << dist << std::endl;
    dist=c1.DistanceToIn(pplz,vx);
    if (OutRange(dist,Constants::kInfinity))
      std::cout << "Error C " << dist << std::endl;
    dist=c4.DistanceToIn(pply,vmy);
    if (OutRange(dist,Constants::kInfinity))
      std::cout << "Error D " << dist << std::endl;

    dist=c1.DistanceToIn(pydx,vmy);
    if (OutRange(dist,70))
      std::cout << "Error E " << dist << std::endl;
    dist=c3.DistanceToIn(pydx,vmy);
    if (OutRange(dist,150-60*std::tan(VECGEOM_NAMESPACE::kPi/6)))
      std::cout << "Error F " << dist << std::endl;

    dist=c1.DistanceToIn(pplx,vmx);
    if (OutRange(dist,20))
      std::cout << "Error G " << dist << std::endl;
    dist=c1.DistanceToIn(pplx,vx);
    if (OutRange(dist,Constants::kInfinity))
      std::cout << "Error G2 " << dist << std::endl;

    dist=c4.DistanceToIn(pbigx,vmx);
    if (OutRange(dist,350))
        std::cout << "Error G3 " << dist << std::endl;

    dist=c4.DistanceToIn(pzero,vx);
    if (OutRange(dist,50))
      std::cout << "Error H " << dist << std::endl;

    dist=c1.DistanceToIn(ponr2,vx);
    if (OutRange(dist,Constants::kInfinity))
        std::cout << "Error I" << dist << std::endl;
    dist=c1.DistanceToIn(ponr2,vmx);
    if (OutRange(dist,0))
        std::cout << "Error I2" << dist << std::endl;

    dist=c1.DistanceToIn(ponr1,vx);
    if (OutRange(dist,0))
        std::cout << "Error J" << dist << std::endl;
    dist=c1.DistanceToIn(ponr1,vmx);
    if (OutRange(dist,2.0*std::sqrt(50*50/2.)))
        std::cout << "Error J2" << dist << std::endl;

    dist=c1.DistanceToIn(ponr2,vmxmy);
    if (OutRange(dist,0))
        std::cout << "Error K" << dist << std::endl;

// Parallel test case -> parallel to both radii
    dist=c8b.DistanceToIn(pparr1,vparr);
    if (OutRange(dist,100*std::sqrt(5.)/2.))
        std::cout << "Error parr1 " << dist << std::endl;
    dist=c8b.DistanceToIn(pparr2,-vparr);
    if (OutRange(dist,Constants::kInfinity))
        std::cout << "Error parr2 " << dist << std::endl;
    dist=c8b.DistanceToIn(pparr3,vparr);
    if (OutRange(dist,Constants::kInfinity))
        std::cout << "Error parr3a " << dist << std::endl;
    dist=c8b.DistanceToIn(pparr3,-vparr);
    if (OutRange(dist,0))
        std::cout << "Error parr3b " << dist << std::endl;

// Check we don't Hit `shadow cone' at `-ve radius' on rmax or rmin
    dist=c8a.DistanceToIn(proot1,vz);
    if (OutRange(dist,1000))
        std::cout << "Error shadow rmax root problem " << dist << std::endl;

    dist=c8c.DistanceToIn(proot2,vz);
    if (OutRange(dist,1000))
        std::cout << "Error shadow rmin root problem " << dist << std::endl;

        dist = cms2.DistanceToIn(Vec_t(-344.13684353113,
                                                258.98049377272,
                                               -158.20772167926),
                 Vec_t(-0.30372022869765,
                           -0.55811472925794,
                           0.77218001247454)) ;
    if (OutRange(dist,Constants::kInfinity))
    std::cout<<"cms2.DistanceToIn(Vec_t(-344.1 ... = "<<dist<<std::endl;

    dist=ctest10.DistanceToIn(pct10,vx);
    if (OutRange(dist,Constants::kInfinity))
        std::cout << "ctest10.DistanceToIn(pct10,vx) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10,vmx);
    if (OutRange(dist,110))
        std::cout << "ctest10.DistanceToIn(pct10,vmx) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10,vy);
    if (OutRange(dist,10.57961))
        std::cout << "ctest10.DistanceToIn(pct10,vy) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10,vmy);
    if (OutRange(dist,71.5052))
        std::cout << "ctest10.DistanceToIn(pct10,vmy) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10,vz);
    if (OutRange(dist,Constants::kInfinity))
        std::cout << "ctest10.DistanceToIn(pct10,vz) = " << dist << std::endl;


    dist=ctest10.DistanceToIn(pct10phi1,vx);
    if (OutRange(dist,Constants::kInfinity))
        std::cout << "ctest10.DistanceToIn(pct10phi1,vx) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10phi1,vmx);
    if (OutRange(dist,0))
        std::cout << "ctest10.DistanceToIn(pct10phi1,vmx) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10phi1,vy);
    if (OutRange(dist,0))
        std::cout << "ctest10.DistanceToIn(pct10phi1,vy) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10phi1,vmy);
    if (OutRange(dist,80.83778))
        std::cout << "ctest10.DistanceToIn(pct10phi1,vmy) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10phi1,vz);
    if (OutRange(dist,33.3333))
        std::cout << "ctest10.DistanceToIn(pct10phi1,vz) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10phi1,vmz);
    if (OutRange(dist,Constants::kInfinity))
        std::cout << "ctest10.DistanceToIn(pct10phi1,vmz) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10phi2,vx);
    if (OutRange(dist,Constants::kInfinity))
        std::cout << "ctest10.DistanceToIn(pct10phi2,vx) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10phi2,vmx);
    if (OutRange(dist,0))
        std::cout << "ctest10.DistanceToIn(pct10phi2,vmx) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10phi2,vy);
    if (OutRange(dist,77.78352))
        std::cout << "ctest10.DistanceToIn(pct10phi2,vy) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10phi2,vmy);
    if (OutRange(dist,0))
        std::cout << "ctest10.DistanceToIn(pct10phi2,vmy) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10phi2,vz);
    if (OutRange(dist,33.3333))
        std::cout << "ctest10.DistanceToIn(pct10phi2,vz) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10phi2,vmz);
    if (OutRange(dist,Constants::kInfinity))
        std::cout << "ctest10.DistanceToIn(pct10phi2,vmz) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10mx,vx);
    if (OutRange(dist,Constants::kInfinity))
        std::cout << "ctest10.DistanceToIn(pct10mx,vx) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10mx,vmx);
    if (OutRange(dist,0))
        std::cout << "ctest10.DistanceToIn(pct10mx,vmx) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10mx,vy);
    if (OutRange(dist,77.78352))
        std::cout << "ctest10.DistanceToIn(pct10mx,vy) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10mx,vmy);
    if (OutRange(dist,0))
        std::cout << "ctest10.DistanceToIn(pct10mx,vmy) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10mx,vz);
    if (OutRange(dist,33.3333))
        std::cout << "ctest10.DistanceToIn(pct10mx,vz) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10mx,vmz);
    if (OutRange(dist,Constants::kInfinity))
        std::cout << "ctest10.DistanceToIn(pct10mx,vmz) = " << dist << std::endl;



    dist=ctest10.DistanceToIn(pct10e1,d1);
    if (OutRange(dist,Constants::kInfinity))
        std::cout << "ctest10.DistanceToIn(pct10e1,d1) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pct10e4,d1);
    if (OutRange(dist,Constants::kInfinity))
        std::cout << "ctest10.DistanceToIn(pct10e4,d1) = " << dist << std::endl;

    dist=ctest10.DistanceToIn(pt10s2,vt10d);
    // if (OutRange(dist,kInfinity))
        std::cout << "ctest10.DistanceToIn(pt10s2,vt10d) = " << dist << std::endl;

        double arad = 90.;

  Vec_t pct10phi1r( arad*std::cos(10.*VECGEOM_NAMESPACE::kPi/180.),  arad*std::sin(10*VECGEOM_NAMESPACE::kPi/180.), 0);
  Vec_t pct10phi2r( arad*std::cos(50.*VECGEOM_NAMESPACE::kPi/180.), -arad*std::sin(50*VECGEOM_NAMESPACE::kPi/180.), 0);

    dist = ctest10.DistanceToIn(pct10phi1r,vmy);
    // if (OutRange(dist,kInfinity))
        std::cout << "ctest10.DistanceToIn(pct10phi1r,vmy) = " << dist << std::endl;

    dist = ctest10.DistanceToIn(pct10phi2r,vx);
    // if (OutRange(dist,kInfinity))
        std::cout << "ctest10.DistanceToIn(pct10phi2r,vx) = " << dist << std::endl;


  Vec_t alex1P(49.840299921054168,-59.39735648688918,-20.893051766050633);
  Vec_t alex1V(0.6068108874999103,0.35615926907657169,0.71058505603651234);

  in = ctest10.Inside(alex1P);
  //std::cout << "ctest10.Inside(alex1P) = " <<in<< std::endl;
  //assert(in == vecgeom::EInside::kSurface);

  dist = ctest10.DistanceToIn(alex1P,alex1V);
  if (OutRange(dist,Constants::kInfinity))
  std::cout << "ctest10.DistanceToIn(alex1P,alex1V) = " << dist << std::endl;

  dist = ctest10.DistanceToOut(alex1P,alex1V,norm,convex);
  if (OutRange(dist,0))
  std::cout << "ctest10.DistanceToOut(alex1P,alex1V) = " << dist << std::endl;


  Vec_t alex2P(127.0075852717127, -514.1050841937065, 69.47104834263656);
  Vec_t alex2V(0.1277616879490939, 0.4093610465777845, 0.9033828007202369);

  in = ctest10.Inside(alex2P);
  //std::cout << "ctest10.Inside(alex2P) = " <<in<< std::endl;
  assert(in == vecgeom::EInside::kOutside);

  dist = ctest10.DistanceToIn(alex2P,alex2V);
  if (OutRange(dist,Constants::kInfinity))
  std::cout << "ctest10.DistanceToIn(alex2P,alex2V) = " << dist << std::endl;

  //Add Error of CMS, point on the Inner Surface going // to imaginary cone
  Cone_t  testc( "cCone", 261.9,270.4,1066.5,1068.7,274.75 , 0., 2*VECGEOM_NAMESPACE::kPi);
  Vec_t dir;
  dir=Vec_t(0.653315775,0.5050862758,0.5639737158);
  double x,y,z;
  x=-296.7662086;y=-809.1328836;z=13210.2270-(12800.5+274.75);
  Vec_t point=Vec_t(x,y,z);
  dist = testc.DistanceToOut(point,dir,norm,convex);
 Vec_t newp=point+dist*dir;
 //std::cout<<"CMS problem: DistOut has to be small="<<testc.DistanceToOut(point,dir,norm,convex)<<std::endl;
 //std::cout<<"CMS problem: DistInNew has to be kInfinity="<<testc.DistanceToIn(newp,dir)<<std::endl;
  assert(dist<0.05);
  dist=testc.DistanceToIn(newp,dir);
//  assert(ApproxEqual(dist,Constants::kInfinity));


    //Second test for Cons derived from testG4Cons1.cc
    pbigx=Vec_t(100,0,0);
    Vec_t pbigy(0,100,0),pbigz(0,0,100);
    Vec_t pbigmx(-100,0,0),pbigmy(0,-100,0),pbigmz(0,0,-100);

    Vec_t ponxside(50,0,0);

    double Dist;

    Cone_t  t1("Solid TubeLike #1",0,50,0,50,50,0,2.*VECGEOM_NAMESPACE::kPi);
    Cone_t  test10("test10",20.0, 80.0, 60.0, 140.0, 100.0,
                           0.17453292519943, 5.235987755983);

    Cone_t  test10a( "aCone", 20, 60, 80, 140, 100,
            10.*VECGEOM_NAMESPACE::kPi/180., 300.*VECGEOM_NAMESPACE::kPi/180. );


// Check name
    assert(t1.GetName()=="Solid TubeLike #1");

// Check Inside
    assert(t1.Inside(pzero)==vecgeom::EInside::kInside);
    assert(t1.Inside(pbigx)==vecgeom::EInside::kOutside);

// Check Surface Normal

    valid=t1.Normal(ponxside,normal);
    assert(ApproxEqual(normal,vx));

// SafetyFromInside(P)
    Dist=t1.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,50));

// DistanceToOut(P,V)
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
    assert(ApproxEqual(Dist,Constants::kInfinity));

    Dist=test10.DistanceToIn(Vec_t(19.218716967888,5.5354239324172,-100.0),
        Vec_t(-0.25644483536346,-0.073799216676426,0.96373737191901));
    std::cout<<"test10::DistToIn ="<<Dist<<std::endl;
    assert(ApproxEqual(Dist,Constants::kInfinity));
    Dist=test10.DistanceToOut(Vec_t(19.218716967888,5.5354239324172,-100.0),
        Vec_t(-0.25644483536346,-0.073799216676426,0.96373737191901),
                  norm,convex);
    //std::cout<<"test10::DistToOut ="<<Dist<<std::endl;
     assert(ApproxEqual(Dist,0));


   // CalculateExtent
   Vec_t minExtent,maxExtent;
   t1.Extent(minExtent,maxExtent);
   assert(ApproxEqual(minExtent,Vec_t(-50,-50,-50)));
   assert(ApproxEqual(maxExtent,Vec_t( 50, 50, 50)));
   ctest10.Extent(minExtent,maxExtent);
   assert(ApproxEqual(minExtent,Vec_t(-140,-140,-100)));
   assert(ApproxEqual(maxExtent,Vec_t( 140, 140, 100)));



    return true;
}

#ifdef VECGEOM_USOLIDS
struct USOLIDSCONSTANTS
{
  static constexpr double kInfinity = DBL_MAX;//UUtils::kInfinity;
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
      TestCons<USOLIDSCONSTANTS, UCons >();
      std::cout << "UCons passed\n";
      
      #else
      std::cerr << "VECGEOM_USOLIDS was not defined\n";
      return 2;
      #endif
    }
    else if( ! strcmp(argv[1], "--vecgeom") )
    {
       testingvecgeom = true;
       TestCons<VECGEOMCONSTANTS, vecgeom::SimpleCone >();
       std::cout<< "VecGeom Cone passed\n";
     
    }
    else
    {
      std::cerr << "need to give argument :--usolids or --vecgeom\n";     
      return 1;
    }


  return 0;
}
