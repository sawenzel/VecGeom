//
//
// TestTotus
//             Ensure asserts are compiled in

#undef NDEBUG
#include "base/Vector3D.h"
#include "ApproxEqual.h"
#include "base/Vector3D.h"
#include "volumes/Torus.h"

//#include "UVector3.hh"
//#include "UTorus.hh"

#include <cmath>
#include <cassert>
template <class Torus_t,class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >
bool testTorus()
{
   int i ;
   double Rtor = 100 ;
   double Rmax = Rtor*0.9 ;
   double Rmin = Rtor*0.1 ;
   double x; 
   double z;
 
   double Dist, dist, vol, volCheck;
   VUSolid::EnumInside side;
   Vec_t normal,norm;
   bool  convex,valid;

   double tolerance = 1e-9 ;

  
   
   Vec_t pzero(0,0,0);

   Vec_t pbigx(240,0,0),pbigy(0,240,0),pbigz(0,0,240);
   Vec_t pbigmx(-240,0,0),pbigmy(0,-240,0),pbigmz(0,0,-240);

   Vec_t ponrmax(190,0,0);
   Vec_t ponrmin(0,110,0);
   Vec_t ponrtor(0,100,0);
   Vec_t ponphi1(100/std::sqrt(2.),100/std::sqrt(2.),0) ;
   Vec_t ponphi2(-100/std::sqrt(2.),100/std::sqrt(2.),0) ;
   Vec_t ponphi12(120/std::sqrt(2.),120/std::sqrt(2.),0) ;
   Vec_t ponphi22(-120/std::sqrt(2.),120/std::sqrt(2.),0) ;
   Vec_t ponphi23(-120/std::sqrt(2.)+0.5,120/std::sqrt(2.),0) ;
    

   Vec_t vx(1,0,0),vy(0,1,0),vz(0,0,1);
   Vec_t vmx(-1,0,0),vmy(0,-1,0),vmz(0,0,-1);
   Vec_t vxy(1/std::sqrt(2.0),1/std::sqrt(2.0),0);
   Vec_t vmxy(-1/std::sqrt(2.0),1/std::sqrt(2.0),0);
   Vec_t vmxmy(-1/std::sqrt(2.0),-1/std::sqrt(2.0),0);
   Vec_t vxmy(1/std::sqrt(2.0),-1/std::sqrt(2.0),0);

   Vec_t pstart((Rtor+Rmax)/std::sqrt(2.0),(Rtor+Rmax)/std::sqrt(2.0),0) ;
   Vec_t vdirect(1/std::sqrt(2.0),-1/std::sqrt(2.0),0) ;
   
   Vec_t pother(110,0,0);
   vdirect = vdirect.Unit() ;
   Vec_t p1;
   Vec_t v1(1,0,0);
   v1 = v1.Unit();


// Check torus roots
   
   Torus_t t1("Solid Torus #1",0,Rmax,Rtor,0,vecgeom::kTwoPi);
   Torus_t t2("Hole cutted Torus #2",Rmin,Rmax,Rtor,0,vecgeom::kPi/2);//vecgeom::kPi/4,vecgeom::kPi/2.);
   Torus_t tn2("tn2",Rmin,Rmax,Rtor,vecgeom::kPi/2.,vecgeom::kPi/2.);
   Torus_t tn3("tn3",Rmin,Rmax,Rtor,vecgeom::kPi/2.,3*vecgeom::kPi/2.);
   Torus_t t3("Hole cutted Torus #3",4*Rmin,Rmax,Rtor,vecgeom::kPi/2.-vecgeom::kPi/24,vecgeom::kPi/12);
   Torus_t t4("Solid Torus #4",0,Rtor - 2.e3*tolerance,Rtor,0,vecgeom::kTwoPi);
   Torus_t t5("Solid cutted Torus #5",0,Rtor - 2.e3*tolerance,Rtor,vecgeom::kPi/4,vecgeom::kPi/2);
   Torus_t * aTub = new Torus_t("Ring1", 0, 100, 
                                         1000, 0, vecgeom::kTwoPi ); 
   Torus_t t6("t6",100, 150, 200,0,vecgeom::kPi/3);
   Torus_t* clad = new Torus_t("clad",0.,10.,100.,0.,vecgeom::kPi);    // external
   Torus_t* core =new Torus_t("core",0.,5,100,0.,vecgeom::kPi); // internal

   Torus_t* cmsEECool3 = new Torus_t("cmsEECool3",0,6,1640,0.09599,1.4312);
   Vec_t temp1=Vec_t (-0.646752, -0.762700, -0.000012);
   temp1=temp1/(temp1.Mag());

    std::cout<<"Dout="<<cmsEECool3->DistanceToOut(Vec_t(1230.993171, 1078.075896, -2.947036), temp1,norm,convex)<<std::endl;//: inf / inf / inf / 1.57856
   //std::cout<<"In="<<cmsEECool3->Inside(Vec_t(1230.993171, 1078.075896, -2.947036))<<" Rtor="<<std::sqrt(1230.99*1230.99+1078.07*1078.07)<<std::endl;
   assert(cmsEECool3->Inside(Vec_t(1230.993171, 1078.075896, -2.947036))==vecgeom::EInside:: kInside);
   Torus_t* cmsEECool30 = new Torus_t("cmsEECool30",0,0.7,164,0.09599,1.4312);
   temp1=Vec_t (0.958700, -0.284408, 0.002748);
   temp1=temp1/(temp1.Mag());
    std::cout.precision(20);
   std::cout<<"DIn="<<cmsEECool30->DistanceToIn(Vec_t(-652.200808, 359.142298, -1.826441), temp1)<<" temp1="<<temp1.Mag()<<std::endl; //689.407 / 689.407 / 689.407 / 687.745
   std::cout<<332858192.02587622404*(-3.0)<<" dif="<<-998574576.07762873173-332858192.02587622404*(-3.0)<<std::endl;
   // std::cout.precision(20);

   Vec_t p1t6( 60.73813233071262,  
                      -27.28494547459707,
                       37.47827539879173);

   Vec_t vt6( 0.3059312222729116,
		      0.8329513862588347,
		     -0.461083588265824);

   Vec_t p2t6(70.75950555416668,
		      -3.552713678800501e-15,
                      22.37458414788935       );

  // Check name
   assert(t1.GetName()=="Solid Torus #1");

  // Check cubic volume

     vol  = t1.Capacity();
   volCheck =vecgeom::kTwoPi*vecgeom::kPi*Rtor*(Rmax*Rmax);
   assert(ApproxEqual(vol,volCheck));
    
   vol  = t2.Capacity();
   volCheck = vecgeom::kPi/2.*vecgeom::kPi*Rtor*(Rmax*Rmax-Rmin*Rmin);
   
   assert(ApproxEqual(vol,volCheck));
   

    // Check Inside
    // std::cout<<t1.Inside(pzero)<<std::endl;
    assert(t1.Inside(pzero)==vecgeom::EInside:: kOutside);
    assert(t1.Inside(pbigx)==vecgeom::EInside:: kOutside);
    assert(t1.Inside(ponrmax)==vecgeom::EInside:: kSurface);
    assert(t2.Inside(ponrmin)==vecgeom::EInside:: kSurface);
    assert(t2.Inside(pbigx)==vecgeom::EInside:: kOutside);
    assert(t2.Inside(pbigy)==vecgeom::EInside:: kOutside);
    assert(t2.Inside(ponphi1)==vecgeom::EInside:: kOutside);
    assert(t2.Inside(ponphi2)==vecgeom::EInside:: kOutside);
    assert(t2.Inside(Vec_t(20,0,0))==vecgeom::EInside:: kSurface);
    assert(t2.Inside(Vec_t(0,20,0))==vecgeom::EInside:: kSurface);

    side=t6.Inside(p1t6);    
    // std::cout<<"t6.Inside(p1t6) = "<<side<<std::endl;
    side=t6.Inside(p2t6);    
    //std::cout<<"t6.Inside(p2t6) = "<<side<<std::endl;
     assert(t6.Inside(p2t6)==vecgeom::EInside:: kSurface);

// Check Surface Normal
   
    double p2=1./std::sqrt(2.); // ,p3=1./std::sqrt(3.);


    valid=t1.Normal(ponrmax,normal);
    assert(ApproxEqual(normal,vx)&&valid);
    valid =t1.Normal(Vec_t(0.,190.,0.),normal);
    assert(ApproxEqual(normal,vy));
    valid =tn2.Normal(Vec_t(0.,Rtor+Rmax,0.),normal);
    assert(ApproxEqual(normal,Vec_t(p2,p2,0.)));
    valid =tn2.Normal(Vec_t(0.,Rtor+Rmin,0.),normal);
    assert(ApproxEqual(normal,Vec_t(p2,-p2,0.)));
    valid =tn2.Normal(Vec_t(0.,Rtor-Rmin,0.),normal);
    assert(ApproxEqual(normal,Vec_t(p2,p2,0.)));
    valid =tn2.Normal(Vec_t(0.,Rtor-Rmax,0.),normal);
    assert(ApproxEqual(normal,Vec_t(p2,-p2,0.)));
    valid =tn3.Normal(Vec_t(Rtor,0.,Rmax),normal);
    assert(ApproxEqual(normal,Vec_t(0.,p2,p2)));
    valid =tn3.Normal(Vec_t(0.,Rtor,Rmax),normal);
    assert(ApproxEqual(normal,Vec_t(p2,0.,p2)));
    
    valid =t2.Normal(ponrmin,normal);
    // std::cout<<" normal="<<normal<<std::endl;
    assert(ApproxEqual(normal,vmxmy));
    valid =t2.Normal(Vec_t(20.,0.,0.),normal);
    assert(ApproxEqual(normal,vmy));
    valid =t2.Normal(Vec_t(0,20,0),normal);
    assert(ApproxEqual(normal,vmx));

// SafetyFromInside(P)
    Dist=t1.SafetyFromInside (ponrmin);
    assert(ApproxEqual(Dist,80));
    Dist=t1.SafetyFromInside (ponrmax);
    assert(ApproxEqual(Dist,0));
    // later: why it was introduced, while they are outside (see above)
    Dist=t2.SafetyFromInside (Vec_t(20,0,0));
    assert(ApproxEqual(Dist,0));
    Dist=t2.SafetyFromInside (Vec_t(0,20,0));
    assert(ApproxEqual(Dist,0));
    
// DistanceToOut(P,V)
    Dist=t1.DistanceToOut(ponrmax,vx,norm,convex);
    std::cout<<"t1.DistanceToOut(p,vx...) = "<<Dist<<" norm="<<norm<<std::endl ;
    assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vx));
   
    Dist=t1.DistanceToOut(Vec_t(130,0,0.0),vx,norm,convex);
    // Dist=t1.DistanceToOut(Vec_t(130,0.00001,0.00001),Vec_t(1,0.001,0.001).Unit(),norm,convex);
    std::cout<<"t1.DistanceToOut(ponphi1,vz,...) = "<<Dist<<" n="<<norm<<std::endl ;
    assert(ApproxEqual(Dist,60)&&ApproxEqual(norm,vx));
    
    Dist=t1.DistanceToOut(ponrmin,vy,norm,convex);
    assert(ApproxEqual(Dist,80)&&ApproxEqual(norm,vy));
    Dist=t1.DistanceToOut(ponrmin,vmy,norm,convex);
    assert(ApproxEqual(Dist,100));
  

    Dist=t1.DistanceToOut(Vec_t(100,0,0),(vz+Vec_t(0.00001,0.000001,0)).Unit(),norm,convex);
    std::cout<<"t1.DistanceToOut(100,0,0,vz,...) = "<<Dist<<" n="<<norm<<std::endl ;
    assert(ApproxEqual(Dist,90));



    Dist=t2.DistanceToOut(Vec_t(20,0,0),vmy,norm,convex);
    std::cout<<"Dist=t2.DistanceToOut(ponphi12,vy) = "<<Dist<<" n="<<norm<<std::endl;
    assert(ApproxEqual(Dist,0)&&convex&&ApproxEqual(norm,-vy));
    Dist=t2.DistanceToOut(Vec_t(0,20,0),vmx,norm,convex);
//    std::cout<<"Dist=t2.DistanceToOut(ponphi22,vmxmy) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,0)&&convex&&ApproxEqual(norm,-vx));

    Vec_t test(0.,0.,1);
    for (int i = 1; i < 5; i++)
      {
       Dist=t1.DistanceToOut(Vec_t(90,0,0),test,norm,convex);
       std::cout<<" i="<<i<<" Dist=t2.DistanceToOut(90,test) = "<<Dist<<" n="<<norm<<std::endl;
       test=test+Vec_t(0.00001,0.00001,0);
       test.Unit();
      }



//SafetyFromOutside(P)
    Dist=t1.SafetyFromOutside (pbigx);
    assert(ApproxEqual(Dist,50));
    Dist=t1.SafetyFromOutside (pbigmx);
    assert(ApproxEqual(Dist,50));
    Dist=t1.SafetyFromOutside (pbigy);
    assert(ApproxEqual(Dist,50));
    Dist=t1.SafetyFromOutside (pbigmy);
    assert(ApproxEqual(Dist,50));
    Dist=t1.SafetyFromOutside (pbigz);
//    std::cout<<"Dist=t1.SafetyFromOutside (pbigz) = "<<Dist<<std::endl;
//    assert(ApproxEqual(Dist,50));
    Dist=t1.SafetyFromOutside (pbigmz);
//    std::cout<<"Dist=t1.SafetyFromOutside (pbigmz) = "<<Dist<<std::endl;
//    assert(ApproxEqual(Dist,50));

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
    assert(ApproxEqual(Dist,vecgeom::kInfinity));
    Dist=t1.DistanceToIn(pbigmz,vz);
    assert(ApproxEqual(Dist,vecgeom::kInfinity));
    Dist=t1.DistanceToIn(pbigx,vxy);
    assert(ApproxEqual(Dist,vecgeom::kInfinity));
    Dist=t1.DistanceToIn(ponrmax,vx);
    std::cout<<"Dist=t1.DIN (p,v) = "<<Dist<<std::endl;
    //assert(ApproxEqual(Dist,vecgeom::kInfinity));
    Dist=t1.DistanceToIn(ponrmax,vmx);
   std::cout<<"Dist=t1.DIN (p,v) = "<<Dist<<std::endl;
   //assert(ApproxEqual(Dist,0));

    //Vec_t vnew(1,0,0) ;
    // vnew.rotateZ(pi/4-5*1e-9) ;    // old test: check pzero with vxy
    // Dist=t2.DistanceToIn(pzero,vnew);
    //assert(ApproxEqual(Dist,vecgeom::kInfinity));
     std::cout<<"Dist=t2.DIN (p,v) = "<<Dist<<std::endl;
    Dist=t2.DistanceToIn(pzero,vy);
    assert(ApproxEqual(Dist,10));
    Dist=t2.DistanceToIn(ponphi12,vy);
    assert(ApproxEqual(Dist,0));
    Dist=t2.DistanceToIn(ponphi12,vmy);
    assert(ApproxEqual(Dist,vecgeom::kInfinity));
    Dist=t2.DistanceToIn(ponphi1,vy);
//    std::cout<<"Dist=t2.DistanceToIn(ponphi1,vy) = "<<Dist<<std::endl;  // about 13
    Dist=t2.DistanceToIn(ponrmin,vy);
    assert(ApproxEqual(Dist,0));
    Dist=t2.DistanceToIn(ponrmin,vmy);
    assert(ApproxEqual(Dist,20));

    Dist=t3.DistanceToIn(ponrtor,vy);
    assert(ApproxEqual(Dist,40));
    Dist=t3.DistanceToIn(ponrtor,vmy);
    assert(ApproxEqual(Dist,40));
    Dist=t3.DistanceToIn(ponrtor,vz);
    assert(ApproxEqual(Dist,40));
    Dist=t3.DistanceToIn(ponrtor,vmz);
    assert(ApproxEqual(Dist,40));
    Dist=t3.DistanceToIn(ponrtor,vx);
    assert(ApproxEqual(Dist,vecgeom::kInfinity));
    Dist=t3.DistanceToIn(ponrtor,vmx);
    assert(ApproxEqual(Dist,vecgeom::kInfinity));

    Dist=t6.DistanceToIn(p1t6,vt6);
    // std::cout<<"t6.DistanceToIn(p1t6,vt6) = "<<Dist<<std::endl;

    // Bug 810

    Vec_t pTmp(0.,0.,0.);

    dist = clad->DistanceToIn(pTmp,vy);   
    pTmp += dist*vy;
    std::cout<<"pTmpX = "<<pTmp.x()<<";  pTmpY = "<<pTmp.y()<<";  pTmpZ = "<<pTmp.z()<<std::endl;
    side=core->Inside(pTmp);    
    std::cout<<"core->Inside(pTmp) = "<<side<<std::endl;
    side=clad->Inside(pTmp);    
    std::cout<<"clad->Inside(pTmp) = "<<side<<std::endl;

    dist = core->DistanceToIn(pTmp,vy);   
    pTmp += dist*vy;
    std::cout<<"pTmpX = "<<pTmp.x()<<";  pTmpY = "<<pTmp.y()<<";  pTmpZ = "<<pTmp.z()<<std::endl;
    side=core->Inside(pTmp);    
    std::cout<<"core->Inside(pTmp) = "<<side<<std::endl;
    side=clad->Inside(pTmp);    
    std::cout<<"clad->Inside(pTmp) = "<<side<<std::endl;

    dist = core->DistanceToOut(pTmp,vy,norm,convex);   
    pTmp += dist*vy;
    std::cout<<"pTmpX = "<<pTmp.x()<<";  pTmpY = "<<pTmp.y()<<";  pTmpZ = "<<pTmp.z()<<std::endl;
    side=core->Inside(pTmp);    
    std::cout<<"core->Inside(pTmp) = "<<side<<std::endl;
    side=clad->Inside(pTmp);    
    std::cout<<"clad->Inside(pTmp) = "<<side<<std::endl;

    dist = clad->DistanceToOut(pTmp,vy,norm,convex);   
    pTmp += dist*vy;
    std::cout<<"pTmpX = "<<pTmp.x()<<";  pTmpY = "<<pTmp.y()<<";  pTmpZ = "<<pTmp.z()<<std::endl;
    side=core->Inside(pTmp);    
    std::cout<<"core->Inside(pTmp) = "<<side<<std::endl;
    side=clad->Inside(pTmp);    
    std::cout<<"clad->Inside(pTmp) = "<<side<<std::endl;

// Check for Distance to In ( start from an external point )

   for ( i=0; i<12; i++ ) 
   {
     x = -1200;
     z = double(i)/10;
     p1 = Vec_t(x,0,z);
     //     std::cout << p1 << " - " << v1 << std::endl;

     Dist = aTub->DistanceToIn (p1,v1) ;
     //     std::cout << "Distance to in dir: " << Dist ;

     Dist = aTub->DistanceToOut (p1,v1) ;
     //     std::cout << "   Distance to out dir: " << Dist << std::endl ;

     // std::cout << "Distance to in : " << aTub->DistanceToIn (p1);
     //  std::cout << "   Distance to out : " << aTub->DistanceToOut (p1)
     //    << std::endl;
     //     std::cout << "   Inside : " << aTub->Inside (p1);
     //     std::cout << std::endl;
   }
    
    
// CalculateExtent

  
    return true;
}

int main()
{
#ifdef NDEBUG
    G4Exception("FAIL: *** Assertions must be compiled in! ***");
#endif
    assert(testTorus<vecgeom::SimpleTorus>());
    return 0;
}

