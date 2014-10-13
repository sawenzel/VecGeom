//
//
// TestBox
//             Ensure asserts are compiled in

#undef NDEBUG
#include "base/Global.h"
#include "base/Vector3D.h"
#include "volumes/Box.h"
#include "volumes/Sphere.h"
#include "ApproxEqual.h"
#ifdef VECGEOM_USOLIDS
#include "UBox.hh"
#include "USphere.hh"
#include "UVector3.hh"
#endif

#include <cassert>
#include <cmath>
#include <iomanip> 

#define PI 3.14159265358979323846


template <class Sphere_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >
bool TestSphere() {
    
    int verbose=0;
    double fRmin=0, fRmax=3, fSPhi=0, fDPhi=2*PI, fSTheta=0, fDTheta=PI;
    double fR=fRmax;
    
    double Dist;
    Vec_t pzero(0,0,0);
    Vec_t pbigx(100,0,0),pbigy(0,100,0),pbigz(0,0,100);
    Vec_t pbigmx(-100,0,0),pbigmy(0,-100,0),pbigmz(0,0,-100);
    
    Sphere_t b1("Solid VecGeomSphere #1",fRmin, fRmax, fSPhi, fDPhi, fSTheta, fDTheta);
    Sphere_t b2("Solid VecGeomSphere #2",fRmin, fRmax, PI/6, PI/2, fSTheta, fDTheta);
    Sphere_t b3("Solid VecGeomSphere #3",1, fRmax, PI/6, PI/2, fSTheta, fDTheta);
    
    // Check name
    assert(b1.GetName()=="Solid VecGeomSphere #1");
    assert(b2.GetName()=="Solid VecGeomSphere #2");

    
   // Check cubic volume
    assert(b1.Capacity() == ((4 * PI / 3) * fR * fR * fR));    
    assert(b2.Capacity() == (((4 * PI / 3) * fR * fR * fR)/4));    
    //std::cout<<"Capacity of B3 : "<<b3.Capacity()<<std::endl;
    assert(ApproxEqual(b3.Capacity(),27.22713633111));
    
    //std::cout<<"Capacity : "<<b2.Capacity()<<std::endl;
    // Check Surface area
    assert(b1.SurfaceArea() == ((4 * PI) * fR * fR));   
    assert(b2.SurfaceArea() == ((((4 * PI) * fR * fR)/4)+(PI*fR*fR)));
    assert(ApproxEqual(b2.SurfaceArea(),56.5486677646));
    //std::cout<<std::setprecision(12)<<"SurfaceArea of B3 : "<< b3.SurfaceArea() << std::endl;
    //std::cout<<"Not : "<< b1.SurfaceArea()/4 << std::endl;
    
    Vec_t minExtent,maxExtent;
    b1.Extent(minExtent,maxExtent);
    assert(ApproxEqual(minExtent,Vec_t(-fR,-fR,-fR)));
    assert(ApproxEqual(maxExtent,Vec_t( fR, fR, fR)));
    b2.Extent(minExtent,maxExtent);
    assert(ApproxEqual(minExtent,Vec_t(-fR,-fR,-fR)));
    assert(ApproxEqual(maxExtent,Vec_t( fR, fR, fR)));
    //assert(ApproxEqual(minExtent,Vec_t(-6,-6,-6)));
    //assert(ApproxEqual(maxExtent,Vec_t( 6, 6, 6)));
    
    bool valid;
    Vec_t ponx(8,0.,0.), pony(0.,8.,0.), ponz(0.,0.,10.);
    Vec_t ponmx(-8,0.,0.), ponmy(0.,-8.,0.), ponmz(0.,0.,-10.);
    Vec_t normal;
    
     
    //______________________________________________________________________________________
    
    //Sphere for testing Real Important functions like Inside, Saftey , DistanceToIn, DistanceToOut
    
    //Tests considering fullPhiSphere and fullThetaSphere
    
    Sphere_t b4("Solid VecGeomSphere #4",6, 8, fSPhi, fDPhi, fSTheta, fDTheta);
    
    //Completely Inside Point
    Vec_t pointI(0.337317535387103,6.265557026812151,3.103064940359008); //Point at the distance of 7 from center of sphere i.e. completely inside point
    assert(b4.Inside(pointI)==vecgeom::EInside::kInside); //Should pass
    
    //Completely Outside Point
    Vec_t pointO(0.162722224118177,-8.624868144449298,-2.566158796458579); //Point at the distance of 9 from center of sphere i.e. completely outside point
    assert(b4.Inside(pointO)==vecgeom::EInside::kOutside); //Should pass
    
    //Point very very very close to fRmax should be considered on the surface
    Vec_t pointOS(-2.393620784732096,-4.753290873255242,5.973006396542139); //Point at the distance of 8.0000000000004 from center of sphere i.e. Surface point
    assert(b4.Inside(pointOS)==vecgeom::EInside::kSurface); //Should pass
    
    //Point very very very close to fRmin should be considered on the surface
    Vec_t pointIS(-4.231374871926247,-4.183095200116632,-0.772775025466741); //Point at the distance of 5.9999999999994 from center of sphere i.e. Surface point
    assert(b4.Inside(pointIS)==vecgeom::EInside::kSurface); //Should pass
    
    //Point Just inside the inner tolerance of fRmax should be considered as inside point
    Vec_t pointJIO(5.644948390009752,-5.435958227995553,1.607767309536356); //Point at the distance of 7.999999994 from center of sphere i.e. completely inside point considering tolerance limit is 1e-9
    assert(b4.Inside(pointJIO)==vecgeom::EInside::kInside); //Should pass
    
    //Point Just outside the inner tolerance of fRmin should be considered as inside point
    Vec_t pointJOI(0.101529841379922,-5.993375366532538,0.262951431162379); //Point at the distance of 6.000000004 from center of sphere i.e. completely inside point considering tolerance limit is 1e-9
    assert(b4.Inside(pointJOI)==vecgeom::EInside::kInside); //Should pass
    
    //Testing Safety Functions
    Dist=b4.SafetyFromOutside(pbigx);
    assert(ApproxEqual(Dist,100-8));
    
    //SafetyFromInside return 0 if point is outside 
    Dist=b4.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,0));
    
        
    Dist=b4.SafetyFromInside(pointO); //using outside point for b4
    assert(ApproxEqual(Dist,0));
    
    Vec_t pointO2(9,0,0);
    Dist=b4.SafetyFromInside(pointO2); //using outside point for b4
    assert(ApproxEqual(Dist,0));
    
    Vec_t pointO3(0,9,0);
    Dist=b4.SafetyFromInside(pointO3); //using outside point for b4
    assert(ApproxEqual(Dist,0));
    
    Vec_t pointO4(0,0,9);
    Dist=b4.SafetyFromInside(pointO4); //using outside point for b4
    assert(ApproxEqual(Dist,0));
    
    
    Vec_t pointBWRminRmaxXI(6.5,0,0);
    Dist=b4.SafetyFromInside(pointBWRminRmaxXI);
    assert(ApproxEqual(Dist,0.5));
    
    Vec_t pointBWRminRmaxYI(0,6.5,0);
    Dist=b4.SafetyFromInside(pointBWRminRmaxYI); 
    assert(ApproxEqual(Dist,0.5));
    
    Vec_t pointBWRminRmaxZI(0,0,6.5);
    Dist=b4.SafetyFromInside(pointBWRminRmaxZI); 
    assert(ApproxEqual(Dist,0.5));
    
    Vec_t pointBWRminRmaxXO(7.5,0,0);
    Dist=b4.SafetyFromInside(pointBWRminRmaxXO); 
    assert(ApproxEqual(Dist,0.5));
    
    Vec_t pointBWRminRmaxYO(0,7.5,0);
    Dist=b4.SafetyFromInside(pointBWRminRmaxYO); 
    assert(ApproxEqual(Dist,0.5));
    
    Vec_t pointBWRminRmaxZO(0,0,7.5);
    Dist=b4.SafetyFromInside(pointBWRminRmaxZO); 
    assert(ApproxEqual(Dist,0.5));
    
    //For Inside point SafetyFromOutside returns 0
    Dist=b4.SafetyFromOutside(pointBWRminRmaxXO); 
    assert(ApproxEqual(Dist,0)); //should pass
    Dist=b4.SafetyFromOutside(pointBWRminRmaxYO); 
    assert(ApproxEqual(Dist,0)); //should pass
    Dist=b4.SafetyFromOutside(pointBWRminRmaxZO); 
    assert(ApproxEqual(Dist,0)); //should pass
    
    Vec_t genPointBWRminRmax(3.796560684305335,-6.207283535497058,2.519078815824183); //Point at distance of 7.7 from center. i.e. inside point, SafetyFromOutside should return 0
    Dist=b4.SafetyFromOutside(genPointBWRminRmax); 
    assert(ApproxEqual(Dist,0)); //should pass
    
    valid = b4.Normal(ponx,normal);
    assert(ApproxEqual(normal,Vec_t(1,0,0)));
    valid = b4.Normal(pony,normal);
    assert(ApproxEqual(normal,Vec_t(0,1,0)));
    valid = b4.Normal(ponz,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,1)));
    valid = b4.Normal(ponmx,normal);
    assert(ApproxEqual(normal,Vec_t(-1,0,0)));
    valid = b4.Normal(ponmy,normal);
    assert(ApproxEqual(normal,Vec_t(0,-1,0)));
    valid = b4.Normal(ponmz,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,-1)));
    
    
    //________________________________________________________________________________________________________________
    //Sphere for testing Real Important functions like Inside, Saftey , DistanceToIn, DistanceToOut
    //Considering !FullPhiSphere but FullThetaSphere
    Sphere_t b5("Solid VecGeomSphere #5",6, 8, fSPhi, fDPhi/2, fSTheta, fDTheta);
    //Completely Inside section of a sphere from 0-PI, but radially inside
    Vec_t pointISec_0_PI(-6.354024677915919,2.724956501738179,-1.095893451057325); //Point at the distance of 7 from center of sphere and in 0-PI Phi range i.e. completely inside point
    assert(b5.Inside(pointISec_0_PI)==vecgeom::EInside::kInside); //Should pass
    
    //Completely Inside section PI-2PI but completely outside section 0-PI, but radially inside
    //Vec_t pointISec_PI_2PI(4.686901690014317,-1.144935913500135,5.071693435344703); //Point at the distance of 7 from center of sphere and in PI-2PI Phi range i.e. completely outside 
    Vec_t pointISec_PI_2PI(-2.523183636323228,-3.188902592924771,-5.697757856405303);
    assert(b5.Inside(pointISec_PI_2PI)==vecgeom::EInside::kOutside); //Should pass
    //std::cout<<"PI - 2PI point line 115 : "<<b5.Inside(pointISec_PI_2PI)<<std::endl;
    
    //Point very very very close to end Phi angle (fSPhi+fDPhi) should be considered on the surface
    Vec_t pointISurface_Sec_0_PI(-6.975035234148935,0.000000000583791,-0.590663594934467); //Point at the distance of 7 from center of sphere and very very very close to PI in the inside region in 0-PI Phi range
                                                                                        //i.e.  radially inside for fullPhiSphere but for spherical section of 0-PI it should be on surface
    assert(b5.Inside(pointISurface_Sec_0_PI)==vecgeom::EInside::kSurface); //Should pass
    
    
    Vec_t pointOSurface_Sec_0_PI(-6.859517189252706,-0.000000002746520,1.395357993615493); //Point at the distance of 7 from center of sphere and very very very close to PI in the outside region in 0-PI Phi range
                                                                                        //i.e.  radially inside for fullPhiSphere but for spherical section of 0-PI it should be on surface
    assert(b5.Inside(pointOSurface_Sec_0_PI)==vecgeom::EInside::kSurface); //Should pass
    
    //Considering !FullPhiSphere but FullThetaSphere
    Sphere_t b6("Solid VecGeomSphere #6",6, 8, PI/6, PI/3, fSTheta, fDTheta); //Spherical section from 30-60 degree
    Vec_t pointI_30_60(5.343263338886987,3.536603816131425,-2.818150162612158);//Point at the distance of 7 from center of sphere and in 30-60 Phi range i.e. completely inside
    assert(b6.Inside(pointI_30_60)==vecgeom::EInside::kInside); //Should pass
    
    Vec_t pointO_30_60(6.043467293054153,0.461937628581358,3.501873313683026); ////Point at the distance of 7 from center of sphere and in 0-30 Phi range i.e. completely outside
    assert(b6.Inside(pointO_30_60)==vecgeom::EInside::kOutside); //Should pass
    
    Vec_t pointISurface_Sec_30_60(0.139868946641446,0.080753374000628,6.998136578429498);
    assert(b6.Inside(pointISurface_Sec_30_60)==vecgeom::EInside::kSurface); //Should pass
    
    Vec_t pointOSurface_Sec_30_60(4.589551141500899,2.649778586424563,4.573258549707597); //Point at the distance of 7 from center of sphere and very very very close to 30 in the outside region in 30-60 Phi range
    assert(b6.Inside(pointOSurface_Sec_30_60)==vecgeom::EInside::kSurface); //Should pass
            
    Sphere_t b8("Solid VecGeomSphere #8",6, 8, PI/6, PI/6, fSTheta, fDTheta); //Spherical section from 30-60 degree in PHI and 0-180 in THETA
    Vec_t pointOSafety_phi_30_60_theta_0_180(0,5,0);
    Dist=b8.SafetyFromOutside(pointOSafety_phi_30_60_theta_0_180);
    assert(ApproxEqual(Dist,2.50000));
    
    Dist=b8.SafetyFromOutside(pbigx);
    assert(ApproxEqual(Dist,92));
    Dist=b8.SafetyFromOutside(pbigy);
    assert(ApproxEqual(Dist,92));
    Dist=b8.SafetyFromOutside(pbigz);
    assert(ApproxEqual(Dist,92));
    Dist=b8.SafetyFromOutside(pbigmx);
    assert(ApproxEqual(Dist,92));
    Dist=b8.SafetyFromOutside(pbigmy);
    assert(ApproxEqual(Dist,92));
    Dist=b8.SafetyFromOutside(pbigmz);
    assert(ApproxEqual(Dist,92));
    
    //For Outside point SafetyFromInside return zero. Following test should pass
    Dist=b8.SafetyFromInside(pbigx);
    assert(ApproxEqual(Dist,0));
    Dist=b8.SafetyFromInside(pbigy);
    assert(ApproxEqual(Dist,0));
    Dist=b8.SafetyFromInside(pbigz);
    assert(ApproxEqual(Dist,0));
    Dist=b8.SafetyFromInside(pbigmx);
    assert(ApproxEqual(Dist,0));
    Dist=b8.SafetyFromInside(pbigmy);
    assert(ApproxEqual(Dist,0));
    Dist=b8.SafetyFromInside(pbigmz);
    assert(ApproxEqual(Dist,0));
    
    //Point between Rmin and Rmax but outside Phi Range
    Vec_t point_phi_60_90(0.041433182037376,3.902131470711286,-6.637895244481554); //Point at a distance of 7.7 from center and in 60-90 phi range. i.e completely outside point. DistanceFromInside shoudl return zero
    Dist=b8.SafetyFromInside(point_phi_60_90);
    assert(ApproxEqual(Dist,0));
    
    Dist=b8.SafetyFromOutside(point_phi_60_90);
    //std::cout<<std::setprecision(20)<<"SafetyFromOutside for point outside phi range line 304 : "<<b8.SafetyFromOutside(point_phi_60_90)<<std::endl;
    assert(ApproxEqual(Dist,1.91518354715165));
    
    //Checking NORMAL FUNCTION
    //Sphere in 60-120 Phi range
    Sphere_t b10("Solid VecGeomSphere #10",6, 8, PI/3, PI/3, fSTheta, fDTheta);
    Vec_t point_phi_60_120_rad_9(2.821923096621196,6.085843652625694,5.999938089059868);
    valid = b10.Normal(point_phi_60_120_rad_9,normal);
    assert(ApproxEqual(normal,Vec_t(0.866025 , -0.5 , 0))); //Verified with Geant4
    //________________________________________________________________________________________________________________
    
    //Tests Considering !FullPhiSphere and !FullThetaSphere
    Sphere_t b7("Solid VecGeomSphere #7",6, 8, PI/6, PI/6, 0., PI/6); //Spherical section from 30-60 degree in PHI and 0-30 in THETA
    
    Vec_t pointI_phi_30_60_theta_0_30(2.051950506806283,1.301790637821730,6.564666042754737); //Point at the distance of 7 from center of sphere and in 30-60 Phi range and 0-30 in Theta
    assert(b7.Inside(pointI_phi_30_60_theta_0_30)==vecgeom::EInside::kInside); //Should pass
    
    Vec_t pointO_phi_30_60_theta_31_45(3.038887886090341,3.240358815458723,5.409735221140894); //Point at the distance of 7 from center of sphere and in 30-60 Phi range and 31-45 in Theta; completely outside in terms of theta
    assert(b7.Inside(pointO_phi_30_60_theta_31_45)==vecgeom::EInside::kOutside); //Should pass
                                                                                                                
    Vec_t pointOSurface_phi_30_60_theta_0_30(2.421058285775014,2.527543625149363,6.062177826478448); //Point at the distance of 7 from center of sphere and very very very close to 30 in the outside region in 0-30 theta range
                                                                                        //i.e.  radially inside, also inside in terms of phi (30-60) but for theta section of 0-30 it should be on surface
    assert(b7.Inside(pointOSurface_phi_30_60_theta_0_30)==vecgeom::EInside::kSurface); //Should pass
    
    Vec_t pointISurface_phi_30_60_theta_0_30(2.983684745081379,1.829651698111631,6.062177826950138); //Point at the distance of 7 from center of sphere and very very very close to 30 in the inside region in 0-30 theta range
                                                                                        //i.e.  radially inside, also inside in terms of phi (30-60) but for theta section of 0-30 it should be on surface
    assert(b7.Inside(pointISurface_phi_30_60_theta_0_30)==vecgeom::EInside::kSurface); //Should pass
    
    
    Sphere_t b9("Solid VecGeomSphere #9",6, 8, PI/6, PI/6, PI/6, PI/6);
    Dist=b9.SafetyFromOutside(pbigx);
    assert(ApproxEqual(Dist,92));
    Dist=b9.SafetyFromOutside(pbigy);
    assert(ApproxEqual(Dist,92));
    Dist=b9.SafetyFromOutside(pbigz);
    assert(ApproxEqual(Dist,92));
    Dist=b9.SafetyFromOutside(pbigmx);
    assert(ApproxEqual(Dist,92));
    Dist=b9.SafetyFromOutside(pbigmy);
    assert(ApproxEqual(Dist,92));
    Dist=b9.SafetyFromOutside(pbigmz);
    assert(ApproxEqual(Dist,92));
    
    //For Outside point SafetyFromInside return zero. Following test should pass
    Dist=b9.SafetyFromInside(pbigx);
    assert(ApproxEqual(Dist,0));
    Dist=b9.SafetyFromInside(pbigy);
    assert(ApproxEqual(Dist,0));
    Dist=b9.SafetyFromInside(pbigz);
    assert(ApproxEqual(Dist,0));
    Dist=b9.SafetyFromInside(pbigmx);
    assert(ApproxEqual(Dist,0));
    Dist=b9.SafetyFromInside(pbigmy);
    assert(ApproxEqual(Dist,0));
    Dist=b9.SafetyFromInside(pbigmz);
    assert(ApproxEqual(Dist,0));
    
    //For Completely inside point it should Dist should be zero. Using data of b7 sphere.
    Dist=b7.SafetyFromOutside(pointI_phi_30_60_theta_0_30);
    assert(ApproxEqual(Dist,0));
    
    Dist=b7.SafetyFromOutside(pointO_phi_30_60_theta_31_45);
     assert(ApproxEqual(Dist,1.142348737370068));
    
    valid = b7.Normal(pointO_phi_30_60_theta_31_45,normal);
    assert(ApproxEqual(normal,Vec_t(-0.434127 , -0.462908 , -0.772819))); //Verified with Geant4
    
    
    
    Vec_t norm;
    bool convex;
    convex = true;
    //JUST A TESTING SPHERE FOR DEBUGGING
    //Sphere_t test("Solid VecGeomSphere #test",6, 8, 0, 2*PI, 0., PI);
    Sphere_t test("Solid VecGeomSphere #test",10, 20, 0.2, 3.6, 0.2, 0.5);
    Vec_t testPoint(-6.083316, 7.548949, 11.675289);
    double mag=std::sqrt(testPoint[0]*testPoint[0] + testPoint[1]*testPoint[1] + testPoint[2]*testPoint[2]);
    std::cout<<"Magnitude of Point : "<< mag <<std::endl;
    Vec_t testDir(0.478757, -0.602168, 0.638894);
    double pdotV = testPoint[0]*testDir[0]+testPoint[1]*testDir[1]+testPoint[2]*testDir[2];
    if(pdotV < 0)
        std::cout<<"Direction of Point : IN"<<std::endl;
    else
        std::cout<<"Direction of Point : Out"<<std::endl;
    
    std::cout<<"Theta is : " <<std::acos(testDir[2]/ mag)<<std::endl;
    std::cout<<"PHI is : "<< std::atan(testDir[1]/testDir[0])<<std::endl;
    Dist=test.DistanceToOut(testPoint,testDir,norm,convex);
    std::cout<<"Distance : "<<Dist<<std::endl;
    
    /*
    std::cout<<"---- Point inside inner radius ----"<<std::endl;
    Sphere_t test2("Solid VecGeomSphere #test2",10, 20, 0, 2*PI, 0., PI);
    Vec_t pointInsideInnerR(8,0,0);
    Vec_t dirPointInsideInner(-1,0,0);
    Dist=test2.DistanceToOut(pointInsideInnerR,dirPointInsideInner,norm,convex);
    std::cout<<"Distance InsideInnerR: "<<Dist<<std::endl;
    */
    
    return true;
}

int main() {
    
#ifdef VECGEOM_USOLIDS
  assert(TestSphere<USphere>());
  std::cout << "USphere passed\n";
#endif
  std::cout<<"-------------------------------------------------------------------------------------------------"<<std::endl;
  assert(TestSphere<vecgeom::SimpleSphere>());
  std::cout << "VecGeomSphere passed\n";
  return 0;
}

