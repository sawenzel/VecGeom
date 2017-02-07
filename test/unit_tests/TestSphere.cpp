//
//
// TestSphere
//             Ensure asserts are compiled in


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
#include <cmath>
#include <iomanip> 


#undef NDEBUG
#include <cassert>

#define PI 3.14159265358979323846
#define deg PI/180.


template <class Sphere_t, class Vec_t = vecgeom::Vector3D<vecgeom::Precision> >
bool TestSphere() {
    
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
    Vec_t ponx(8,0.,0.), pony(0.,8.,0.), ponz(0.,0.,8.);
    Vec_t ponmx(-8,0.,0.), ponmy(0.,-8.,0.), ponmz(0.,0.,-8.);
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
    //assert(b4.Inside(pointJIO)==vecgeom::EInside::kInside); //Should pass
    
    //Point Just outside the inner tolerance of fRmin should be considered as inside point
    Vec_t pointJOI(0.101529841379922,-5.993375366532538,0.262951431162379); //Point at the distance of 6.000000004 from center of sphere i.e. completely inside point considering tolerance limit is 1e-9
    //assert(b4.Inside(pointJOI)==vecgeom::EInside::kInside); //Should pass
    
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
    assert(valid);
    valid = b4.Normal(pony,normal);
    assert(ApproxEqual(normal,Vec_t(0,1,0)));
    assert(valid);
    valid = b4.Normal(ponz,normal);
    assert(valid);
    assert(ApproxEqual(normal,Vec_t(0,0,1)));
    valid = b4.Normal(ponmx,normal);
    assert(valid);
    assert(ApproxEqual(normal,Vec_t(-1,0,0)));
    valid = b4.Normal(ponmy,normal);
    assert(valid);
    assert(ApproxEqual(normal,Vec_t(0,-1,0)));
    valid = b4.Normal(ponmz,normal);
    assert(ApproxEqual(normal,Vec_t(0,0,-1)));
    assert(valid);
    
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
    //assert(b5.Inside(pointISurface_Sec_0_PI)==vecgeom::EInside::kSurface); //Should pass
    
    
    Vec_t pointOSurface_Sec_0_PI(-6.859517189252706,-0.000000002746520,1.395357993615493); //Point at the distance of 7 from center of sphere and very very very close to PI in the outside region in 0-PI Phi range
                                                                                        //i.e.  radially inside for fullPhiSphere but for spherical section of 0-PI it should be on surface
    //assert(b5.Inside(pointOSurface_Sec_0_PI)==vecgeom::EInside::kSurface); //Should pass
    
    //Considering !FullPhiSphere but FullThetaSphere
    Sphere_t b6("Solid VecGeomSphere #6",6, 8, PI/6, PI/3, fSTheta, fDTheta); //Spherical section from 30-60 degree
    Vec_t pointI_30_60(5.343263338886987,3.536603816131425,-2.818150162612158);//Point at the distance of 7 from center of sphere and in 30-60 Phi range i.e. completely inside
    assert(b6.Inside(pointI_30_60)==vecgeom::EInside::kInside); //Should pass
    
    Vec_t pointO_30_60(6.043467293054153,0.461937628581358,3.501873313683026); ////Point at the distance of 7 from center of sphere and in 0-30 Phi range i.e. completely outside
    assert(b6.Inside(pointO_30_60)==vecgeom::EInside::kOutside); //Should pass
    
    Vec_t pointISurface_Sec_30_60(0.139868946641446,0.080753374000628,6.998136578429498);
    //assert(b6.Inside(pointISurface_Sec_30_60)==vecgeom::EInside::kSurface); //Should pass
    
    Vec_t pointOSurface_Sec_30_60(4.589551141500899,2.649778586424563,4.573258549707597); //Point at the distance of 7 from center of sphere and very very very close to 30 in the outside region in 30-60 Phi range
    //assert(b6.Inside(pointOSurface_Sec_30_60)==vecgeom::EInside::kSurface); //Should pass
            
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
   
    //________________________________________________________________________________________________________________ 
    //
    //Add unit test from Geant4
    //


    Vec_t px(30,0,0),py(0,30,0),pz(0,0,30);
    Vec_t pmx(-30,0,0),pmy(0,-30,0),pmz(0,0,-30);
    Vec_t ponrmin1(45,0,0),ponrmax1(50,0,0),ponzmax(0,0,50),
          ponrmin2(45/std::sqrt(2.),45/std::sqrt(2.),0),
          ponrmin3(0,0,-45),ponrminJ(0,0,-300),ponrmaxJ(0,0,-500),
          ponrmax2(50/std::sqrt(2.),50/std::sqrt(2.),0);
    Vec_t ponphi1(48/std::sqrt(2.),-48/std::sqrt(2.),0),
          ponphi2(48/std::sqrt(2.),48/std::sqrt(2.),0),
          pInPhi(48*0.866,-24,0),
          pOverPhi(-48/std::sqrt(2.),48/std::sqrt(2.),0);
    Vec_t pontheta1(0,48*std::sin(PI/4),48*std::cos(PI/4)),
          pontheta2(0,48*std::sin(PI/4),-48*std::cos(PI/4));

    Vec_t ptestphi1(-100,-45/std::sqrt(2.),0),
          ptestphi2(-100,45/std::sqrt(2.),0);

    Vec_t ptesttheta1(0,48/std::sqrt(2.),100),
          ptesttheta2(0,48/std::sqrt(2.),-100);

    // Directions

    Vec_t vx(1,0,0),vy(0,1,0),vz(0,0,1);
    Vec_t vmx(-1,0,0),vmy(0,-1,0),vmz(0,0,-1);
    Vec_t vxy(1/std::sqrt(2.),1/std::sqrt(2.),0),vmxmy(-1/std::sqrt(2.),-1/std::sqrt(2.),0);
    Vec_t vxmy(1/std::sqrt(2.),-1/std::sqrt(2.),0),vmxy(-1/std::sqrt(2.),1/std::sqrt(2.),0);
    Vec_t vxmz(1/std::sqrt(2.),0.,-1/std::sqrt(2.)),vymz(0.,1/std::sqrt(2.),-1/std::sqrt(2.));
    Vec_t vmxmz(-1/std::sqrt(2.),0.,-1/std::sqrt(2.)),vmxz(-1/std::sqrt(2.),0.,1/std::sqrt(2.));

    Vec_t v345exit1(-0.8,0.6,0),v345exit2(0.8,0.6,0),v345exit3(0.6,0.8,0);

    Vec_t pRand, vRand, norm;
    bool convex;
    VUSolid::EnumInside  inside;
    Sphere_t s1("Solid Sphere_t",0,50,0,2*PI,0,PI);
    Sphere_t sn1("sn1",0,50,PI*0.5,3.*PI*0.5,0,PI);

    // Theta sections

    Sphere_t sn11("sn11",0,50,0,2*PI,0.,0.5*PI);
    Sphere_t sn12("sn12",0,50,0,2*PI,0.,0.25*PI);
    Sphere_t sn13("sn13",0,50,0,2*PI,0.75*PI,0.25*PI);
    Sphere_t sn14("sn14",0,50,0,2*PI,0.25*PI,0.75*PI);
    Sphere_t sn15("sn15",0,50,0,2*PI,89.*deg,91.*deg);
    Sphere_t sn16("sn16",0,50,0,2*PI,0.*deg,89.*deg);
    Sphere_t sn17("sn17",0,50,0,2*PI,91.*deg,89.*deg);

    Sphere_t s2("Spherical Shell",45,50,0,2*PI,0,PI);
    Sphere_t sn2("sn2",45,50,0.5*PI,0.5*PI,0,PI);
    Sphere_t sn22("sn22",0,50,0.5*PI,0.5*PI,0,PI);

    Sphere_t s3("Band (theta segment)",45,50,0,2*PI,PI/4,0.5*PI);
    Sphere_t s32("Band (theta segment2)",45,50,0,2*PI,0,PI/4);
    Sphere_t s33("Band (theta segment1)",45,50,0,2*PI,PI*3/4,PI/4);
    Sphere_t s34("Band (theta segment)",4,50,0,2*PI,PI/4,0.5*PI);
    Sphere_t s4("Band (phi segment)",45,50,-PI/4,0.5*PI,0,2*PI);
    //    std::cout<<"s4.fSPhi = "<<s4.GetSPhi()<<std::endl;
    Sphere_t s41("Band (phi segment)",5,50,-PI,3.*PI/2.,0,2*PI);
    Sphere_t s42("Band (phi segment)",5,50,-PI/2,3.*PI/2.,0,2*PI);
    Sphere_t s5("Patch (phi/theta seg)",45,50,-PI/4,0.5*PI,PI/4,0.5*PI);

    Sphere_t s6("John example",300,500,0,5.76,0,PI) ; 
    Sphere_t s7("sphere7",1400.,1550.,0.022321428571428572,0.014642857142857141,
	                  1.5631177553663251,0.014642857142857141    );
    Sphere_t s8("sphere",278.746, 280.0, 0.0*deg, 360.0*deg,
	                               0.0*deg, 90.0*deg);
    Sphere_t b216("b216", 1400.0, 1550.0, 
                  0.022321428571428572, 
		  0.014642857142857141,
                  1.578117755366325,
                  0.014642857142867141);

    Sphere_t s9("s9",0,410,0*deg,360*deg,90*deg,90*deg);

    Sphere_t b402("b402", 475, 480, 
               0*deg,360*deg,17.8*deg,144.4*deg);

    Sphere_t b1046("b1046", 4750*1e6, 4800*1e6, 
               0*deg,360*deg,0*deg,90*deg);

    Sphere_t testTheta("opticalEscape",0.,2.995,0*deg,360*deg,0*deg,120*deg);
   
    Vec_t p402(471.7356120367253, 51.95081450791341, 5.938043020529463);
    
    Vec_t v402(-0.519985502840818, 0.2521089719986221, 0.8161226274728446);

    Vec_t p216(1549.9518578505142,1.2195415370970153,-12.155289555510985), 
              v216(-0.61254821852534425,-0.51164551429243466,-0.60249775741147549);

    Vec_t s9p(384.8213314370455,
	       134.264386151667,
	       -44.56026800002064);

    Vec_t s9v(-0.6542770611918751,
		   -0.0695116921641141,
		   -0.7530535517814154);

    Sphere_t s10("s10",0,0.018,0*deg,360*deg,0*deg,180*deg);

    Vec_t s10p(0.01160957408065766,
                     0.01308205826682229,0.004293345210644617);

    Sphere_t s11("s11",5000000.,
                    3700000000.,
                   0*deg,360*deg,0*deg,180*deg);

    Vec_t ps11( -3072559844.81995153427124, -1924000000, -740000000 );

    Sphere_t sAlex("sAlex",500.,
                    501.,
                   0*deg,360*deg,0*deg,180*deg);

    Vec_t psAlex(-360.4617031263808,
                     -158.1198807105035,308.326878333183);

    Vec_t vsAlex(-0.7360912456240805,-0.4955800202572754,0.4610532741813497 );

    Sphere_t sLHCB("sLHCB",8600, 8606, 
    -1.699135525184141*deg,
    3.398271050368263*deg,88.52855940538514*deg,2.942881189229715*deg );

    Vec_t pLHCB(8600.242072535835,-255.1193517702246,-69.0010277128286);

    Sphere_t b658("b658", 209.6, 211.2658, 
                           0.0*deg, 360*deg, 0.0*deg, 90*deg); 


    Vec_t p658(-35.69953348982516, 198.3183279249958, 56.30959457033987 ); 
    Vec_t v658(-.2346058124516908,-0.9450502890785083,0.2276841318065671); 

    Sphere_t spAroundX("SpAroundX",  10., 1000., -1.0*deg, 
                                                     2.0*deg, 
                                        0.*deg, 180.0*deg );

    double  radOne = 100.0;
    double  angle = -1.0*deg - 0.25*vecgeom::kTolerance*1000;
    Vec_t  ptPhiMinus= Vec_t( radOne*std::cos(angle) ,
                                           radOne*std::sin(angle),
                                           0.0 );

   // spheres for theta cone intersections

   Sphere_t s13("Band (theta segment)",5,50,0,2*PI,PI/6.,0.5*PI);
   Sphere_t s14("Band (theta segment)",5,50,0,2*PI,PI/3.,0.5*PI);


   // b. 830

   double mainInnerRadius =  21.45 * 10 ;
   double mainOuterRadius = 85.0 * 10 ;
   double minTheta = 18.0 * deg ;
 
   Sphere_t sb830( "mainSp",
                    mainInnerRadius,
                    mainOuterRadius,
                    0.0, M_PI*2,
                    minTheta,
                    M_PI - 2*minTheta);

   
   Vec_t pb830(81.61117212,-27.77179755,196.4143423);
   Vec_t vb830(0.1644697995,0.18507236,0.9688642354);
    
// Check Sphere_t::Inside

    inside = s7.Inside(Vec_t(1399.984667238032,
                                             5.9396696802500299,
                                            -2.7661927818688308)  ) ;
    // std::cout<<"s7.Inside(Vec_t(1399.98466 ... = "
    //       <<inside<<std::endl ;

    inside = s8.Inside(Vec_t(-249.5020724528353,
					       26.81253142743162,
                                              114.8988524453591  )  ) ;
    // std::cout<<"s8.Inside(Vec_t(-249.5020 ... = "<<inside<<std::endl ;
    inside = b216.Inside(p216);
    // std::cout<<"b216.Inside(p216) = "<<inside<<std::endl ;
    inside = s1.Inside(pz);
    // std::cout<<"s1.Inside(pz) = "<<inside<<std::endl ;

    inside = s9.Inside(s9p);
    // std::cout<<"s9.Inside(s9p) = "<<inside<<std::endl ;

    inside = b402.Inside(p402);
    // std::cout<<"p402.Inside(p402) = "<<nside<<std::endl ;

    inside = s10.Inside(s10p);
    // std::cout<<"s10.Inside(s10p) = "<<inside<<std::endl ;
    // std::cout<<"p radius = "<<s10p.mag()<<std::endl ;
    
    inside = s11.Inside(ps11);
    // std::cout<<"s11.Inside(ps11) = "<<inside<<std::endl ;
    // std::cout<<"ps11.mag() = "<<ps11.mag()<<std::endl ;
    
    inside = sLHCB.Inside(pLHCB);
    // std::cout<<"sLHCB.Inside(pLHCB) = "<<inside<<std::endl ;
    // std::cout<<"pLHCB.mag() = "<<pLHCB.mag()<<std::endl ;
    
    inside = spAroundX.Inside(ptPhiMinus);
    //std::cout<<"spAroundX.Inside(ptPhiMinus) = "<<inside<<std::endl ;
    //assert(inside==vecgeom::EInside::kSurface);
    inside = b658.Inside(p658);
    assert(inside==vecgeom::EInside::kOutside);
    //std::cout<<"b658.Inside(p658) = "<<inside<<std::endl ;

    assert(s1.Inside(pzero)==vecgeom::EInside::kInside);
    assert(s2.Inside(pzero)==vecgeom::EInside::kOutside);
    assert(s2.Inside(ponrmin2)==vecgeom::EInside::kSurface);
    assert(s2.Inside(ponrmax2)==vecgeom::EInside::kSurface);
    assert(s3.Inside(pontheta1)==vecgeom::EInside::kSurface);
    assert(s3.Inside(pontheta2)==vecgeom::EInside::kSurface);
    assert(s4.Inside(ponphi1)==vecgeom::EInside::kSurface);
    assert(s4.Inside(ponphi1)==vecgeom::EInside::kSurface);
    assert(s4.Inside(pOverPhi)==vecgeom::EInside::kOutside);
    assert(s4.Inside(pInPhi)==vecgeom::EInside::kInside);
    assert(s5.Inside(pbigz)==vecgeom::EInside::kOutside);

    assert(s41.Inside(pmx)==vecgeom::EInside::kSurface);
    assert(s42.Inside(pmx)==vecgeom::EInside::kSurface);
    assert(sn11.Inside(pzero)==vecgeom::EInside::kSurface);
    assert(sn12.Inside(pzero)==vecgeom::EInside::kSurface);
    //std::cout<<"sn11.Inside0="<<sn11.Inside(pzero)<<std::endl;
    //std::cout<<"sn12.Inside0="<<sn12.Inside(pzero)<<std::endl;
    // std::cout<<"sn13.Inside0="<<sn13.Inside(pzero)<<std::endl;
    //std::cout<<"sn13.InsideZ="<<sn13.Inside(Vec_t(0,0,1.))<<std::endl;
    assert(sn13.Inside(pzero)==vecgeom::EInside::kSurface);
    assert(sn13.Inside(-pz)==vecgeom::EInside::kInside);
    assert(sn12.Inside(pz)==vecgeom::EInside::kInside);
    assert(sn12.Inside(px)==vecgeom::EInside::kOutside);
    assert(sn11.Inside(px)==vecgeom::EInside::kSurface);

// Checking Sphere_t::SurfaceNormal
    double p2=1./std::sqrt(2.),p3=1./std::sqrt(3.);


    valid=sn1.Normal(Vec_t(0.,0.,50.),norm);
    assert(ApproxEqual(norm,Vec_t(p3,p3,p3)));
    valid=sn1.Normal(Vec_t(0.,0.,0.),norm);
    assert(ApproxEqual(norm,Vec_t(p2,p2,0.)));

    valid=sn11.Normal(Vec_t(0.,0.,0.),norm);
    assert(ApproxEqual(norm,Vec_t(0.,0.,-1.)));
    valid=sn12.Normal(Vec_t(0.,0.,0.),norm);
    assert(ApproxEqual(norm,Vec_t(0.,0.,-1.)));
    valid=sn13.Normal(Vec_t(0.,0.,0.),norm);
    assert(ApproxEqual(norm,Vec_t(0.,0.,1.)));

    valid=sn2.Normal(Vec_t(-45.,0.,0.),norm);
    assert(ApproxEqual(norm,Vec_t(p2,-p2,0.)));



    valid=s1.Normal(ponrmax1,norm);
    assert(ApproxEqual(norm,vx));

// Checking Sphere_t::SafetyFromInside(P)
    //Full Sphere
    Dist=s1.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,50));
    Dist=s1.SafetyFromInside(ponrmax1);
    assert(ApproxEqual(Dist,0));
    Dist=s1.SafetyFromInside(px);
    assert(ApproxEqual(Dist,20));
    //Full Sphere with Rmin
    Dist=s2.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,0));
    Dist=s2.SafetyFromInside(ponrmax1);
    assert(ApproxEqual(Dist,0));
    Dist=s2.SafetyFromInside(ponrmin1);
    assert(ApproxEqual(Dist,0));
    //Sphere with full Phi and Theta section
    Dist=sn11.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,0));
    Dist=sn11.SafetyFromInside(px);
    //std::cout<<"Dist=sn11.SafetyFromInside(px) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,0));
    Dist=sn12.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,0));
    Dist=sn12.SafetyFromInside(pz);
    //std::cout<<"Dist=sn12.Inside(pz) = "<<sn12.Inside(pz)<<std::endl;
    assert(ApproxEqual(Dist,20));
    Dist=sn13.SafetyFromInside(-pz);
    assert(ApproxEqual(Dist,20));
    Dist=s3.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,0));
    //std::cout<<"Dist=sn3.pzero = "<<Dist<<std::endl;

    Dist=s9.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,0));
    Dist=s9.SafetyFromInside(px);
    //std::cout<<"Dist=sn11.SafetyFromInside(px) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,0));
    //Sphere with Phi section
    Dist=s41.SafetyFromInside(pzero);
    assert(ApproxEqual(Dist,0));
    Dist=s41.SafetyFromInside(ponrmax1);
    assert(ApproxEqual(Dist,0));


// Checking Sphere_t::DistanceToOut(p,v)


     Dist=s1.DistanceToOut(pz,vz,norm,convex);
     // std::cout<<"Dist=s1.DistanceToOut(pz,vz) = "<<Dist<<std::endl;
     assert(ApproxEqual(Dist,20.));

     Dist=s1.DistanceToOut(ponrmax1,vx,norm,convex);
     assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vx));

     Dist=s2.DistanceToOut(ponrmin1,vx,norm,convex);
     assert(ApproxEqual(Dist,5)&&ApproxEqual(norm,vx));

     Dist=s2.DistanceToOut(ponrmax2,vx,norm,convex);
     assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vxy));

    Dist=s1.DistanceToOut(pzero,vx,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vx));
    Dist=s1.DistanceToOut(pzero,vmx,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vmx));
    Dist=s1.DistanceToOut(pzero,vy,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vy));
    Dist=s1.DistanceToOut(pzero,vmy,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vmy));
    Dist=s1.DistanceToOut(pzero,vz,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vz));
    Dist=s1.DistanceToOut(pzero,vmz,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vmz));
    Dist=s1.DistanceToOut(pzero,vxy,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vxy));

    Dist=s4.DistanceToOut(ponphi1,vx,norm,convex);
    //assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmxmy));
    Dist=s4.DistanceToOut(ponphi2,vx,norm,convex);
    // assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmxy));
    Dist=s3.DistanceToOut(pontheta1,vz,norm,convex);
    // assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vy));
    Dist=s32.DistanceToOut(pontheta1,vmz,norm,convex);
    //assert(ApproxEqual(Dist,0)&&ApproxEqual(norm,vmy));
    Dist=s32.DistanceToOut(pontheta1,vz,norm,convex);
    //assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vz));
    Dist=s1.DistanceToOut(pzero,vmz,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vmz));
    Dist=s1.DistanceToOut(pzero,vxy,norm,convex);
    assert(ApproxEqual(Dist,50)&&ApproxEqual(norm,vxy));
    
    
    Dist=s2.DistanceToOut(ponrmin1,vxy,norm,convex);
    //    std::cout<<"Dist=s2.DistanceToOut(pormin1,vxy) = "<<Dist<<std::endl;

    Dist=s2.DistanceToOut(ponrmax1,vmx,norm,convex);
    //    std::cout<<"Dist=s2.DistanceToOut(ponxside,vmx) = "<<Dist<<std::endl;
    Dist=s2.DistanceToOut(ponrmax1,vmxmy,norm,convex);
    //    std::cout<<"Dist=s2.DistanceToOut(ponxside,vmxmy) = "<<Dist<<std::endl;
    Dist=s2.DistanceToOut(ponrmax1,vz,norm,convex);
    //    std::cout<<"Dist=s2.DistanceToOut(ponxside,vz) = "<<Dist<<std::endl;

    //    Dist=s2.DistanceToOut(pbigx,vx,norm,convex);
    //    std::cout<<"Dist=s2.DistanceToOut(pbigx,vx) = "<<Dist<<std::endl;
    //    Dist=s2.DistanceToOut(pbigx,vxy,norm,convex);
    //    std::cout<<"Dist=s2.DistanceToOut(pbigx,vxy) = "<<Dist<<std::endl;
    //    Dist=s2.DistanceToOut(pbigx,vz,norm,convex);
    //    std::cout<<"Dist=s2.DistanceToOut(pbigx,vz) = "<<Dist<<std::endl;
    //Test Distance for phi section
    Dist=sn22.DistanceToOut(Vec_t(0.,49.,0.),vmy,norm,convex);
    assert(ApproxEqual(Dist,49.));
    Dist=sn22.DistanceToOut(Vec_t(-45.,0.,0.),vx,norm,convex);
    assert(ApproxEqual(Dist,45.));
    //std::cout<<"Dist from Center ="<<sn22.DistanceToOut(Vec_t(0.,49.,0),Vec_t(0,-1,0),norm,convex)<<std::endl;
    //std::cout<<"Dist from Center ="<<sn22.DistanceToOut(Vec_t(-45.,0.,0),Vec_t(1,0,0),norm,convex)<<std::endl;

    Dist=s13.DistanceToOut(Vec_t(20.,0.,0.),vz,norm,convex);
    // std::cout<<"s13.DistanceToOut(Vec_t(20.,0.,0.),vz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,34.641016151377549));

    Dist=s13.DistanceToOut(Vec_t(20.,0.,0.),vmz,norm,convex);
    // std::cout<<"s13.DistanceToOut(Vec_t(20.,0.,0.),vmz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,11.547005383792508));

    Dist=s14.DistanceToOut(Vec_t(20.,0.,0.),vz,norm,convex);
    // std::cout<<"s14.DistanceToOut(Vec_t(20.,0.,0.),vz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,11.547005383792508));

    Dist=s14.DistanceToOut(Vec_t(20.,0.,0.),vmz,norm,convex);
    // std::cout<<"s14.DistanceToOut(Vec_t(20.,0.,0.),vmz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,34.641016151377549));

    Dist=sb830.DistanceToOut(pb830,vb830,norm,convex);
    //std::cout<<"sb830.DistanceToOut(pb830,vb830... = "<<Dist<<std::endl;
    inside = sb830.Inside(pb830+Dist*vb830);
    assert(inside==vecgeom::EInside::kSurface);
    //std::cout<<"sb830.Inside(pb830+Dist*vb830) = "<<inside<<std::endl ;

    Dist=sn11.DistanceToOut( Vec_t(0.,0.,20.), vmz, norm,convex);
    // std::cout<<"sn11.DistanceToOut(Vec_t(0.,0.,20.),vmz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,20.));

    Dist=sn11.DistanceToOut(Vec_t(0.,0.,0.),vmz,norm,convex);
    // std::cout<<"sn11.DistanceToOut(Vec_t(0.,0.,0.),vmz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,0.));

    Dist=sn11.DistanceToOut(Vec_t(0.,0.,0.),vz,norm,convex);
    // std::cout<<"sn11.DistanceToOut(Vec_t(0.,0.,0.),vmz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,50.));

    Dist=sn11.DistanceToOut(Vec_t(10.,0.,0.),vmz,norm,convex);
    // std::cout<<"sn11.DistanceToOut(Vec_t(10.,0.,0.),vmz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,0.));
   

    Dist=sn12.DistanceToOut(Vec_t(0.,0.,20.),vxmz,norm,convex);
    // std::cout<<"sn12.DistanceToOut(Vec_t(0.,0.,20.),vxmz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,20./std::sqrt(2.)));

    Dist=sn12.DistanceToOut(Vec_t(0.,0.,0.),vmz,norm,convex);
    // std::cout<<"sn12.DistanceToOut(Vec_t(0.,0.,0.),vmz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,0.));

    Dist=sn12.DistanceToOut(Vec_t(10.,0.,10.),vz,norm,convex);
    // std::cout<<"sn12.DistanceToOut(Vec_t(10.,0.,10.),vz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,38.989794855663561179));


    Dist=sn12.DistanceToOut(Vec_t(10.,0.,10.),vmz,norm,convex);
    // std::cout<<"sn12.DistanceToOut(Vec_t(10.,0.,10.),vmz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,0.));


    Dist=sn14.DistanceToOut(Vec_t(10.,0.,10.),vz,norm,convex);
    // std::cout<<"sn14.DistanceToOut(Vec_t(10.,0.,10.),vz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,0.));


    Dist=sn14.DistanceToOut(Vec_t(10.,0.,5.),vz,norm,convex);
    // std::cout<<"sn14.DistanceToOut(Vec_t(10.,0.,5.),vz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,5.));


    Dist=sn14.DistanceToOut(Vec_t(10.,0.,5.),vmxz,norm,convex);
    // std::cout<<"sn14.DistanceToOut(Vec_t(10.,0.,5.),vmxz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,3.5355339059327381968));

    Dist=sn14.DistanceToOut(Vec_t(10.,0.,10.),vmxz,norm,convex);
    // std::cout<<"sn14.DistanceToOut(Vec_t(10.,0.,10.),vmxz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,0.));

    Dist=sn12.DistanceToOut(Vec_t(0.,0.,0.),vx,norm,convex);
    //std::cout<<"sn12.DistanceToOut(Vec_t(0.,0.,0.),vx... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,0.));

    pRand = Vec_t(16.20075504802145, -22.42917454903122, -41.6469406430184);
    vRand = Vec_t( -0.280469198715188, 0.4463870649961534, 0.8497503261406731 );
    valid = sn17.Normal(pRand,norm);
    //std::cout<<"norm = sn17.Normal(pRand) = "<<norm.x()<<", "<<norm.y()<<", "<<norm.z()<<std::endl;
    assert(valid);
    inside = sn17.Inside(pRand);
    //std::cout<<"sn17.Inside(pRand) = "<<inside<<std::endl ;
    assert(inside==vecgeom::EInside::kSurface);
    Dist = sn17.DistanceToOut(pRand,vRand,norm,convex);    
    //std::cout<<"sn17.DistanceToOut(pRand,vRand,...) = "<<Dist<<std::endl;  
    assert(ApproxEqual(Dist,48.95871780757848768));

// Checking Sphere_t::SafetyFromOutside(P)

    Dist=s2.SafetyFromOutside(pzero);
    assert(ApproxEqual(Dist,45));
    Dist=s1.SafetyFromOutside(ponrmax1);
    assert(ApproxEqual(Dist,0));

// Checking Sphere_t::DistanceToIn(P,V)


    pRand = Vec_t( -10.276604989981144911, -4.7072500741022338389, 1);
    vRand = Vec_t( -0.28873162920537848164, 0.51647890054938394577, -0.80615357816219324061);
    Dist = sn17.DistanceToIn(pRand,vRand);
    //std::cout<<"sn17.DistanceToIn(pRand,vRand) = "<<Dist<<std::endl;  
    assert(ApproxEqual(Dist,1.4874615083453075481));


    Dist=s1.DistanceToIn(ponzmax,vz);
    //std::cout<<"s1.DistanceToIn(ponzmax,vz) = "<<Dist<<std::endl;
    Dist=s1.DistanceToIn(pbigy,vy);
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));
    Dist=s1.DistanceToIn(pbigy,vmy);
    assert(ApproxEqual(Dist,50));

    Dist=s2.DistanceToIn(pzero,vy);
    //std::cout<<"s2.DistanceToIn(pzero,vx) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,45));
    Dist=s2.DistanceToIn(pzero,vmy);
    assert(ApproxEqual(Dist,45));
    Dist=s2.DistanceToIn(ponrmin1,vx);
    //std::cout<<"s2.DistanceToIn(ponmin1,vx) = "<<Dist<<std::endl;
    assert(Dist==0);
    Dist=s2.DistanceToIn(ponrmin1,vmx);
    assert(ApproxEqual(Dist,90));

    Dist=s2.DistanceToIn(ponrmin2,vx);
    assert(Dist==0);
    Dist=s2.DistanceToIn(ponrmin2,vmx);
    assert(ApproxEqual(Dist,90/std::sqrt(2.)));


    Dist=s3.DistanceToIn(ptesttheta1,vmz);
    //    std::cout<<"s3.DistanceToIn(ptesttheta1,vmz) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,100-48/std::sqrt(2.)));
    Dist=s3.DistanceToIn(pontheta1,vz);
    //    std::cout<<"s3.DistanceToIn(pontheta1,vz) = "<<Dist<<std::endl;
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));
    Dist=s3.DistanceToIn(pontheta1,vmz);
    assert(Dist==0);
    Dist=s3.DistanceToIn(pontheta2,vz);
    //    std::cout<<"s3.DistanceToIn(pontheta2,vz) = "<<Dist<<std::endl;
    assert(Dist==0);
    Dist=s3.DistanceToIn(pontheta2,vmz);
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));
    Dist=s32.DistanceToIn(pontheta1,vz);
    //    std::cout<<"s32.DistanceToIn(pontheta1,vz) = "<<Dist<<std::endl;
    assert(Dist==0);
    Dist=s32.DistanceToIn(pontheta1,vmz);
    //    std::cout<<"s32.DistanceToIn(pontheta1,vmz) = "<<Dist<<std::endl;
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));
    Dist=s33.DistanceToIn(pontheta2,vz);
    //    std::cout<<"s33.DistanceToIn(pontheta2,vz) = "<<Dist<<std::endl;
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));
    Dist=s33.DistanceToIn(pontheta2,vmz);
    //    std::cout<<"s33.DistanceToIn(pontheta2,vmz) = "<<Dist<<std::endl;
    assert(Dist==0);

    Dist=s4.DistanceToIn(pbigy,vmy);
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));
    //assert(Dist==kInfinity);
    Dist=s4.DistanceToIn(pbigz,vmz);
    assert(ApproxEqual(Dist,50));
    Dist=s4.DistanceToIn(pzero,vy);
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));
    //assert(Dist==kInfinity);
    Dist=s4.DistanceToIn(pzero,vx);
    assert(ApproxEqual(Dist,45));

    Dist=s4.DistanceToIn(ptestphi1,vx);
    assert(ApproxEqual(Dist,100+45/std::sqrt(2.)));
    Dist=s4.DistanceToIn(ponphi1,vmxmy);
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));
    //assert(Dist==kInfinity);
    Dist=s4.DistanceToIn(ponphi1,vxy);
    //     std::cout<<"s4.DistanceToIn(ponphi1,vxy) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,0));

    Dist=s4.DistanceToIn(ptestphi2,vx);
    assert(ApproxEqual(Dist,100+45/std::sqrt(2.)));
    Dist=s4.DistanceToIn(ponphi2,vmxy);
    if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    assert(ApproxEqual(Dist,UUtils::Infinity()));
    //assert(Dist==kInfinity);
    Dist=s4.DistanceToIn(ponphi2,vxmy);
    //     std::cout<<"s4.DistanceToIn(ponphi2,vxmy) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,0));

    Dist=s3.DistanceToIn(pzero,vx);
    assert(ApproxEqual(Dist,45));
    Dist=s3.DistanceToIn(ptesttheta1,vmz);
    assert(ApproxEqual(Dist,100-48/std::sqrt(2.)));
    Dist=b216.DistanceToIn(p216,v216);
    // std::cout<<"b216.DistanceToIn(p216,v216) = "<<Dist<<std::endl;

    Dist=b658.DistanceToIn(pzero,vz);
    //std::cout<<"b658.DistanceToIn(pzero,vz) = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,209.59999999999999432));


    Dist=s13.DistanceToIn(Vec_t(20.,0.,-70.),vz);
    // std::cout<<"s13.DistanceToIn(Vec_t(20.,0.,-70.),vz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,58.452994616207498));

    Dist=s13.DistanceToIn(Vec_t(20.,0.,70.),vmz);
    // std::cout<<"s13.DistanceToIn(Vec_t(20.,0.,70.),vmz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,35.358983848622451));


    Dist=s14.DistanceToIn(Vec_t(20.,0.,-70.),vz);
    // std::cout<<"s14.DistanceToIn(Vec_t(20.,0.,-70.),vz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,35.358983848622451));

    Dist=s14.DistanceToIn(Vec_t(20.,0.,70.),vmz);
    // std::cout<<"s14.DistanceToIn(Vec_t(20.,0.,70.),vmz... = "<<Dist<<std::endl;
    assert(ApproxEqual(Dist,58.452994616207498));

    Dist=b1046.DistanceToIn(Vec_t(0.,0.,4800*1e6),vmz);
	assert(ApproxEqual(Dist,0.));
    //std::cout<<"b1046.DistanceToIn(Vec_t(0.,0.,4800*km),vmz... = "<<Dist<<std::endl;
    //if( Dist >= UUtils::kInfinity ) Dist = UUtils::Infinity(); 
    //assert(ApproxEqual(Dist,UUtils::Infinity()));
  

   //Check DistanceToOut for  center point
   
  
   Sphere_t sntest("sntest",0,50,0,2*PI,0.0,0.75*PI);
   Vec_t in1, in2;
   in1 = Vec_t(0.0, 0.0, 30);//0,0,30);
   in2 = Vec_t(0,0,0);
   double length = (in1-in2).Mag();
   Vec_t dir12 = (in2-in1)/length;
   //for(int i = 0 ; i < 100; i++)
   int cnt=0;
   for(int i = 0 ; i < 100; i++)
   {
       convex=true;
       if(sntest.Inside(in1) != vecgeom::EInside::kOutside)
       {std::cout<<std::setprecision(25);
       
	 Dist=sntest.DistanceToOut(in1,dir12,norm,convex);
	
	double diff=Dist-length;

	if(diff > 1e-6)
    {
        std::cout<<" i="<<i<<" Dout="<<Dist<<" dif="<< Dist-length<<std::endl;
        std::cout<<"In-1 : "<<in1<<"  :: Dir : "<<dir12<<std::endl;
        cnt++;
    }
        in1=in1+Vec_t(0.00001,0.00001,0);
        //in2=in2+Vec_t(-0.00001,-0.00001,0);
        length = (in1-in2).Mag();
        dir12 = (in2-in1)/length;
       }
       else
       {
         std::cout<<" error in test In="<<sntest.Inside(in1)<<std::endl;
       }


   }
std::cout<<"\n--------------------------------------------------------------------\n";
std::cout<<"Number of Points with Unexpected Distance Values  : "<<cnt<<std::endl;
std::cout<<"--------------------------------------------------------------------\n";




    //
    //End adding unit test from Geant4
    //

    //
    //Adding some more test cases
    //

    Sphere_t sntestB("sntestB",0,50,0,2*PI,0.50*PI,0.50*PI); //Bottom Hemisphere
    Dist = sntestB.DistanceToOut(pzero,vmz,norm,convex);
    assert(ApproxEqual(Dist,50.));

    Dist = sntestB.DistanceToOut(pzero,vz,norm,convex);
    assert(ApproxEqual(Dist,0.));

    Dist = sntestB.DistanceToOut(px,vz,norm,convex);
    assert(ApproxEqual(Dist,0.));

    Dist = sntestB.DistanceToOut(px,vmz,norm,convex);
    assert(ApproxEqual(Dist,40.));

    Dist = sntestB.DistanceToOut(py,vz,norm,convex);
    assert(ApproxEqual(Dist,0.));

    Dist = sntestB.DistanceToOut(py,vmz,norm,convex);
    assert(ApproxEqual(Dist,40.));

    Dist = sntestB.DistanceToOut(pmz,vz,norm,convex);
    assert(ApproxEqual(Dist,30.));

    Dist = sntestB.DistanceToOut(pmz,vmz,norm,convex);
    assert(ApproxEqual(Dist,20.));

    Dist = sntestB.DistanceToOut(Vec_t(30,0,-30),vz,norm,convex);
    assert(ApproxEqual(Dist,30.));

    Dist = sntestB.DistanceToOut(Vec_t(30,0,-30),vmz,norm,convex);
    assert(ApproxEqual(Dist,10.));

    Dist = sntestB.DistanceToOut(px,vx,norm,convex);
    assert(ApproxEqual(Dist,20.));

    Dist = sntestB.DistanceToOut(px,vmx,norm,convex);
    assert(ApproxEqual(Dist,80.));

    Dist = sntestB.DistanceToOut(pmx,vx,norm,convex);
    assert(ApproxEqual(Dist,80.));

    Dist = sntestB.DistanceToOut(pmx,vmx,norm,convex);
    assert(ApproxEqual(Dist,20.));

    Dist = sntestB.DistanceToOut(pmy,vy,norm,convex);
    assert(ApproxEqual(Dist,80.));

    Dist = sntestB.DistanceToOut(pmy,vmy,norm,convex);
    assert(ApproxEqual(Dist,20.));

    Dist = sntestB.DistanceToOut(pmy,vx,norm,convex);
    assert(ApproxEqual(Dist,40.));

    Dist = sntestB.DistanceToOut(pmy,vmx,norm,convex);
    assert(ApproxEqual(Dist,40.));

    Dist = sntestB.DistanceToIn(pzero,vmz);
    assert(ApproxEqual(Dist,0.));

    Vec_t southPolePoint(0,0,-50);
    Dist = sntestB.DistanceToIn(southPolePoint,vz);
    assert(ApproxEqual(Dist,0.));

    assert(sntestB.Inside(southPolePoint) == vecgeom::EInside::kSurface);
    assert(sntestB.Inside(pzero) == vecgeom::EInside::kSurface);
    assert(sntestB.Inside(px) == vecgeom::EInside::kSurface);
    assert(sntestB.Inside(-py) == vecgeom::EInside::kSurface);

    Vec_t northPolePoint(0,0,50);
    Dist = sntestB.DistanceToIn(northPolePoint,vmz);
    assert(ApproxEqual(Dist,50.));

    assert(sntestB.Inside(northPolePoint) == vecgeom::EInside::kOutside);

    Dist = sntest.DistanceToIn(southPolePoint,vz);
    assert(ApproxEqual(Dist,50.));

    assert(sntest.Inside(southPolePoint) == vecgeom::EInside::kOutside);

    Dist = sntest.DistanceToIn(northPolePoint,vmz);
    assert(ApproxEqual(Dist,0.));

    assert(sntest.Inside(northPolePoint) == vecgeom::EInside::kSurface);
    
	assert(sntest.Inside(southPolePoint) == vecgeom::EInside::kOutside);
	
	assert(sntest.Inside(px) == vecgeom::EInside::kInside);
	assert(sntest.Inside(py) == vecgeom::EInside::kInside);
	//std::cout<<"Location of Px : "<<sntest.Inside(px)<<std::endl;

	assert(sntest.Inside(pz) == vecgeom::EInside::kInside);
	assert(sntest.Inside(-pz) == vecgeom::EInside::kOutside);
	assert(sntestB.Inside(pz) == vecgeom::EInside::kOutside);
	assert(sntestB.Inside(-pz) == vecgeom::EInside::kInside);
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
   assert(TestSphere<USphere>());
   std::cout << "USphere passed\n";
       #else
       std::cerr << "VECGEOM_USOLIDS was not defined\n";
       return 2;
 #endif
     }
     else if( ! strcmp(argv[1], "--vecgeom") )
     {
   assert(TestSphere<vecgeom::SimpleSphere>());
   std::cout << "VecGeomSphere passed\n";
     }
     else
     {
       std::cerr << "need to give argument :--usolids or --vecgeom\n";
       return 1;
     }


   return 0;
 }

/*
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
*/
