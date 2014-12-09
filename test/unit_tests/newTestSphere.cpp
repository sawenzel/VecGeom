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
    Vec_t norm;
    bool convex;
    convex = true;
    //JUST A TESTING SPHERE FOR DEBUGGING
    //Sphere_t test("Solid VecGeomSphere #test",6, 8, 0, 2*PI, 0., PI);
    //std::cout<<std::setprecision(15);
    Sphere_t test("Solid VecGeomSphere #test",10, 15, 0.0,PI/6, 0.0, PI);
    
    //Vec_t testPoint(2.243909, -0.241734, -0.499920);
    Vec_t testPoint(29.99519584853791442924, 22.27714142630187410532, 28.44427311866511587368);
    //double mag=std::sqrt(testPoint[0]*testPoint[0] + testPoint[1]*testPoint[1] + testPoint[2]*testPoint[2]);
   // std::cout<<"Magnitude of Point : "<< mag <<std::endl;
    //Vec_t testDir(0.002330, 0.974209, 0.225635);
    Vec_t testDir(-0.60416382879034258302, -0.63234763164111817968, -0.48489436039127348232);
    //double pdotV = testPoint[0]*testDir[0]+testPoint[1]*testDir[1]+testPoint[2]*testDir[2];
   
    //if(pdotV < 0)
      //  std::cout<<"Direction of Point : IN"<<std::endl;
    //else
      //  std::cout<<"Direction of Point : Out"<<std::endl;
    
    //std::cout<<"Theta is : " <<std::acos(testDir[2]/ mag)<<std::endl;
    //std::cout<<"PHI is : "<< std::atan(testDir[1]/testDir[0])<<std::endl;
    Dist=test.DistanceToIn(testPoint,testDir);//,norm,convex);
//Dist = test.SafetyFromInside(testPoint);
    //Dist = test.SafetyFromOutside(testPoint);
    std::cout<<std::endl<<"Distance : "<<Dist<<std::endl;
    
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

