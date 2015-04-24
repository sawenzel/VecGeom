#include "G4Hype.hh"
#include "G4Sphere.hh"
#include "G4ThreeVector.hh"
#define PI 3.14159265358979323846
int main(){

G4Hype hype("testHype",0,10,0,PI/4,50);
std::cout<<"Volume : "<<hype.GetCubicVolume()<<std::endl;
std::cout<<"SurfaceArea : "<<hype.GetSurfaceArea()<<std::endl;
G4ThreeVector testPoint(11,0,0);
G4ThreeVector normal1 = hype.SurfaceNormal(testPoint);
std::cout<< "Normal Calculated from Geant4 - 1 : "<< normal1 <<std::endl;

std::cout<< "Expected Safety Distance from G4 using Normal : "<< hype.DistanceToIn(testPoint,normal1)<<std::endl;

std::cout<< "Expected Safety Distance from G4 : "<< hype.DistanceToIn(testPoint)<<std::endl;
//G4Sphere sph("testSphere",10,15,0,2*PI,0,PI);
//std::cout<<"Surface Area of Sphere : "<<sph.GetSurfaceArea()<<std::endl;

std::cout<<"--------------------------------"<<std::endl;
G4Hype testN("testHype",0,3,0,0.785398,10);
G4ThreeVector testPointN(5,5,8);
G4ThreeVector normal = testN.SurfaceNormal(testPointN);
std::cout<< "Normal Calculated from Geant4 : "<< normal <<std::endl;

return 0;
}
