#include <iostream>
#include "USphere.hh"
#define PI 3.14159265358979323846

int main(){
USphere *sph=new USphere("testSphere",6,8,0,2*PI,0,PI);
//double point[3]={0,0,6.5};
UVector3 pzero(0,0,0);

UVector3 pointX(6.5,0,0);
std::cout<<"Distance From Inside X: "<<sph->SafetyFromInside(pointX)<<std::endl; //Gives 0.5, expected

UVector3 pointY(0,6.5,0);
std::cout<<"Distance From Inside Y: "<<sph->SafetyFromInside(pointY)<<std::endl; //Gives 0.5, expected

UVector3 pointZ(0,0,-6.5);
std::cout<<"Raman Distance From Inside Z: "<<sph->SafetyFromInside(pointZ)<<std::endl; //Gives 0, UNEXPECTED

std::cout<<"Distance From pzero "<<sph->SafetyFromInside(pzero)<<std::endl; 

std::cout<<std::endl<<std::endl;

UVector3 pI(5,0,0);
UVector3 pO(9,0,0);

std::cout<<"Point is outside and SafetyFromOutside gives : "<<sph->SafetyFromOutside(pO)<<std::endl;
std::cout<<"Point is outside and SafetyFromOutside gives : "<<sph->SafetyFromOutside(pI)<<std::endl;

std::cout<<std::endl<<std::endl;

std::cout<<"Point is outside and SafetyFromInside gives : "<<sph->SafetyFromInside(pO)<<std::endl;
std::cout<<"Point is outside and SafetyFromInside gives : "<<sph->SafetyFromInside(pI)<<std::endl;

std::cout<<std::endl<<std::endl;

std::cout<<"---------------------------------------------"<<std::endl<<std::endl;

UVector3 pmid(0,7,0);
std::cout<<"Point is Inside exactly in middle and SafetyFromOutside gives : "<<sph->SafetyFromOutside(pmid)<<std::endl;
std::cout<<"Point is Inside exactly in middle and SafetyFromInside gives : "<<sph->SafetyFromInside(pmid)<<std::endl;


std::cout<<"---------------------------------------------"<<std::endl<<std::endl;
return 0;
}
