//  UVector3 = bucket type for Vector type
// 
#include "UVector3.hh"
#include "UUtils.hh"

//______________________________________________________________________________
UVector3::UVector3( double theta, double phi )
{
// Creates a unit vector based on theta and phi angles
   x = UUtils::Sin(theta) * UUtils::Cos(phi);
   y = UUtils::Sin(theta) * UUtils::Sin(phi);
   z = UUtils::Cos(theta);
}

//______________________________________________________________________________
double UVector3::Angle(const UVector3 & q) const 
{
   // return the angle w.r.t. another 3-vector
   double ptot2 = Mag2()*q.Mag2();
   if(ptot2 <= 0) {
      return 0.0;
   } else {
      double arg = Dot(q)/UUtils::Sqrt(ptot2);
      if(arg >  1.0) arg =  1.0;
      if(arg < -1.0) arg = -1.0;
      return UUtils::ACos(arg);
   }
}

//______________________________________________________________________________
double UVector3::Mag() const 
{ 
   // return the magnitude (rho in spherical coordinate system)
   
   return UUtils::Sqrt(Mag2()); 
}

//______________________________________________________________________________
double UVector3::Perp() const 
{ 
   //return the transverse component  (R in cylindrical coordinate system)

   return UUtils::Sqrt(Perp2()); 
}

//______________________________________________________________________________
double UVector3::Phi() const 
{
   //return the  azimuth angle. returns phi from -pi to pi
   return x == 0.0 && y == 0.0 ? 0.0 : UUtils::ATan2(y,x);
}

//______________________________________________________________________________
double UVector3::Theta() const 
{
   //return the polar angle from 0 to pi
   double mag2 = Mag2();
   if (mag2 == 0.0) return 0.0;
   return UUtils::ACos(z/UUtils::Sqrt(mag2));
}

//______________________________________________________________________________
UVector3 UVector3::Unit() const 
{
   // return unit vector parallel to this.
   double  tot = Mag2();
   UVector3 p(x,y,z);
   return tot > 0.0 ? p *= (1.0/UUtils::Sqrt(tot)) : p;
}

//______________________________________________________________________________
double UVector3::Normalize()
{
   // Normalize to unit. Return normalization factor.
   double  mag = Mag2();
   if (mag == 0.0) return mag;;
   mag = UUtils::Sqrt(mag);
   x /= mag;
   y /= mag;
   z /= mag;
   return mag;
}

//______________________________________________________________________________
void UVector3::RotateX(double angle) {
   //rotate vector around X
   double s = UUtils::Sin(angle);
   double c = UUtils::Cos(angle);
   double yy = y;
   y = c*yy - s*z;
   z = s*yy + c*z;
}

//______________________________________________________________________________
void UVector3::RotateY(double angle) {
   //rotate vector around Y
   double s = UUtils::Sin(angle);
   double c = UUtils::Cos(angle);
   double zz = z;
   z = c*zz - s*x;
   x = s*zz + c*x;
}

//______________________________________________________________________________
void UVector3::RotateZ(double angle) {
   //rotate vector around Z
   double s = UUtils::Sin(angle);
   double c = UUtils::Cos(angle);
   double xx = x;
   x = c*xx - s*y;
   y = s*xx + c*y;
}

UVector3 operator + (const UVector3 & a, const UVector3 & b) {
   return UVector3(a.x + b.x, a.y + b.y, a.z + b.z);
}

UVector3 operator - (const UVector3 & a, const UVector3 & b) {
   return UVector3(a.x - b.x, a.y - b.y, a.z - b.z);
}

UVector3 operator * (const UVector3 & p, double a) {
   return UVector3(a*p.x, a*p.y, a*p.z);
}

UVector3 operator * (double a, const UVector3 & p) {
   return UVector3(a*p.x, a*p.y, a*p.z);
}

double operator * (const UVector3 & a, const UVector3 & b) {
   return a.Dot(b);
}
