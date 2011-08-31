////////////////////////////////////////////////////////////////////////////////
//
//  A simple Orb defined by half-lengths on the three axis. The center of the
//  Orb matches the origin of the local reference frame.  
//
////////////////////////////////////////////////////////////////////////////////
#include "UOrb.hh"

#include <iostream>
#include "UUtils.hh"
//______________________________________________________________________________
UOrb::UOrb(const char *name, double r)
     :VUSolid(name), fR(r)
{
  const double epsilon = 2.e-11;  // relative tolerance of fR

  // Check radius
  //
  if ( r < 10*VUSolid::fgTolerance ) // cartesian tolerance
  {
	// TODO: introduce exceptions in library

    /// G4Exception("G4Orb::G4Orb()", "InvalidSetup", FatalException, "Invalid radius > 10*kCarTolerance.");
  }
  // fRTolerance is radial tolerance (note: half of G4 tolerance)
  fRTolerance =  std::max( VUSolid::frTolerance, epsilon*r);
}

//______________________________________________________________________________
///////////////////////////////////////////////////////////////////////////
//
// Return whether point inside/outside/on surface
// Split into radius checks
// 
// Classify point location with respect to solid:
//  o eInside       - inside the solid
//  o eSurface      - close to surface within tolerance
//  o eOutside      - outside the solid

// ok
VUSolid::EnumInside UOrb::Inside(const UVector3 &p) const
{
  double rad2 = p.x*p.x+p.y*p.y+p.z*p.z;
//  if (false) double rad = std::sqrt(rad2);

  double tolRMax = fR - fRTolerance;

  // Check radial surface
  double tolRMax2 = tolRMax*tolRMax;
  if ( rad2 <= tolRMax2)  
	  return eInside; 
  else
  {
    tolRMax = fR + fRTolerance;
    if ( rad2 <= tolRMax2)
		return eSurface;
    else
		return eOutside;
  }
}


// Computes distance from a point presumably outside the solid to the solid 
// surface. Ignores first surface if the point is actually inside. Early return
// infinity in case the safety to any surface is found greater than the proposed
// step aPstep.
// The normal vector to the crossed surface is filled only in case the Orb is 
// crossed, otherwise aNormal.IsNull() is true.
//______________________________________________________________________________
double UOrb::DistanceToIn(const UVector3 &p, 
                          const UVector3 &v, 
//                          UVector3 &aNormal, 
                          double aPstep) const
{
  double snxt = UUtils::kInfinity;      // snxt = default return value

  double rad, pDotV3d; // , tolORMax2, tolIRMax2;
  double c, d2, s = UUtils::kInfinity;

  const double dRmax = 100.*fR;

  // General Precalcs

  rad    = std::sqrt(p.x*p.x + p.y*p.y + p.z*p.z);
  pDotV3d = p.x*v.x + p.y*v.y + p.z*v.z;

  // Radial Precalcs

  // tolORMax2 = (fR+fRTolerance*0.5)*(fR+fRTolerance*0.5);
  // tolIRMax2 = (fR-fRTolerance*0.5)*(fR-fRTolerance*0.5);

  // Outer spherical shell intersection
  // - Only if outside tolerant fR
  // - Check for if inside and outer G4Orb heading through solid (-> 0)
  // - No intersect -> no intersection with G4Orb
  //
  // Shell eqn: x^2+y^2+z^2 = RSPH^2
  //
  // => (px+svx)^2+(py+svy)^2+(pz+svz)^2=R^2
  //
  // => (px^2+py^2+pz^2) +2s(pxvx+pyvy+pzvz)+s^2(vx^2+vy^2+vz^2)=R^2
  // =>      rad2        +2s(pDotV3d)       +s^2                =R^2
  //
  // => s=-pDotV3d+-std::sqrt(pDotV3d^2-(rad2-R^2))

  c = (rad - fR)*(rad + fR); // c = (rad2-R^2))

  if( rad > fR-fRTolerance ) // not inside in terms of Inside(p)
  {
	  double fRTolerance2 = 2*fRTolerance;
    if ( c > fRTolerance2*fR ) // TODO: check if 2* is really needed
    {
      // If outside tolerant boundary of outer G4Orb in terms of c
      // [ should be std::sqrt(rad2) - fR > fRTolerance*0.5 ]

      d2 = pDotV3d*pDotV3d - c;

      if ( d2 >= 0 )
      {
        s = -pDotV3d - std::sqrt(d2); // ok! = [ ( -2 p dot v) +- sqrt [(-2p dot v)2 - 4*(rad - fR)*(rad + fR)] ] /  2
		// pDotV3d must be positive always, if not use alternative http://en.wikipedia.org/wiki/Quadratic_equation#Alternative_quadratic_formula
        if ( s >= 0 )
        {
          if ( s > dRmax ) // Avoid rounding errors due to precision issues seen on
          {                // 64 bits systems. Split long distances and recompute
            double fTerm = s - std::fmod(s,dRmax);
            s = fTerm + DistanceToIn(p+fTerm*v,v);
          } 
          return snxt = s;
        }
      }
      else    // No intersection with UOrb
      {
        return snxt = UUtils::kInfinity;
      }
    }
    else // not outside in terms of c
    {
      if ( c > -fRTolerance2*fR )  // on surface   TODO: check if 2* is really needed
      {
        d2 = pDotV3d*pDotV3d - c;             
        if ( (d2 < fRTolerance2*fR) || (pDotV3d >= 0) ) // pDotV3d = cos si >= 0
        {
          return snxt = UUtils::kInfinity;
        }
        else
        {
          return snxt = 0.;
        }
      }
    }
  }
#ifdef UDEBUG
  else // inside ???
  {
      G4Exception("G4Orb::DistanceToIn(p,v)", "Notification", JustWarning, "Point p is inside !?");
  }
#endif

  return snxt;
}

//______________________________________________________________________________
double UOrb::DistanceToOut( const UVector3  &p, const UVector3 &v,
			       UVector3 &n, bool &convex, double /*aPstep*/) const
{
// Computes distance from a point presumably intside the solid to the solid 
// surface. Ignores first surface along each axis systematically (for points
// inside or outside. Early returns zero in case the second surface is behind
// the starting point.
// o The proposed step is ignored.
// o The normal vector to the crossed surface is always filled.

  double snxt = 0;     // snxt: distance to next surface, is default return value 
  bool notOutside = false;
  convex = true; // orb is always convex, if we leave surface of Orb, we will neber bump on the orb again ...
  
  double rad2,pDotV3d; 
  double xi,yi,zi;      // Intersection point
  double c,d2;
                 
  rad2    = p.x*p.x + p.y*p.y + p.z*p.z;
  pDotV3d = p.x*v.x + p.y*v.y + p.z*v.z;
    
  // Radial Intersection from UOrb::DistanceToIn
  //
  // Outer spherical shell intersection
  // - Only if outside tolerant fR
  // - Check for if inside and outer UOrb heading through solid (-> 0)
  // - No intersect -> no intersection with UOrb
  //
  // Shell eqn: x^2+y^2+z^2=RSPH^2
  //
  // => (px+svx)^2+(py+svy)^2+(pz+svz)^2=R^2
  //
  // => (px^2+py^2+pz^2) +2s(pxvx+pyvy+pzvz)+s^2(vx^2+vy^2+vz^2)=R^2
  // =>      rad2        +2s(pDotV3d)       +s^2                =R^2
  //
  // => s=-pDotV3d+-std::sqrt(pDotV3d^2-(rad2-R^2))
  
  const double rPlus = fR + fRTolerance;
  double rad = std::sqrt(rad2);

  if ( rad <= rPlus )
  {
    c = (rad - fR)*(rad + fR); // rad2 - fR2

    if ( c < fRTolerance*fR ) 
    {
      // Within tolerant Outer radius 
      // 
      // The test is
      //     rad  - fR < 0.5*fRTolerance
      // =>  rad  < fR + 0.5*kRadTol
      // =>  rad2 < (fR + 0.5*kRadTol)^2
      // =>  rad2 < fR^2 + 2.*0.5*fR*kRadTol + 0.25*kRadTol*kRadTol
      // =>  rad2 - fR^2    <~    fR*kRadTol 

      d2 = pDotV3d*pDotV3d - c;

      if( ( c > -2*fRTolerance*fR) &&         // => point is on tolerant surface (i.e. within +- tolerance)
          ( ( pDotV3d >= 0 )   || ( d2 < 0 )) )   // if (pDotV3d >= 0 ) => leaving outside from Rmax; i.e. from surface
                                                  // not re-entering
												  // if (d2 < 0) => it means the point is already outside
      {
        // if(calcNorm) // NOTE: we do not have this variable, calcNorm is true always
        {
//          *validNorm = true; // NOTE: we do not have this variable, probably always true
          n = UVector3(p.x/fR,p.y/fR,p.z/fR);
        }
        return snxt = 0;
      }
      else 
      {
		 // we are inside, with + version of quadratic eq. solution we calculate solution for distance
        snxt = -pDotV3d + std::sqrt(d2);    // second root since inside Rmax
											// the solution is safe because pDotV3d is negative
											// c alternative formula, see http://en.wikipedia.org/wiki/Quadratic_equation#Alternative_quadratic_formula
											// is not neccessary in this case
        notOutside = true; 
      }
    }
  }
  else // p is outside ???
  {
    std::cout.precision(16);
    std::cout << std::endl;
//    DumpInfo();
    std::cout << "Position:"  << std::endl << std::endl;
    std::cout << "p.x() = "   << p.x << std::endl;
    std::cout << "p.y() = "   << p.y << std::endl;
    std::cout << "p.z() = "   << p.z << std::endl << std::endl;
    std::cout << "Rp = "<< std::sqrt( p.x*p.x+p.y*p.y+p.z*p.z) << std::endl << std::endl;
    std::cout << "Direction:" << std::endl << std::endl;
    std::cout << "v.x() = "   << v.x << std::endl;
    std::cout << "v.y() = "   << v.y << std::endl;
    std::cout << "v.z() = "   << v.z << std::endl << std::endl;
    std::cout << "Proposed distance :" << std::endl << std::endl;
    std::cout << "snxt = "    << snxt << std::endl << std::endl;

	/*
    std::cout.precision(6);
    G4Exception("G4Orb::DistanceToOut(p,v,..)", "Notification",
                JustWarning, "Logic error: snxt = kInfinity ???");
	****/
  }
  
  // if (calcNorm)    // Output switch operator
  {
    if ( notOutside )
    {
        xi=p.x+snxt*v.x; // we move to the point on surface, then return normal at that point which for orb, see method bool UOrb::Normal( const UVector3& p, UVector3 &n)
        yi=p.y+snxt*v.y;
        zi=p.z+snxt*v.z;
        n=UVector3(xi/fR,yi/fR,zi/fR); // we return normalized vector
	}
	else
	{
        std::cout.precision(16);
        std::cout << std::endl;
//        DumpInfo();
        std::cout << "Position:"  << std::endl << std::endl;
        std::cout << "p.x() = "   << p.x << " mm" << std::endl;
        std::cout << "p.y() = "   << p.y << " mm" << std::endl;
        std::cout << "p.z() = "   << p.z << " mm" << std::endl << std::endl;
        std::cout << "Direction:" << std::endl << std::endl;
        std::cout << "v.x() = "   << v.x << std::endl;
        std::cout << "v.y() = "   << v.y << std::endl;
        std::cout << "v.z() = "   << v.z << std::endl << std::endl;
        std::cout << "Proposed distance :" << std::endl << std::endl;
        std::cout << "snxt = "    << snxt << " mm" << std::endl << std::endl;
        std::cout.precision(6);
//        G4Exception("G4Orb::DistanceToOut(p,v,..)","Notification",JustWarning, "Undefined side for valid surface normal to solid.");
    }
  }
  return snxt;
}

// Estimates the isotropic safety from a point inside the current solid to any 
// of its surfaces. The algorithm may be accurate or should provide a fast 
// underestimate.
//______________________________________________________________________________
// Note: In geant4, these methods are DistanceToOut, without given direction
// Note: ??? Should not Return 0 anymore if point outside, just the value
// OK
double UOrb::SafetyFromInside ( const UVector3 p, bool /*aAccurate*/) const
{

/////////////////////////////////////////////////////////////////////////
//
// Calculate distance (<=actual) to closest surface of shape from inside

  double safe=0.0,rad = std::sqrt(p.x*p.x+p.y*p.y+p.z*p.z);

#ifdef UDEBUG
  if( Inside(p) == kOutside )
  {
//     int oldprc = std::cout.precision(16);
     std::cout << std::endl;
//     DumpInfo();
     std::cout << "Position:"  << std::endl << std::endl;
     std::cout << "p.x = "   << p.x << std::endl;
     std::cout << "p.y = "   << p.y << std::endl;
     std::cout << "p.z = "   << p.z << std::endl << std::endl;
//     std::cout.precision(oldprc);
     G4Exception("G4Orb::DistanceToOut(p)", "Notification", JustWarning, 
                 "Point p is outside !?" );
  }
#endif

  safe = fR - rad;
  if ( safe < 0. ) safe = 0.;
  return safe;
}


// Estimates the isotropic safety from a point outside the current solid to any 
// of its surfaces. The algorithm may be accurate or should provide a fast 
// underestimate.
// Note: In geant4, this method is equivalent to DistanceToIn, without given direction
//______________________________________________________________________________
//////////////////////////////////////////////////////////////////////
//
// Calculate distance (<= actual) to closest surface of shape from outside
// - Calculate distance to radial plane
// - Return 0 if point inside
// OK
double UOrb::SafetyFromOutside ( const UVector3 p, bool aAccurate) const
{
    double safe = 0.0;
    double rad  = std::sqrt(p.x*p.x+p.y*p.y+p.z*p.z);
	safe = rad - fR;
	if( safe < 0 ) { safe = 0.; }
	return safe;
}


/////////////////////////////////////////////////////////////////////
//
// Return unit normal of surface closest to p
//______________________________________________________________________________
// 
// From http://lists.trolltech.com/qt-interest/2002-09/thread01124-0.html :
//> does anybody here have an algorithm to calculate the normal vector in a
//> given point in space (x, y, z) in a sphere? I know that it's not about qt
//> but i'll like very mutch the help.
// It's simply the connecting vector from the centre of the sphere to the point 
// (other way around for inward normals) obtained through vector subtraction, 
// normalized to unity.
//
// You really should get an algebra book though, as you are bound to encounter 
// more of these problems in a 3d application.
//
// OK
bool UOrb::Normal( const UVector3& p, UVector3 &n)
{
	double rad2 = p.x*p.x+p.y*p.y+p.z*p.z;
	double rad = std::sqrt(rad2);

	n = UVector3(p.x/rad,p.y/rad,p.z/rad);

	double tolRMaxP = fR + fRTolerance;
	double tolRMaxM = fR - fRTolerance;

	// Check radial surface
	bool result = (( rad2 <= tolRMaxP*tolRMaxP ) && ( rad2 >= tolRMaxM*tolRMaxM)); // means we are on surface
	return result;
}

   
// Returns extent of the solid along a given cartesian axis
// OK
//______________________________________________________________________________
void UOrb::Extent( EAxisType aAxis, double &aMin, double &aMax )
{
	switch (aAxis)
	{
      case eXaxis:
      case eYaxis:
      case eZaxis:
		 aMin = -fR; aMax = fR;
         break;
      default:
         std::cout << "Extent: unknown axis" << aAxis << std::endl;
   }      
}            

// Returns the full 3D cartesian extent of the solid.
// OK
//______________________________________________________________________________
void UOrb::Extent ( double aMin[3], double aMax[3] )
{
   aMin[0] = aMin[1] = aMin[2] = -fR;
   aMax[0] = aMax[1] = aMax[2] = fR;
}
