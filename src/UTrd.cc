
/**
* @file
* @author  John Doe <jdoe@example.com>
* @version 0.1
*
* @section LICENSE
*
* @section DESCRIPTION
*
* A simple Orb defined by half-lengths on the three axis. The center of the Orb matches the origin of the local reference frame.
*/
 
#include <iostream>
#include <cmath>

#include "UTrd.hh"
#include "UUtils.hh"

// OK: inside => use geant4
// normal => integrate root which is slightly faster, if it is possible to do easily
// OK: safety*, integrate root for accurate case, for start we can use two versions
// use distances from root

inline double amin(int n, const double *a) {
	// Return value from array with the minimum element.
	double xmin = a[0];
	for  (int i = 1; i < n; i++) 
	{
		if (xmin > a[i]) xmin = a[i];
	}
	return xmin;
}

inline double amax(int n, const double *a) {
	// Return value from array with the maximum element.
	double xmax = a[0];
	for  (int i = 1; i < n; i++)
	{
		if (xmax < a[i]) xmax = a[i];
	}
	return xmax;
}



/*
* Estimates the isotropic safety from a point inside the current solid to any 
* of its surfaces. The algorithm may be accurate or should provide a fast 
* underestimate.
* ______________________________________________________________________________
* Note: In geant4, these methods are DistanceToOut, without given direction
* Note: ??? Should not Return 0 anymore if point outside, just the value
* OK
*/

// Geant4 version is faster according to the profiler


double UTrd::SafetyFromInside ( const UVector3 &p, bool aAccurate) const
{
	if (aAccurate) return SafetyFromInsideAccurate(p);

	double safe, zbase, tanxz, xdist, saf1, tanyz, ydist, saf2;

#ifdef G4CSGDEBUG
	if( Inside(p) == kOutside )
	{
		G4int oldprc = G4cout.precision(16) ;
		G4cout << G4endl ;
		DumpInfo();
		G4cout << "Position:"  << G4endl << G4endl ;
		G4cout << "p.x() = "   << p.x()/mm << " mm" << G4endl ;
		G4cout << "p.y() = "   << p.y()/mm << " mm" << G4endl ;
		G4cout << "p.z() = "   << p.z()/mm << " mm" << G4endl << G4endl ;
		G4cout.precision(oldprc) ;
		G4Exception("G4Trd::DistanceToOut(p)", "Notification", JustWarning, 
			"Point p is outside !?" );
	}
#endif

	safe=fDz-std::fabs(p.z);  // z perpendicular Dist

	zbase=fDz+p.z;

	// xdist = distance perpendicular to z axis to closest x plane from p
	//       = (x half width of shape at p.z) - std::abs(p.x)
	//
	tanxz=(fDx2-fDx1)*0.5/fDz; // angle between two parts on trinngle, which reflects proportional part on triangle related to position of point, see it on picture
	xdist=fDx1+tanxz*zbase-std::fabs(p.x); // distance to closest point on x border on x-axis on which point is located to
	saf1=xdist/std::sqrt(1.0+tanxz*tanxz); // x*std::cos(ang_xz) =
	// shortest (perpendicular)
	// distance to plane, see picture
	tanyz=(fDy2-fDy1)*0.5/fDz;
	ydist=fDy1+tanyz*zbase-std::fabs(p.y);
	saf2=ydist/std::sqrt(1.0+tanyz*tanyz);

	// Return minimum x/y/z distance
	//
	if (safe>saf1) safe=saf1;
	if (safe>saf2) safe=saf2;

	if (safe<0) safe=0;
	return safe;     
}



inline double UTrd::SafetyFromInsideAccurate ( const UVector3 &p) const
{
	// computes the closest distance from given point to this shape, according
	// to option. The matching point on the shape is stored in spoint.

	double saf[3];
	//--- Compute safety first
	// check Z facettes
	saf[0] = fDz-std::abs(p.z);
	double fx = 0.5*(fDx1-fDx2)/fDz;
	double calf = 1./std::sqrt(1.0+fx*fx);
	// check X facettes
	double distx = 0.5*(fDx1+fDx2)-fx*p.z;
	if (distx<0) saf[1]=UUtils::kInfinity;
	else         saf[1]=(distx-std::abs(p.x))*calf;

	double fy = 0.5*(fDy1-fDy2)/fDz;
	calf = 1./std::sqrt(1.0+fy*fy);
	// check Y facettes
	distx = 0.5*(fDy1+fDy2)-fy*p.z;
	if (distx<0) saf[2]=UUtils::kInfinity;
	else         saf[2]=(distx-std::abs(p.y))*calf;

	return amin(3,saf);
}

/*
* Estimates the isotropic safety from a point outside the current solid to any 
* of its surfaces. The algorithm may be accurate or should provide a fast 
* underestimate.
* Note: In geant4, this method is equivalent to DistanceToIn, without given direction
* ______________________________________________________________________________
*
* Calculate distance (<= actual) to closest surface of shape from outside
* - Calculate distance to radial plane
* - Return 0 if point inside
* OK
*/
double UTrd::SafetyFromOutside ( const UVector3 &p, bool aAccurate) const
{
	if (aAccurate) return SafetyFromOutsideAccurate(p);

	double safe=0.0;
	double tanxz,distx,safx;
	double tanyz,disty,safy;
	double zbase;

	safe=std::abs(p.z)-fDz;
	if (safe<0) safe=0;      // Also used to ensure x/y distances
	// POSITIVE
	zbase=fDz+p.z;

	// Find distance along x direction to closest x plane
	//
	tanxz=(fDx2-fDx1)*0.5/fDz;
	//    widx=fDx1+tanxz*(fDz+p.z()); // x width at p.z
	//    distx=std::abs(p.x())-widx;      // distance to plane
	distx=std::abs(p.x)-(fDx1+tanxz*zbase); // distance to point on border of trd, related to axis on which point p lies
	if (distx>safe)
	{
		safx=distx/std::sqrt(1.0+tanxz*tanxz); // perpendicular distance calculation; vector Dist=Dist*std::cos(ang), it can be probably negative, then comparing in next statement, we will get rid of such distance, because it directs away from the solid
		if (safx>safe) safe=safx;
	}

	// Find distance along y direction to slanted wall
	tanyz=(fDy2-fDy1)*0.5/fDz;
	//    widy=fDy1+tanyz*(fDz+p.z()); // y width at p.z
	//    disty=std::abs(p.y())-widy;      // distance to plane
	disty=std::abs(p.y)-(fDy1+tanyz*zbase);
	if (disty>safe)    
	{
		safy=disty/std::sqrt(1.0+tanyz*tanyz); // distance along vector
		if (safy>safe) safe=safy;
	}
	return safe;
}

inline double UTrd::SafetyFromOutsideAccurate ( const UVector3 &p) const
{
	// computes the closest distance from given point to this shape, according
	// to option. The matching point on the shape is stored in spoint.

	double saf[3];
	//--- Compute safety first
	// check Z facettes
	saf[0] = fDz-std::abs(p.z);
	double fx = 0.5*(fDx1-fDx2)/fDz;
	double calf = 1./std::sqrt(1.0+fx*fx);
	// check X facettes
	double distx = 0.5*(fDx1+fDx2)-fx*p.z;
	if (distx<0) saf[1]=UUtils::kInfinity;
	else         saf[1]=(distx-std::abs(p.x))*calf;

	double fy = 0.5*(fDy1-fDy2)/fDz;
	calf = 1./std::sqrt(1.0+fy*fy);
	// check Y facettes
	distx = 0.5*(fDy1+fDy2)-fy*p.z;
	if (distx<0) saf[2]=UUtils::kInfinity;
	else         saf[2]=(distx-std::abs(p.y))*calf;

	for (int i=0; i<3; i++) saf[i]=-saf[i];
	return amax(3,saf);
}




//dx1 	Half-length along x at the surface positioned at -dz
//dx2 	Half-length along x at the surface positioned at +dz
//dy1 	Half-length along y at the surface positioned at -dz
//dy2 	Half-length along y at the surface positioned at +dz
//dz 	Half-length along z axis 
//______________________________________________________________________________
UTrd::UTrd(const char* pName,
	double pdx1, double pdx2,
	double pdy1, double pdy2,
	double pdz)
	: VUSolid(pName)
{
	if ( pdx1>0&&pdx2>0&&pdy1>0&&pdy2>0&&pdz>0 )
	{
		fDx1=pdx1; fDx2=pdx2;
		fDy1=pdy1; fDy2=pdy2;
		fDz=pdz;
	}
	else
	{
		if ( pdx1>=0 && pdx2>=0 && pdy1>=0 && pdy2>=0 && pdz>=0 )
		{
			// double  Minimum_length= (1+per_thousand) * VUSolid::fgTolerance/2.;
			// FIX-ME : temporary solution for ZERO or very-small parameters
			//
			double  Minimum_length= fgTolerance;
			fDx1=std::max(pdx1,Minimum_length); 
			fDx2=std::max(pdx2,Minimum_length); 
			fDy1=std::max(pdy1,Minimum_length); 
			fDy2=std::max(pdy2,Minimum_length); 
			fDz=std::max(pdz,Minimum_length);
		}
		else
		{
			std::cout << "ERROR - G4Trd()::CheckAndSetAllParameters(): " << GetName()
				<< std::endl
				<< "        Invalid dimensions, some are < 0 !" << std::endl
				<< "          X - " << pdx1 << ", " << pdx2 << std::endl
				<< "          Y - " << pdy1 << ", " << pdy2 << std::endl
				<< "          Z - " << pdz << std::endl;
			// G4Exception("G4Trd::CheckAndSetAllParameters()", "InvalidSetup", FatalException, "Invalid parameters.");
		}
	}
}

//______________________________________________________________________________
/**
*
* Return whether point inside/outside/on surface
* Split into radius checks
* 
* Classify point location with respect to solid:
*  o eInside       - inside the solid
*  o eSurface      - close to surface within tolerance
*  o eOutside      - outside the solid
*/
// ok
// Geant4 routine was selected because:
// 1. it is able to detect if the point is on surface
// 2. it counts with tolerance
// 3. algorithm checks are similar:
// Geant4: x=0.5*(fDx2*zbase1+fDx1*zbase2)/fDz - fgTolerance/2;
// Root: double dx = 0.5*(fDx2*(point.z+fDz)+fDx1*(fDz-point.z))/fDz;
//
VUSolid::EnumInside UTrd::Inside(const UVector3 &p) const
{
	VUSolid::EnumInside in=eOutside;
	double x,y,zbase1,zbase2;

	if (std::abs(p.z)<=fDz-fgTolerance/2)
	{
		zbase1=p.z+fDz;  // Dist from -ve z plane
		zbase2=fDz-p.z;  // Dist from +ve z plane

		// Check whether inside x tolerance
		//
		x=0.5*(fDx2*zbase1+fDx1*zbase2)/fDz - fgTolerance/2; // calculate x coordinate of one corner point corresponding to p.z inside trd (on x axis), ... by using proportional calculation related to triangle
		if (std::abs(p.x)<=x)
		{
			y=0.5*((fDy2*zbase1+fDy1*zbase2))/fDz - fgTolerance/2;
			if (std::abs(p.y)<=y)
			{
				in=eInside;
			}
			else if (std::abs(p.y)<=y+fgTolerance)
			{
				in=eSurface;
			}
		}
		else if (std::abs(p.x)<=x+fgTolerance)
		{
			// y = y half width of shape at z of point + tolerant boundary
			//
			y=0.5*((fDy2*zbase1+fDy1*zbase2))/fDz + fgTolerance/2;
			if (std::abs(p.y)<=y)
			{
				in=eSurface;
			}  
		}
	}
	else if (std::abs(p.z)<=fDz+fgTolerance/2)
	{
		// Only need to check outer tolerant boundaries
		//
		zbase1=p.z+fDz;  // Dist from -ve z plane
		zbase2=fDz-p.z;   // Dist from +ve z plane

		// x = x half width of shape at z of point plus tolerance
		//
		x=0.5*(fDx2*zbase1+fDx1*zbase2)/fDz + fgTolerance/2;
		if (std::abs(p.x)<=x)
		{
			// y = y half width of shape at z of point
			//
			y=0.5*((fDy2*zbase1+fDy1*zbase2))/fDz + fgTolerance/2;
			if (std::abs(p.y)<=y) in=eSurface;
		}
	}  
	return in;
}


/*
* Computes distance from a point presumably outside the solid to the solid 
* surface. Ignores first surface if the point is actually inside. Early return
* infinity in case the safety to any surface is found greater than the proposed
* step aPstep.
* The normal vector to the crossed surface is filled only in case the Orb is 
* crossed, otherwise aNormal.IsNull() is true.
*/
double UTrd::DistanceToIn(const UVector3 &p, 
	const UVector3 &v, 
	//                          UVector3 &aNormal, 
	double) const
{
	double snxt = UUtils::kInfinity;    // snxt = default return value
	double smin,smax;
	double s1,s2,tanxz,tanyz,ds1,ds2;
	double ss1,ss2,sn1=0.,sn2=0.,dist;

	if ( v.z )  // Calculate valid z intersect range
	{
		if ( v.z > 0 )   // Calculate smax: must be +ve or no intersection.
		{
			dist = fDz - p.z ;  // to plane at +dz

			if (dist >= 0.5*VUSolid::fgTolerance)
			{
				smax = dist/v.z; // distance to intersection with +dz, maximum
				smin = -(fDz + p.z)/v.z; // distance to intersection with +dz, minimum
			}
			else  return snxt ;
		}
		else // v.z <0
		{
			dist=fDz+p.z;  // plane at -dz

			if ( dist >= 0.5*VUSolid::fgTolerance )
			{
				smax = -dist/v.z;
				smin = (fDz - p.z)/v.z;
			} 
			else return snxt ; 
		}
		if (smin < 0 ) smin = 0 ;
	}
	else // v.z=0
	{
		if (std::abs(p.z) >= fDz ) return snxt ;     // Outside & no intersect
		else
		{
			smin = 0 ;    // Always inside z range
			smax = UUtils::kInfinity;
		}
	}

	// Calculate x intersection range
	//
	// Calc half width at p.z, and components towards planes

	tanxz = (fDx2 - fDx1)*0.5/fDz ;
	s1    = 0.5*(fDx1+fDx2) + tanxz*p.z ;  // x half width at p.z
	ds1   = v.x - tanxz*v.z ;       // Components of v towards faces at +-x
	ds2   = v.x + tanxz*v.z ;
	ss1   = s1 - p.x;         // -delta x to +ve plane
	// -ve when outside
	ss2   = -s1 - p.x;        // -delta x to -ve plane
	// +ve when outside
	if (ss1 < 0 && ss2 <= 0 )
	{
		if (ds1 < 0)   // In +ve coord Area
		{
			sn1 = ss1/ds1 ;

			if ( ds2 < 0 ) sn2 = ss2/ds2 ;           
			else           sn2 = UUtils::kInfinity ;
		}
		else return snxt ;
	}
	else if ( ss1 >= 0 && ss2 > 0 )
	{
		if ( ds2 > 0 )  // In -ve coord Area
		{
			sn1 = ss2/ds2 ;

			if (ds1 > 0)  sn2 = ss1/ds1 ;      
			else          sn2 = UUtils::kInfinity;      

		}
		else   return snxt ;
	}
	else if (ss1 >= 0 && ss2 <= 0 )
	{
		// Inside Area - calculate leaving distance
		// *Don't* use exact distance to side for tolerance
		//                                             = ss1*std::cos(ang xz)
		//                                             = ss1/std::sqrt(1.0+tanxz*tanxz)
		sn1 = 0 ;

		if ( ds1 > 0 )
		{
			if (ss1 > 0.5*VUSolid::fgTolerance) sn2 = ss1/ds1 ; // Leave +ve side extent
			else                         return snxt ;   // Leave immediately by +ve 
		}
		else  sn2 = UUtils::kInfinity ;

		if ( ds2 < 0 )
		{
			if ( ss2 < -0.5*VUSolid::fgTolerance )
			{
				dist = ss2/ds2 ;            // Leave -ve side extent
				if (dist < sn2) sn2 = dist ;
			}
			else return snxt ;
		}    
	}
	else if (ss1 < 0 && ss2 > 0 )
	{
		// Within +/- plane cross-over areas (not on boundaries ss1||ss2==0)

		if ( ds1 >= 0 || ds2 <= 0 )
		{   
			return snxt ;
		}
		else  // Will intersect & stay inside
		{
			sn1  = ss1/ds1 ;
			dist = ss2/ds2 ;
			if (dist > sn1 ) sn1 = dist ;
			sn2 = UUtils::kInfinity ;
		}
	}

	// Reduce allowed range of distances as appropriate

	if ( sn1 > smin ) smin = sn1 ;
	if ( sn2 < smax ) smax = sn2 ;

	// Check for incompatible ranges (eg z intersects between 50 ->100 and x
	// only 10-40 -> no intersection)

	if ( smax < smin ) return snxt ;

	// Calculate valid y intersection range 
	// (repeat of x intersection code)

	tanyz = (fDy2-fDy1)*0.5/fDz ;
	s2    = 0.5*(fDy1+fDy2) + tanyz*p.z;  // y half width at p.z
	ds1   = v.y - tanyz*v.z;       // Components of v towards faces at +-y
	ds2   = v.y + tanyz*v.z;
	ss1   = s2 - p.y;         // -delta y to +ve plane
	ss2   = -s2 - p.y;        // -delta y to -ve plane

	if ( ss1 < 0 && ss2 <= 0 )
	{
		if (ds1 < 0 ) // In +ve coord Area
		{
			sn1 = ss1/ds1 ;
			if ( ds2 < 0 )  sn2 = ss2/ds2 ;
			else            sn2 = UUtils::kInfinity ;
		}
		else   return snxt ;
	}
	else if ( ss1 >= 0 && ss2 > 0 )
	{
		if ( ds2 > 0 )  // In -ve coord Area
		{
			sn1 = ss2/ds2 ;
			if ( ds1 > 0 )  sn2 = ss1/ds1 ;
			else            sn2 = UUtils::kInfinity ;      
		}
		else   return snxt ;
	}
	else if (ss1 >= 0 && ss2 <= 0 )
	{
		// Inside Area - calculate leaving distance
		// *Don't* use exact distance to side for tolerance
		//                                          = ss1*std::cos(ang yz)
		//                                          = ss1/std::sqrt(1.0+tanyz*tanyz)
		sn1 = 0 ;

		if ( ds1 > 0 )
		{
			if (ss1 > 0.5*VUSolid::fgTolerance) sn2 = ss1/ds1 ; // Leave +ve side extent
			else                         return snxt ;   // Leave immediately by +ve
		}
		else  sn2 = UUtils::kInfinity ;

		if ( ds2 < 0 )
		{
			if ( ss2 < -0.5*VUSolid::fgTolerance )
			{
				dist = ss2/ds2 ; // Leave -ve side extent
				if (dist < sn2) sn2=dist;
			}
			else return snxt ;
		}    
	}
	else if (ss1 < 0 && ss2 > 0 )
	{
		// Within +/- plane cross-over areas (not on boundaries ss1||ss2==0)

		if (ds1 >= 0 || ds2 <= 0 )  
		{
			return snxt ;
		}
		else  // Will intersect & stay inside
		{
			sn1 = ss1/ds1 ;
			dist = ss2/ds2 ;
			if (dist > sn1 ) sn1 = dist ;
			sn2 = UUtils::kInfinity ;
		}
	}

	// Reduce allowed range of distances as appropriate

	if ( sn1 > smin) smin = sn1 ;
	if ( sn2 < smax) smax = sn2 ;

	// Check for incompatible ranges (eg x intersects between 50 ->100 and y
	// only 10-40 -> no intersection). Set snxt if ok

	if ( smax > smin ) snxt = smin ;
	if (snxt < 0.5*VUSolid::fgTolerance ) snxt = 0.0 ;

	return snxt ;
}



//_____________________________________________________________________________
// double TGeoTrd2::DistFromOutside(double *point, double *dir, int iact, double step, double *safe) const

// root routine is faster, but it gives wrong results on surface points. therefore, we use currently
// Geant4 version

inline double UTrd::DistanceToInRoot(const UVector3 &point, const UVector3 &dir, double) const
{
	// Compute distance from outside point to surface of the trd2
	// Boundary safe algorithm
	double snxt = UUtils::kInfinity;

	/*
	if (iact<3 && safe) {
	// compute safe distance
	*safe = SafetyFromOutside(point);
	if (iact==0) return UUtils::kInfinity;
	if (iact==1 && step<*safe) return UUtils::kInfinity;
	}
	*/

	// find a visible face
	double xnew,ynew,znew;
	double fx = 0.5*(fDx1-fDx2)/fDz;
	double fy = 0.5*(fDy1-fDy2)/fDz;
	double cn;
	// check visibility of X faces
	double distx = 0.5*(fDx1+fDx2)-fx*point.z;
	double disty = 0.5*(fDy1+fDy2)-fy*point.z;
	bool in = true;
	double safx = distx-std::abs(point.x);
	double safy = disty-std::abs(point.y);
	double safz = fDz-std::abs(point.z);
	//--- Compute distance to this shape
	// first check if Z facettes are crossed
	if (point.z<=-fDz) {
		cn = -dir.z;
		if (cn>=0) return UUtils::kInfinity;
		in = false;
		snxt = (fDz+point.z)/cn;	
		// find extrapolated X and Y
		xnew = point.x+snxt*dir.x;
		if (std::abs(xnew) < fDx1) {
			ynew = point.y+snxt*dir.y;
			if (std::abs(ynew) < fDy1) return snxt;
		}
	} else if (point.z>=fDz) {
		cn = dir.z;
		if (cn>=0) return UUtils::kInfinity;
		in = false;
		snxt = (fDz-point.z)/cn;
		// find extrapolated X and Y
		xnew = point.x+snxt*dir.x;
		if (std::abs(xnew) < fDx2) {
			ynew = point.y+snxt*dir.y;
			if (std::abs(ynew) < fDy2) return snxt;
		}
	}
	// check if X facettes are crossed
	if (point.x<=-distx) {
		cn = -dir.x+fx*dir.z;
		if (cn>=0) return UUtils::kInfinity;
		in = false;
		snxt = (point.x+distx)/cn;
		// find extrapolated Y and Z
		znew = point.z+snxt*dir.z;
		if (std::abs(znew) < fDz) {
			double dy = 0.5*(fDy1+fDy2)-fy*znew;
			ynew = point.y+snxt*dir.y;
			if (std::abs(ynew) < dy) return snxt;
		}
	}            
	if (point.x>=distx) {
		cn = dir.x+fx*dir.z;
		if (cn>=0) return UUtils::kInfinity;
		in = false;
		snxt = (distx-point.x)/cn;
		// find extrapolated Y and Z
		znew = point.z+snxt*dir.z;
		if (std::abs(znew) < fDz) {
			double dy = 0.5*(fDy1+fDy2)-fy*znew;
			ynew = point.y+snxt*dir.y;
			if (std::abs(ynew) < dy) return snxt;
		}
	}
	// finally check Y facettes
	if (point.y<=-disty) {
		cn = -dir.y+fy*dir.z;
		in = false;
		if (cn>=0) return UUtils::kInfinity;
		snxt = (point.y+disty)/cn;
		// find extrapolated X and Z
		znew = point.z+snxt*dir.z;
		if (std::abs(znew) < fDz) {
			double dx = 0.5*(fDx1+fDx2)-fx*znew;
			xnew = point.x+snxt*dir.x;
			if (std::abs(xnew) < dx) return snxt;
		}
	}            
	if (point.y>=disty) {
		cn = dir.y+fy*dir.z;
		if (cn>=0) return UUtils::kInfinity;
		in = false;
		snxt = (disty-point.y)/cn;
		// find extrapolated X and Z
		znew = point.z+snxt*dir.z;
		if (std::abs(znew) < fDz) {
			double dx = 0.5*(fDx1+fDx2)-fx*znew;
			xnew = point.x+snxt*dir.x;
			if (std::abs(xnew) < dx) return snxt;
		}
	}
	if (!in) return UUtils::kInfinity;

	//   TODO: Inside points:  If a point direction dot normal of surface is negative return 0, if outwards (dot product positive), check another surface

	// Point actually inside
	if (safz<safx && safz<safy) {
		if (point.z*dir.z>=0) return UUtils::kInfinity;
		return 0.0;
	}
	if (safy<safx) {
		cn = UUtils::Sign(1.0,point.y)*dir.y+fy*dir.z;     
		if (cn>=0) return UUtils::kInfinity;
		return 0.0;
	}   
	cn = UUtils::Sign(1.0,point.x)*dir.x+fx*dir.z;     
	if (cn>=0) return UUtils::kInfinity;

	return 0.0;      
}


/*
* Computes distance from a point presumably intside the solid to the solid 
* surface. Ignores first surface along each axis systematically (for points
* inside or outside. Early returns zero in case the second surface is behind
* the starting point.
* o The proposed step is ignored.
* o The normal vector to the crossed surface is always filled.
* ______________________________________________________________________________
*/

#define normal

inline double UTrd::DistanceToOut( const UVector3  &p, const UVector3 &v,
	UVector3 &n, bool & /*convex*/ /* not used */, double /*aPstep*/) const
{
#ifdef normal
	ESide side = kUndefined, snside = kUndefined;
#endif

	double snxt,pdist;
	double central,ss1,ss2,ds1,ds2,sn=0.,sn2=0.;
	double tanxz=0.,cosxz=0.,tanyz=0.,cosyz=0.;

	// Calculate z plane intersection
	if (v.z>0)
	{
		pdist=fDz-p.z;
		if (pdist>VUSolid::fgTolerance/2)
		{
			snxt=pdist/v.z;
#ifdef normal
			side=kPZ;
#endif
		}
		else
		{
			n=UVector3(0,0,1);
			return snxt=0;
		}
	}
	else if (v.z<0) 
	{
		pdist=fDz+p.z;
		if (pdist>VUSolid::fgTolerance/2)
		{
			snxt=-pdist/v.z;
#ifdef normal
			side=kMZ;
#endif
		}
		else
		{
			//      if (calcNorm)
			{
				n=UVector3(0,0,-1);
			}
			return snxt=0;
		}
	}
	else
	{
		snxt=UUtils::kInfinity;
	}

	//
	// Calculate x intersection
	//
	tanxz=(fDx2-fDx1)*0.5/fDz;
	central=0.5*(fDx1+fDx2);

	// +ve plane (1)
	//
	ss1=central+tanxz*p.z-p.x;  // distance || x axis to plane
	// (+ve if point inside)
	ds1=v.x-tanxz*v.z;    // component towards plane at +x
	// (-ve if +ve -> -ve direction)
	// -ve plane (2)
	//
	ss2=-tanxz*p.z-p.x-central;  //distance || x axis to plane
	// (-ve if point inside)
	ds2=tanxz*v.z+v.x;    // component towards plane at -x

	if (ss1>0&&ss2<0)
	{
		// Normal case - entirely inside region
		if (ds1<=0&&ds2<0)
		{   
			if (ss2<-VUSolid::fgTolerance/2)
			{
				sn=ss2/ds2;  // Leave by -ve side
#ifdef normal
				snside=kMX;
#endif
			}
			else
			{
				sn=0; // Leave immediately by -ve side
#ifdef normal
				snside=kMX;
#endif
			}
		}
		else if (ds1>0&&ds2>=0)
		{
			if (ss1>VUSolid::fgTolerance/2)
			{
				sn=ss1/ds1;  // Leave by +ve side
#ifdef normal
				snside=kPX;
#endif
			}
			else
			{
				sn=0; // Leave immediately by +ve side
#ifdef normal
				snside=kPX;
#endif // normal
			}
		}
		else if (ds1>0&&ds2<0)
		{
			if (ss1>VUSolid::fgTolerance/2)
			{
				// sn=ss1/ds1;  // Leave by +ve side
				if (ss2<-VUSolid::fgTolerance/2)
				{
					sn=ss1/ds1;  // Leave by +ve side
					sn2=ss2/ds2;
					if (sn2<sn)
					{
						sn=sn2;
#ifdef normal
						snside=kMX;
#endif
					}
					else
					{
#ifdef normal
						snside=kPX;
#endif
					}
				}
				else
				{
					sn=0; // Leave immediately by -ve
#ifdef normal
					snside=kMX;
#endif
				}      
			}
			else
			{
				sn=0; // Leave immediately by +ve side
#ifdef normal
				snside=kPX;
#endif
			}
		}
		else
		{
			// Must be || to both
			//
			sn=UUtils::kInfinity;    // Don't leave by either side
		}
	}
	else if (ss1<=0&&ss2<0)
	{
		// Outside, in +ve Area

		if (ds1>0)
		{
			sn=0;       // Away from shape
			// Left by +ve side
#ifdef normal
			snside=kPX;
#endif
		}
		else
		{
			if (ds2<0)
			{
				// Ignore +ve plane and use -ve plane intersect
				//
				sn=ss2/ds2; // Leave by -ve side
#ifdef normal
				snside=kMX;
#endif
			}
			else
			{
				// Must be || to both -> exit determined by other axes
				//
				sn=UUtils::kInfinity; // Don't leave by either side
			}
		}
	}
	else if (ss1>0&&ss2>=0)
	{
		// Outside, in -ve Area

		if (ds2<0)
		{
			sn=0;       // away from shape
			// Left by -ve side
#ifdef normal
			snside=kMX;
#endif
		}
		else
		{
			if (ds1>0)
			{
				// Ignore +ve plane and use -ve plane intersect
				//
				sn=ss1/ds1; // Leave by +ve side
#ifdef normal
				snside=kPX;
#endif
			}
			else
			{
				// Must be || to both -> exit determined by other axes
				//
				sn=UUtils::kInfinity; // Don't leave by either side
			}
		}
	}

	// Update minimum exit distance

	if (sn<snxt)
	{
		snxt=sn;
#ifdef normal
		side=snside;
#endif
	}
	if (snxt>0)
	{
		// Calculate y intersection
		tanyz=(fDy2-fDy1)*0.5/fDz;
		central=0.5*(fDy1+fDy2);

		// +ve plane (1)
		//
		ss1=central+tanyz*p.z-p.y; // distance || y axis to plane
		// (+ve if point inside)
		ds1=v.y-tanyz*v.z;  // component towards +ve plane
		// (-ve if +ve -> -ve direction)
		// -ve plane (2)
		//
		ss2=-tanyz*p.z-p.y-central; // distance || y axis to plane
		// (-ve if point inside)
		ds2=tanyz*v.z+v.y;  // component towards -ve plane

		if (ss1>0&&ss2<0)
		{
			// Normal case - entirely inside region

			if (ds1<=0&&ds2<0)
			{   
				if (ss2<-VUSolid::fgTolerance/2)
				{
					sn=ss2/ds2;  // Leave by -ve side
#ifdef normal
					snside=kMY;
#endif
				}
				else
				{
					sn=0; // Leave immediately by -ve side
#ifdef normal
					snside=kMY;
#endif
				}
			}
			else if (ds1>0&&ds2>=0)
			{
				if (ss1>VUSolid::fgTolerance/2)
				{
					sn=ss1/ds1;  // Leave by +ve side
#ifdef normal
					snside=kPY;
#endif
				}
				else
				{
					sn=0; // Leave immediately by +ve side
#ifdef normal
					snside=kPY;
#endif
				}
			}
			else if (ds1>0&&ds2<0)
			{
				if (ss1>VUSolid::fgTolerance/2)
				{
					// sn=ss1/ds1;  // Leave by +ve side
					if (ss2<-VUSolid::fgTolerance/2)
					{
						sn=ss1/ds1;  // Leave by +ve side
						sn2=ss2/ds2;
						if (sn2<sn)
						{
							sn=sn2;
#ifdef normal
							snside=kMY;
#endif
						}
						else
						{
#ifdef normal
							snside=kPY;
#endif
						}
					}
					else
					{
						sn=0; // Leave immediately by -ve
#ifdef normal
						snside=kMY;
#endif
					}
				}
				else
				{
					sn=0; // Leave immediately by +ve side
#ifdef normal
					snside=kPY;
#endif
				}
			}
			else
			{
				// Must be || to both
				//
				sn=UUtils::kInfinity;    // Don't leave by either side
			}
		}
		else if (ss1<=0&&ss2<0)
		{
			// Outside, in +ve Area

			if (ds1>0)
			{
				sn=0;       // Away from shape
				// Left by +ve side
#ifdef normal
				snside=kPY;
#endif
			}
			else
			{
				if (ds2<0)
				{
					// Ignore +ve plane and use -ve plane intersect
					//
					sn=ss2/ds2; // Leave by -ve side
#ifdef normal
					snside=kMY;
#endif
				}
				else
				{
					// Must be || to both -> exit determined by other axes
					//
					sn=UUtils::kInfinity; // Don't leave by either side
				}
			}
		}
		else if (ss1>0&&ss2>=0)
		{
			// Outside, in -ve Area
			if (ds2<0)
			{
				sn=0;       // away from shape
				// Left by -ve side
#ifdef normal
				snside=kMY;
#endif
			}
			else
			{
				if (ds1>0)
				{
					// Ignore +ve plane and use -ve plane intersect
					//
					sn=ss1/ds1; // Leave by +ve side
#ifdef normal
					snside=kPY;
#endif
				}
				else
				{
					// Must be || to both -> exit determined by other axes
					//
					sn=UUtils::kInfinity; // Don't leave by either side
				}
			}
		}

		// Update minimum exit distance

		if (sn<snxt)
		{
			snxt=sn;
#ifdef normal
			side=snside;
#endif
		}
	}

#ifdef normal
	{
		switch (side)
		{
		case kPX:
			cosxz=1.0/std::sqrt(1.0+tanxz*tanxz);
			n.Set(cosxz,0,-tanxz*cosxz);
			break;
		case kMX:
			cosxz=-1.0/std::sqrt(1.0+tanxz*tanxz);
			n.Set(cosxz,0,tanxz*cosxz);
			break;
		case kPY:
			cosyz=1.0/std::sqrt(1.0+tanyz*tanyz);
			n.Set(0,cosyz,-tanyz*cosyz);
			break;
		case kMY:
			cosyz=-1.0/std::sqrt(1.0+tanyz*tanyz);
			n.Set(0,cosyz,tanyz*cosyz);
			break;
		case kPZ:
			n.Set(0,0,1);
			break;
		case kMZ:
			n.Set(0,0,-1);
			break;
		default:
			//        DumpInfo();
			// G4Exception("G4Trd::DistanceToOut(p,v,..)","Notification",JustWarning, "Undefined side for valid surface normal to solid.");
			break;
		}
	}
#endif
	return snxt; 
}

//NOTE: Root does not fill vector n on return. Maybe that is the reason it is slightly faster. It also does not count with tolerances

inline double UTrd::DistanceToOutRoot( const UVector3  &p, const UVector3 &v,
	UVector3 & /*nRootDoesNotFillTh*/, bool & /*convex*/ /* not used */, double /*aPstep*/) const
{
	// Compute distance from inside point to surface of the trd2
	// Boundary safe algorithm
	double snxt = UUtils::kInfinity;

	/*
	if (iact<3 && safe) {
	// compute safe distance
	*safe = Safety(point, kTRUE);
	if (iact==0) return UUtils::kInfinity;
	if (iact==1 && step<*safe) return UUtils::kInfinity;
	}
	*/

	double fx = 0.5*(fDx1-fDx2)/fDz;
	double fy = 0.5*(fDy1-fDy2)/fDz;
	double cn;

	double distx = 0.5*(fDx1+fDx2)-fx*p.z;
	double disty = 0.5*(fDy1+fDy2)-fy*p.z;

	//--- Compute distance to this shape
	// first check if Z facettes are crossed
	double dist[3];
	for (int i=0; i<3; i++) dist[i]=UUtils::kInfinity;
	if (v.z<0) {
		dist[0]=-(p.z+fDz)/v.z;
	} else if (v.z>0) {
		dist[0]=(fDz-p.z)/v.z;
	}      
	if (dist[0]<=0) return 0.0;     
	// now check X facettes
	cn = -v.x+fx*v.z;
	if (cn>0) {
		dist[1] = p.x+distx;
		if (dist[1]<=0) return 0.0;
		dist[1] /= cn;
	}   
	cn = v.x+fx*v.z;
	if (cn>0) {
		double s = distx-p.x;
		if (s<=0) return 0.0;
		s /= cn;
		if (s<dist[1]) dist[1] = s;
	}
	// now check Y facettes
	cn = -v.y+fy*v.z;
	if (cn>0) {
		dist[2] = p.y+disty;
		if (dist[2]<=0) return 0.0;
		dist[2] /= cn;
	}
	cn = v.y+fy*v.z;
	if (cn>0) {
		double s = disty-p.y;
		if (s<=0) return 0.0;
		s /= cn;
		if (s<dist[2]) dist[2] = s;
	}
	snxt = amin(3,dist);
	return snxt;
}




// double *point, double *dir, double *norm;
inline bool UTrd::NormalRoot(const UVector3& point, UVector3 &norm) const
{
	// Compute normal to closest surface from POINT. 
	double safe, safemin;
	double fx = 0.5*(fDx1-fDx2)/fDz;
	double calf = 1./std::sqrt(1.0+fx*fx);

	// check Z facettes
	safe = safemin = std::abs(fDz-std::abs(point.z));
	norm.x = norm.y = 0;
	norm.z = 1;
	if (safe<1E-6) return true;

	// check X facettes
	double distx = 0.5*(fDx1+fDx2)-fx*point.z;
	if (distx>=0) {
		safe=std::abs(distx-std::abs(point.x))*calf;
		if (safe<safemin) {
			safemin = safe;
			norm.x = (point.x>0)?calf:(-calf);
			norm.y = 0;
			norm.z = calf*fx;
			/*
			double dot = norm.x*dir.x+norm.y*dir.y+norm.z*dir.z;
			if (dot<0) {
			norm.x=-norm.x;
			norm.z=-norm.z;
			}
			*/
			if (safe<1E-6) return true;
		}
	}

	double fy = 0.5*(fDy1-fDy2)/fDz;
	calf = 1./std::sqrt(1.0+fy*fy);

	// check Y facettes
	distx = 0.5*(fDy1+fDy2)-fy*point.z;
	if (distx>=0) {
		safe=std::abs(distx-std::abs(point.y))*calf;
		if (safe<safemin) {
			norm.x = 0;
			norm.y = (point.y>0)?calf:(-calf);
			norm.z = calf*fx;
			/*
			double dot = norm.x*dir.x+norm.y*dir.y+norm.z*dir.z;
			if (dot<0) {
			norm.y=-norm.y;
			norm.z=-norm.z;
			}
			*/
		}
	}
	return true;
}

/**
*
* Return unit normal of surface closest to p
* TODO: why root has this routine: 
* void TGeoTrd2::ComputeNormal(double *point, double *dir, double *norm)
* ???
*/

bool UTrd::Normal( const UVector3& p, UVector3 &n) const
{
	return NormalGeant4(p,n);
}

bool UTrd::NormalGeant4( const UVector3& p, UVector3 &norm) const
{
	UVector3 sumnorm(0.,0.,0.);
	int noSurfaces = 0; 
	double z = 2.0*fDz;
	double delta = 0.5*fgTolerance;

	double tanx  = (fDx2 - fDx1)/z;
	double secx  = std::sqrt(1.0+tanx*tanx);
	double newpx = std::abs(p.x)-p.z*tanx;
	double widx  = fDx2 - fDz*tanx;

	double tany  = (fDy2 - fDy1)/z;
	double secy  = std::sqrt(1.0+tany*tany);
	double newpy = std::abs(p.y)-p.z*tany;
	double widy  = fDy2 - fDz*tany;

	double distx = std::abs(newpx-widx)/secx;       // perp. distance to x side
	double disty = std::abs(newpy-widy)/secy;       //                to y side
	double distz = std::abs(std::abs(p.z)-fDz);  //                to z side

	if (distx <= delta) // we are near the x corner?      
	{
		double fcos = 1.0/secx;
		noSurfaces ++;
		if ( p.x >= 0.) sumnorm.x += fcos;
		else sumnorm.x -= fcos;
		sumnorm.z -= tanx*fcos;
	}
	if (disty <= delta) // y corner ...
	{
		double fcos = 1.0/secy; 
		noSurfaces++;
		if ( p.y >= 0.) sumnorm.y += fcos;
		else sumnorm.y -= fcos;
		sumnorm.z -= tany*fcos;
	}
	if (distz <= delta)  
	{
		noSurfaces++;
		sumnorm.z += (p.z >= 0.) ? 1.0 : -1.0; 
	}
	if ( noSurfaces == 0 )
	{
#ifdef G4CSGDEBUG
		// G4Exception("G4Trd::SurfaceNormal(p)", "Notification", JustWarning, "Point p is not on surface !?" );
#endif 
		// point is not on surface, calculate normal closest to surface. this is not likely to be used too often..., normally the user gives point on surface...
		//     norm = ApproxSurfaceNormal(p);

		// find closest side
		//
		double fcos;

		if (distx<=disty)
		{ 
			if (distx<=distz) 
			{
				// Closest to X
				//
				fcos=1.0/secx;
				// normal=(+/-std::cos(ang),0,-std::sin(ang))
				if (p.x>=0)
					norm=UVector3(fcos,0,-tanx*fcos);
				else
					norm=UVector3(-fcos,0,-tanx*fcos);
			}
			else
			{
				// Closest to Z
				//
				if (p.z>=0)
					norm=UVector3(0,0,1);
				else
					norm=UVector3(0,0,-1);
			}
		}
		else
		{  
			if (disty<=distz)
			{
				// Closest to Y
				//
				fcos=1.0/secy;
				if (p.y>=0)
					norm=UVector3(0,fcos,-tany*fcos);
				else
					norm=UVector3(0,-fcos,-tany*fcos);
			}
			else 
			{
				// Closest to Z
				//
				if (p.z>=0)
					norm=UVector3(0,0,1);
				else
					norm=UVector3(0,0,-1);
			}
		}
	}
	else 
	{
		norm = sumnorm;
		// if we added to the vector more than once, we have to normalize the vector
		if ( noSurfaces > 1 ) norm.Normalize();
	}
	return noSurfaces > 0; // (Inside(p) == eSurface);
}


/////////////////////////////////////////////////////////////////////////////
//
// Algorithm for SurfaceNormal() following the original specification
// for points not on the surface

inline UVector3 UTrd::ApproxSurfaceNormal( const UVector3& p ) const
{
	UVector3 norm;
	double z,tanx,secx,newpx,widx;
	double tany,secy,newpy,widy;
	double distx,disty,distz,fcos;

	z=2.0*fDz;

	tanx=(fDx2-fDx1)/z;
	secx=std::sqrt(1.0+tanx*tanx);
	newpx=std::abs(p.x)-p.z*tanx;
	widx=fDx2-fDz*tanx;

	tany=(fDy2-fDy1)/z;
	secy=std::sqrt(1.0+tany*tany);
	newpy=std::abs(p.y)-p.z*tany;
	widy=fDy2-fDz*tany;

	distx=std::abs(newpx-widx)/secx;  // perpendicular distance to x side
	disty=std::abs(newpy-widy)/secy;  //                        to y side
	distz=std::abs(std::abs(p.z)-fDz);  //                        to z side

	// find closest side
	//
	if (distx<=disty)
	{ 
		if (distx<=distz) 
		{
			// Closest to X
			//
			fcos=1.0/secx;
			// normal=(+/-std::cos(ang),0,-std::sin(ang))
			if (p.x>=0)
				norm=UVector3(fcos,0,-tanx*fcos);
			else
				norm=UVector3(-fcos,0,-tanx*fcos);
		}
		else
		{
			// Closest to Z
			//
			if (p.z>=0)
				norm=UVector3(0,0,1);
			else
				norm=UVector3(0,0,-1);
		}
	}
	else
	{  
		if (disty<=distz)
		{
			// Closest to Y
			//
			fcos=1.0/secy;
			if (p.y>=0)
				norm=UVector3(0,fcos,-tany*fcos);
			else
				norm=UVector3(0,-fcos,-tany*fcos);
		}
		else 
		{
			// Closest to Z
			//
			if (p.z>=0)
				norm=UVector3(0,0,1);
			else
				norm=UVector3(0,0,-1);
		}
	}
	return norm;   
}


/**
* Returns extent of the solid along a given cartesian axis
* OK
*/
void UTrd::Extent( EAxisType aAxis, double &aMin, double &aMax ) const
{
	switch (aAxis)
	{
	case eXaxis:
		aMin = -std::max (fDx1, fDx2); 
		aMax = std::max (fDx1, fDx2);
		break;
	case eYaxis:
		aMin = -std::max (fDy1, fDy2); 
		aMax = std::max (fDy1, fDy2);
		break;
	case eZaxis:
		aMin = -fDz; aMax = fDz;
		break;
	default:
		std::cout << "Extent: unknown axis" << aAxis << std::endl;
	}      
}            

/**
* Returns the full 3D cartesian extent of the solid.
* OK
*/
void UTrd::Extent ( UVector3 &aMin, UVector3 &aMax) const
{
	aMin.Set(-std::max (fDx1, fDx2), -std::max (fDy1, fDy2), -fDz);
	aMax.Set(std::max (fDx1, fDx2), std::max (fDy1, fDy2), fDz);
}

double UTrd::Capacity() 
{
	return 0;
}

double UTrd::SurfaceArea()
{
	return 0;
}

//////////////////////////////////////////////////////////////////////////
//
// Stream object contents to an output stream

std::ostream& UTrd::StreamInfo( std::ostream& os ) const
{
	int oldprc = os.precision(16);
	os << "-----------------------------------------------------------\n"
		 << "		*** Dump for solid - " << GetName() << " ***\n"
		 << "		===================================================\n"
		 << " Solid type: UTrd\n"
		 << " Parameters: \n"
		 << "		half length X, surface -dZ: " << fDx1 << " mm \n"
		 << "		half length X, surface +dZ: " << fDx2 << " mm \n"
		 << "		half length Y, surface -dZ: " << fDy1 << " mm \n"
		 << "		half length Y, surface +dZ: " << fDy2 << " mm \n"
		 << "		half length Z						 : " << fDz << " mm \n"
		 << "-----------------------------------------------------------\n";
	os.precision(oldprc);

	return os;
}

////////////////////////////////////////////////////////////////////////
//
// GetPointOnSurface
//
// Return a point (UVector3) randomly and uniformly
// selected on the solid surface

UVector3 UTrd::GetPointOnSurface() const
{
	double px, py, pz, tgX, tgY, secX, secY, select, sumS, tmp;
	double Sxy1, Sxy2, Sxy, Sxz, Syz;

	tgX	= 0.5*(fDx2-fDx1)/fDz;
	secX = std::sqrt(1+tgX*tgX);
	tgY	= 0.5*(fDy2-fDy1)/fDz;
	secY = std::sqrt(1+tgY*tgY);

	// calculate 0.25 of side surfaces, sumS is 0.25 of total surface

	Sxy1 = fDx1*fDy1; 
	Sxy2 = fDx2*fDy2;
	Sxy	= Sxy1 + Sxy2; 
	Sxz	= (fDx1 + fDx2)*fDz*secY; 
	Syz	= (fDy1 + fDy2)*fDz*secX;
	sumS = Sxy + Sxz + Syz;

	select = sumS*UUtils::Random();
 
	if( select < Sxy )									// Sxy1 or Sxy2
	{
		if( select < Sxy1 ) 
		{
			pz = -fDz;
			px = -fDx1 + 2*fDx1*UUtils::Random();
			py = -fDy1 + 2*fDy1*UUtils::Random();
		}
		else			
		{
			pz =	fDz;
			px = -fDx2 + 2*fDx2*UUtils::Random();
			py = -fDy2 + 2*fDy2*UUtils::Random();
		}
	}
	else if ( ( select - Sxy ) < Sxz )		// Sxz
	{
		pz	= -fDz	+ 2*fDz*UUtils::Random();
		tmp =	fDx1 + (pz + fDz)*tgX;
		px	= -tmp	+ 2*tmp*UUtils::Random();
		tmp =	fDy1 + (pz + fDz)*tgY;

		if(UUtils::Random() > 0.5) { py =	tmp; }
		else											{ py = -tmp; }
	}
	else																	 // Syz
	{
		pz	= -fDz	+ 2*fDz*UUtils::Random();
		tmp =	fDy1 + (pz + fDz)*tgY;
		py	= -tmp	+ 2*tmp*UUtils::Random();
		tmp =	fDx1 + (pz + fDz)*tgX;

		if(UUtils::Random() > 0.5) { px =	tmp; }
		else											{ px = -tmp; }
	} 
	return UVector3(px,py,pz);
}

UPolyhedron* UTrd::CreatePolyhedron () const
{
	return new UPolyhedronTrd2 (fDx1, fDx2, fDy1, fDy2, fDz);
}
