//
// ********************************************************************
// * License and Disclaimer																					 *
// *																																	*
// * The	Geant4 software	is	copyright of the Copyright Holders	of *
// * the Geant4 Collaboration.	It is provided	under	the terms	and *
// * conditions of the Geant4 Software License,	included in the file *
// * LICENSE and available at	http://cern.ch/geant4/license .	These *
// * include a list of copyright holders.														 *
// *																																	*
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work	make	any representation or	warranty, express or implied, *
// * regarding	this	software system or assume any liability for its *
// * use.	Please see the license in the file	LICENSE	and URL above *
// * for the full disclaimer and the limitation of liability.				 *
// *																																	*
// * This	code	implementation is the result of	the	scientific and *
// * technical work of the GEANT4 collaboration.											*
// * By using,	copying,	modifying or	distributing the software (or *
// * any work based	on the software)	you	agree	to acknowledge its *
// * use	in	resulting	scientific	publications,	and indicate your *
// * acceptance of all terms of the Geant4 Software license.					*
// ********************************************************************
//
//
// $Id: UPolycone3.cc 66241 2012-12-13 18:34:42Z gunter $
//
// 
// --------------------------------------------------------------------
// GEANT 4 class source file
//
//
// UPolycone3.cc
//
// Implementation of a CSG polycone
//
// --------------------------------------------------------------------

#include "UUtils.hh"
#include <string>
#include <cmath>
#include <sstream>
#include "UPolycone3.hh"
#include "UPolycone.hh"

#include "UEnclosingCylinder.hh"
#include "UReduciblePolygon.hh"

#include "UTubs.hh"
#include "UCons.hh"
//#include "UHedra.hh"
#include "UTransform3D.hh"

using namespace std;

UBox box;
 

UPolycone3::UPolycone3( const std::string& name, 
  double phiStart,
  double phiTotal,
  int numZPlanes,
  const double zPlane[],
  const double rInner[],
  const double rOuter[]	)
  : VUSolid( name ), fNumSides(0)
{
  Init(phiStart, phiTotal, numZPlanes, zPlane, rInner, rOuter);
}

//
// Constructor (GEANT3 style parameters)
//	
void UPolycone3::Init(double phiStart,
															double phiTotal,
															int numZPlanes,
												const double zPlane[],
												const double rInner[],
												const double rOuter[])
{
	//
	// Some historical ugliness
	//
	fOriginalParameters = new UPolyconeHistorical();
	
	fOriginalParameters->fStartAngle = phiStart;
	fOriginalParameters->fOpeningAngle = phiTotal;
	fOriginalParameters->fNumZPlanes = numZPlanes;
	fOriginalParameters->fZValues.resize(numZPlanes);
	fOriginalParameters->Rmin.resize(numZPlanes);
	fOriginalParameters->Rmax.resize(numZPlanes);

  double prevZ, prevRmax, prevRmin;

//  int curSolid = 0;

	int i;
	for (i=0; i<numZPlanes; i++)
	{
		if (( i < numZPlanes-1) && ( zPlane[i] == zPlane[i+1] ))
		{
			if( (rInner[i]	 > rOuter[i+1])
				||(rInner[i+1] > rOuter[i])	 )
			{
				
				std::ostringstream message;
				message << "Cannot create a Polycone with no contiguous segments."
								<< std::endl
								<< "				Segments are not contiguous !" << std::endl
								<< "				rMin[" << i << "] = " << rInner[i]
								<< " -- rMax[" << i+1 << "] = " << rOuter[i+1] << std::endl
								<< "				rMin[" << i+1 << "] = " << rInner[i+1]
								<< " -- rMax[" << i << "] = " << rOuter[i];
				// UException("UPolycone3::UPolycone3()", "GeomSolids0002",
				//						FatalErrorInArgument, message);
			}
		} 

    double rMin = rInner[i];
    double rMax = rOuter[i];
    double z = zPlane[i];

    if (i > 0)
    {
      if (z > prevZ)
      {
        VUSolid *solid;
        double dz = (z - prevZ)/2;
        UTransform3D *trans = new UTransform3D(0, 0, prevZ + dz);

        bool tubular = (rMin == prevRmin && prevRmax == rMax);

        if (fNumSides == 0)
        {
          if (tubular)
          {
            solid = new UTubs("", rMin, rMax, dz, phiStart, phiTotal);
          }
          else
          {
            solid = new UCons("", prevRmin, prevRmax, rMin, rMax, dz, phiStart, phiTotal);
          }
        }
        else
        {
//          solid = new UHedra("", prevRmin, prevRmax, rMin, rMax, dz, phiStart, phiTotal, fNumSides);
        }

        fZs.push_back(z);

        int zi = fZs.size() - 1;
        double shift = fZs[zi-1] + 0.5 * (fZs[zi] - fZs[zi-1]);

        UPolyconeSection section;
        section.shift = shift;
        section.tubular = tubular;
        section.solid = solid;
//        section.left = fZs[zi-1];
//        section.right = z;

        fSections.push_back(section);
      }
      else
        i = i;
    }
    else fZs.push_back(z);

		fOriginalParameters->fZValues[i] = zPlane[i];
		fOriginalParameters->Rmin[i] = rInner[i];
		fOriginalParameters->Rmax[i] = rOuter[i];

    prevZ = z;
    prevRmin = rMin;
    prevRmax = rMax;
	}

  fMaxSection = fZs.size() - 2;

	//
	// Build RZ polygon using special PCON/PGON GEANT3 constructor
	//
	UReduciblePolygon *rz = new UReduciblePolygon( rInner, rOuter, zPlane, numZPlanes);

  double mxy = rz->Amax(); 
  double alfa = UUtils::kPi / fNumSides;

  double r= rz->Amax();

  if (fNumSides != 0) 
  {
    // mxy *= std::sqrt(2.0); // this is old and wrong, works only for n = 4
    double k = std::tan(alfa) * mxy;
    double l = mxy / std::cos(alfa);
    mxy = l;
    r = l;
  }

  mxy += fgTolerance;

  box.Set(mxy, mxy, (rz->Bmax() - rz->Bmin()) /2);

  phiIsOpen = (phiTotal > 0 && phiTotal <= 2*UUtils::kPi-1E-10);
  startPhi = phiStart;
  endPhi = phiTotal;

  //
  // Make enclosingCylinder
  //
  
  enclosingCylinder = new UEnclosingCylinder(r, rz->Bmax(), rz->Bmin(), phiIsOpen, phiStart, phiTotal);
	
	delete rz;
}


/*
//
// Constructor (generic parameters)
//
UPolycone3::UPolycone3( const std::string& name, 
															double phiStart,
															double phiTotal,
															int		numRZ,
												const double r[],
												const double z[]	 )
	: VUSolid( name )
{
	UReduciblePolygon *rz = new UReduciblePolygon( r, z, numRZ );

  box.Set(rz->Amax(), rz->Amax(), (rz->Bmax() - rz->Bmin()) /2);
	
	// Set fOriginalParameters struct for consistency
	//
	SetOriginalParameters();
	
	delete rz;
}
*/


//
// Destructor
//
UPolycone3::~UPolycone3()
{
	//delete [] corners;
	//delete fOriginalParameters;
}



//
// Stream object contents to an output stream
//
std::ostream& UPolycone3::StreamInfo( std::ostream& os ) const
{
	int oldprc = os.precision(16);
	os << "-----------------------------------------------------------\n"
		 << "		*** Dump for solid - " << GetName() << " ***\n"
		 << "		===================================================\n"
		 << " Solid type: UPolycone3\n"
		 << " Parameters: \n"
		 << "		starting phi angle : " << startPhi/(UUtils::kPi/180.0) << " degrees \n"
		 << "		ending phi angle	 : " << endPhi/(UUtils::kPi/180.0) << " degrees \n";
	int i=0;
	if (!genericPcon)
	{
		int numPlanes = fOriginalParameters->fNumZPlanes;
		os << "		number of Z planes: " << numPlanes << "\n"
			 << "							Z values: \n";
		for (i=0; i<numPlanes; i++)
		{
			os << "							Z plane " << i << ": "
				 << fOriginalParameters->fZValues[i] << "\n";
		}
		os << "							Tangent distances to inner surface (Rmin): \n";
		for (i=0; i<numPlanes; i++)
		{
			os << "							Z plane " << i << ": "
				 << fOriginalParameters->Rmin[i] << "\n";
		}
		os << "							Tangent distances to outer surface (Rmax): \n";
		for (i=0; i<numPlanes; i++)
		{
			os << "							Z plane " << i << ": "
				 << fOriginalParameters->Rmax[i] << "\n";
		}
	}
	os << "		number of RZ points: " << numCorner << "\n"
		 << "							RZ values (corners): \n";
		 for (i=0; i<numCorner; i++)
		 {
			 os << "												 "
					<< corners[i].r << ", " << corners[i].z << "\n";
		 }
	os << "-----------------------------------------------------------\n";
	os.precision(oldprc);

	return os;
}


VUSolid::EnumInside UPolycone3::InsideSection(int index, const UVector3 &p) const
{
  const UPolyconeSection &section = fSections[index];
  UVector3 ps(p.x, p.y, p.z - section.shift);

  if (fNumSides) return section.solid->Inside(ps);

  double rMinPlus, rMaxPlus; //, rMinMinus, rMaxMinus;
  double dz;
  static double halfTolerance = fgTolerance * 0.5;

  if (section.tubular)
  {
    UTubs *tubs = (UTubs *) section.solid;
    rMinPlus = tubs->GetRMin() + halfTolerance;
    rMaxPlus = tubs->GetRMax() + halfTolerance;
    dz = tubs->GetDz();
  }
  else
  {
    UCons *cons = (UCons *) section.solid;

    double rMax1 = cons->GetRmax1();
    double rMax2 = cons->GetRmax2();
    double rMin1 = cons->GetRmin1();
    double rMin2 = cons->GetRmin2();

    dz = cons->GetDz();
    double ratio = (ps.z + dz) / (2*dz);
    rMinPlus = rMin1 + (rMin2 - rMin1) * ratio + halfTolerance;
    rMaxPlus = rMax1 + (rMax2 - rMax1) * ratio + halfTolerance;
  }

  double rMinMinus = rMinPlus - fgTolerance;
  double rMaxMinus = rMaxPlus - fgTolerance;

  double r2 = p.x*p.x + p.y*p.y;

  if (r2 < rMinMinus * rMinMinus || r2 > rMaxPlus*rMaxPlus) return eOutside;
  if (r2 < rMinPlus * rMinPlus || r2 > rMaxMinus*rMaxMinus) return eSurface;

  if (endPhi == UUtils::kTwoPi) 
  {
    if (ps.z < -dz + halfTolerance || ps.z > dz - halfTolerance) 
      return eSurface;
    return eInside;
  }

  if (r2 < 1e-10) return eInside;

  double phi = std::atan2(p.y,p.x);// * UUtils::kTwoPi;
  if (phi < 0) phi += UUtils::kTwoPi;
  double ddp = phi - startPhi;
  if (ddp < 0) ddp += UUtils::kTwoPi;
  if (ddp <= endPhi + frTolerance)  
  {
    if (ps.z < -dz + halfTolerance || ps.z > dz - halfTolerance) 
      return eSurface;
    if (endPhi - ddp < frTolerance)
      return eSurface;

    return eInside;
  }
  return eOutside;

  // old code:
  EnumInside res = section.solid->Inside(ps);

  // this two lines make no difference
  //    EnumInside res = section.tubular ? ((UTubs *) section.solid)->Inside(ps) : ((UCons *) section.solid)->Inside(ps);

  return res;
}


VUSolid::EnumInside UPolycone3::Inside( const UVector3 &p ) const
{
  double shift = fZs[0] + box.GetDz();
  UVector3 pb(p.x, p.y, p.z - shift);
  if (box.Inside(pb) == eOutside)
    return eOutside;

  static const double htolerance = 0.5 * fgTolerance;
  int index = GetSection(p.z);

  EnumInside pos = InsideSection(index, p);
  if (pos == eInside) return eInside;

  int nextSection;
  EnumInside nextPos;

  if (index > 0 && p.z  - fZs[index] < htolerance)
  {
    nextSection = index-1;
    nextPos = InsideSection(nextSection, p);
  }
  else if (index < fMaxSection && fZs[index+1] - p.z < htolerance)
  {
    nextSection = index+1;
    nextPos = InsideSection(nextSection, p);
  }
  else 
    return pos;

  if (nextPos == eInside) return eInside;

  if(pos == eSurface && nextPos == eSurface)
  {
    UVector3 n, n2;
    NormalSection(index, p, n); NormalSection(nextSection, p, n2);
    if ((n +  n2).Mag2() < 1000*frTolerance)
      return eInside;
  }

  return (nextPos == eSurface || pos == eSurface) ? eSurface : eOutside;

//  return (res == VUSolid::eOutside) ? nextPos : res;
}

  /*
  if (p.z < fZs.front() - htolerance || p.z > fZs.back() + htolerance) return VUSolid::eOutside;
  */

double UPolycone3::DistanceToIn( const UVector3 &p,
  const UVector3 &v, double) const
{
  double shift = fZs[0] + box.GetDz();
  UVector3 pb(p.x, p.y, p.z - shift);
  
  double idistance;
  
  idistance = box.DistanceToIn(pb, v); // using only box, this appears 
  // to be faster than: idistance = enclosingCylinder->DistanceTo(pb, v);
  if (idistance >= UUtils::kInfinity) return idistance;

  // this line can be here or not. not a big difference in performance
  // TODO: fix enclosingCylinder for polyhedra!!! - the current radius appears to be too small
//  if (enclosingCylinder->ShouldMiss(p, v)) return UUtils::kInfinity;

  // this just takes too much time
//  idistance = enclosingCylinder->DistanceTo(pb, v);
//  if (idistance == UUtils::kInfinity) return idistance;

  pb = p + idistance * v;
  int index = GetSection(pb.z);
  pb = p;
  int increment = (v.z > 0) ? 1 : -1;
  if (std::fabs(v.z) < fgTolerance) increment = 0;

  double distance = UUtils::kInfinity;
  do
  {
    const UPolyconeSection &section = fSections[index];
    pb.z -= section.shift;
    distance = section.solid->DistanceToIn(pb,v);
    if (distance < UUtils::kInfinity || !increment)
      break;
    index += increment;
    pb.z += section.shift;
  }
  while (index >= 0 && index <= fMaxSection);

  return distance;
}

double UPolycone3::DistanceToOut( const UVector3  &p, const UVector3 &v,
  UVector3 &n, bool &convex, double /*aPstep*/) const
{
  UVector3 pn(p);
  if (fOriginalParameters->fNumZPlanes == 2)
  {
    const UPolyconeSection &section = fSections[0];
    pn.z -= section.shift;
    return section.solid->DistanceToOut(pn, v, n, convex);
  }
  int index =  GetSection(p.z);
  double totalDistance = 0;
  int increment = (v.z > 0) ? 1 : -1;

//  UVector3 normal;
  do
  {
    const UPolyconeSection &section = fSections[index];
    if (totalDistance != 0)
    {
      pn = p + (totalDistance /*+ 0 * 1e-8*/) * v; // point must be shifted, so it could eventually get into another solid
      pn.z -= section.shift;
      if (section.solid->Inside(pn) == eOutside)
      {
        index = index;
        break;
      }
    }
    else pn.z -= section.shift;

//    if (totalDistance > 0) totalDistance += 1e-8;
    double distance = section.solid->DistanceToOut(pn, v, n, convex);

    /*
    if (convex == false)
    {
      n.Set(0);

//      ps += distance * v;
//      section.solid->Normal(ps, n);
    }
  */

    /*
    double dif = (n2 - n).Mag();
    if (dif > 0.0000001) 
      n = n;
      */

    if (distance == 0) 
    {
      distance = distance;
      break;
    }
//    n = normal;

    totalDistance += distance;

    index += increment;

    // pn.z check whether it still relevant
  }
  while (index >= 0 && index <= fMaxSection);

//  pn = p + (totalDistance + 1e-8) * v;

//  Normal(pn, n);

//  convex = (DistanceToIn(pn, n) == UUtils::kInfinity);
  convex = false;

  return totalDistance;
}

double UPolycone3::SafetyFromInside ( const UVector3 &p, bool /*aAccurate*/) const
{
  int index = UVoxelizer::BinarySearch(fZs, p.z);
  if (index < 0 || index > fMaxSection) return 0;

  double minSafety = SafetyFromInsideSection(index, p);
  if (minSafety > UUtils::kInfinity) return 0;
  if (minSafety < 1e-6) return 0;

  double zbase = fZs[index+1];
  for (int i = index+1; i <= fMaxSection; ++i)
  {
    double dz = fZs[i] - zbase;
    if (dz >= minSafety) break;
    double safety = SafetyFromOutsideSection(i, p); // safety from Inside cannot be called in this context, because point is not inside, we have to call SafetyFromOutside for given section
    if (safety < minSafety) minSafety = safety;
  }

  if (index > 0)
  {
    zbase = fZs[index-1];
    for (int i = index-1; i >= 0; --i)
    {
      double dz = zbase - fZs[i];
      if (dz >= minSafety) break;
      double safety = SafetyFromOutsideSection(i, p);
      if (safety < minSafety) minSafety = safety;
    }
  }
  return minSafety;
}


double UPolycone3::SafetyFromOutside ( const UVector3 &p, bool aAccurate) const
{
  if (!aAccurate)
    return enclosingCylinder->SafetyFromOutside(p);

  int index = GetSection(p.z);
  double minSafety = SafetyFromOutsideSection(index, p);
  if (minSafety < 1e-6) return minSafety;

  double zbase = fZs[index+1];
  for (int i = index+1; i <= fMaxSection; ++i)
  {
    double dz = fZs[i] - zbase;
    if (dz >= minSafety) break;
    double safety = SafetyFromOutsideSection(i, p);
    if (safety < minSafety) minSafety = safety;
  }

  zbase = fZs[index-1];
  for (int i = index-1; i >= 0; --i)
  {
    double dz = zbase - fZs[i];
    if (dz >= minSafety) break;
    double safety = SafetyFromOutsideSection(i, p);
    if (safety < minSafety) minSafety = safety;
  }   
  return minSafety;
}

bool UPolycone3::Normal( const UVector3& p, UVector3 &n) const
{
  double htolerance = 0.5 * fgTolerance;
  int index = GetSection(p.z);

  EnumInside nextPos;
  int nextSection;

  if (index > 0 && p.z  - fZs[index] < htolerance)
  {
    nextSection = index-1;
    nextPos = InsideSection(nextSection, p);
  }
  else if (index < fMaxSection && fZs[index+1] - p.z < htolerance)
  {
    nextSection = index+1;
    nextPos = InsideSection(nextSection, p);
  }
  else 
  {
    const UPolyconeSection &section = fSections[index];
    UVector3 ps(p.x, p.y, p.z - section.shift);
    bool res = section.solid->Normal(ps, n);

    return res;

    // the code bellow is not used can be deleted
    
    nextPos = section.solid->Inside(ps);
    if (nextPos == eSurface)
    {
      return res;
    }
    else
    {
      int i = 0;
      // "TODO: here should be implementation for case point was not on surface. We would have to look also at other sections. It is not clear if it is possible to solve this problem at all, since we would need precise safety... If it is outside, than it might be OK, but if it is inside..., than I beleive we do not have precise safety";

      // "... or we should at least warn that this case is not supported. actually",
      //  "i do not see any algorithm which would obtain right normal of point closest to surface.";

      return false;
    }
  }

  // even if it says we are on the surface, actually it do not have to be 

//  "TODO special case when point is on the border of two z-sections",
//    "we should implement this after safety's";

  EnumInside pos = InsideSection(index, p);

  if (nextPos == eInside) 
  {
    UVector3 n;
    NormalSection(index, p, n);
    return false;
  }

  if( pos == eSurface && nextPos == eSurface)
  {
    UVector3 n, n2;
    NormalSection(index, p, n);
    NormalSection(nextSection, p, n2);
    if ( (n + n2).Mag2() < 1000*frTolerance)
    {
      // "we are inside. see TODO above";
      NormalSection(index, p, n);
      return false;
    }
  }

  if (nextPos == eSurface || pos == eSurface) 
  {
    if (pos != eSurface) index = nextSection;
    bool res = NormalSection(index, p, n);
    return res;
  }
  
  NormalSection(index, p, n);
  // "we are outside. see TODO above";
  return false;
}


void UPolycone3::Extent (UVector3 &aMin, UVector3 &aMax) const
{
  double r = enclosingCylinder->radius;
	aMin.Set(-r, -r, fZs.front());
	aMax.Set(r, r, fZs.back());
}

double UPolycone3::Capacity() 
{
  double capacity = 0;
  for (int i = 0; i < fMaxSection; i++)
  {
    UPolyconeSection &section = fSections[i];
    capacity += section.solid->Capacity();
  }
	return capacity;
}

double UPolycone3::SurfaceArea()
{
  double area = 0;
  for (int i = 0; i < fMaxSection; ++i)
  {
    UPolyconeSection &section = fSections[i];
    area += section.solid->SurfaceArea();
  }
  return area;
}

/////////////////////////////////////////////////////////////////////////
//
// GetPointOnSurface

UVector3 UPolycone3::GetPointOnSurface() const
{
  int index = (int) UUtils::Random(0, fMaxSection);
  const UPolyconeSection &section = fSections[index];

  UVector3 point;
  do 
  {
    point = section.solid->GetPointOnSurface();
  }
  while (Inside(point) != eSurface);

	return point;
}

UPolyhedron* UPolycone3::CreatePolyhedron () const
{
	return NULL;
}
