//
// ********************************************************************
// * License and Disclaimer                                           *
// *                                                                  *
// * The  Geant4 software  is  copyright of the Copyright Holders  of *
// * the Geant4 Collaboration.  It is provided  under  the terms  and *
// * conditions of the Geant4 Software License,  included in the file *
// * LICENSE and available at  http://cern.ch/geant4/license .  These *
// * include a list of copyright holders.                             *
// *                                                                  *
// * Neither the authors of this software system, nor their employing *
// * institutes,nor the agencies providing financial support for this *
// * work  make  any representation or  warranty, express or implied, *
// * regarding  this  software system or assume any liability for its *
// * use.  Please see the license in the file  LICENSE  and URL above *
// * for the full disclaimer and the limitation of liability.         *
// *                                                                  *
// * This  code  implementation is the result of  the  scientific and *
// * technical work of the GEANT4 collaboration and of QinetiQ Ltd,   *
// * subject to DEFCON 705 IPR conditions.                            *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
// $Id: UTriangularFacet.cc,v 1.16 2010-09-23 10:27:25 gcosmo Exp $
// GEANT4 tag $Name: not supported by cvs2svn $
//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// MODULE:              UTriangularFacet.cc
//
// Date:                15/06/2005
// Author:              P R Truscott
// Organisation:        QinetiQ Ltd, UK
// Customer:            UK Ministry of Defence : RAO CRP TD Electronic Systems
// Contract:            C/MAT/N03517
//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// CHANGE HISTORY
// --------------
//
// 31 October 2004, P R Truscott, QinetiQ Ltd, UK - Created.
//
// 01 August 2007   P R Truscott, QinetiQ Ltd, UK
//                  Significant modification to correct for errors and enhance
//                  based on patches/observations kindly provided by Rickard
//                  Holmberg
//
// 26 September 2007
//                  P R Truscott, QinetiQ Ltd, UK
//                  Further chamges implemented to the Intersect member
//                  function to correctly treat rays nearly parallel to the
//                  plane of the triangle.
//
// 12 April 2010    P R Truscott, QinetiQ, bug fixes to treat optical
//                  photon transport, in particular internal reflection
//                  at surface.
//
// 22 August 2011   I Hrivnacova, Orsay, fix in Intersect() to take into
//                  account geometrical tolerance and cases of zero distance
//                  from surface.
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include "UTriangularFacet.hh"
#include "UVector2.hh"
//#include "Randomize.hh"

#include <sstream>

using namespace std;

///////////////////////////////////////////////////////////////////////////////
//
// Definition of triangular facet using absolute vectors to vertices.
// From this for first vector is retained to define the facet location and
// two relative vectors (E0 and E1) define the sides and orientation of 
// the outward surface normal.
//
UTriangularFacet::UTriangularFacet (const UVector3 Pt0,
             const UVector3 vt1, const UVector3 vt2,
                   UFacetVertexType vertexType)
  : UFacet(), sMin(0.), sMax(1.), tMin(0.), sqrDist(0.)
{
  tGeomAlg  = UTessellatedGeometryAlgorithms::GetInstance();
  P0        = Pt0;
  nVertices = 3;
  if (vertexType == UABSOLUTE)
  {
    P.push_back(vt1);
    P.push_back(vt2);
  
    E.push_back(vt1 - P0);
    E.push_back(vt2 - P0);
  }
  else
  {
    P.push_back(P0 + vt1);
    P.push_back(P0 + vt2);
  
    E.push_back(vt1);
    E.push_back(vt2);
  }

  double Emag1 = E[0].Mag();
  double Emag2 = E[1].Mag();
  double Emag3 = (E[1]-E[0]).Mag();
  
  if (Emag1 <= kCarTolerance || Emag2 <= kCarTolerance ||
      Emag3 <= kCarTolerance)
  {
    std::ostringstream message;
    message << "Length of sides of facet are too small." << endl
            << "P0 = " << P0   << endl
            << "P1 = " << P[0] << endl
            << "P2 = " << P[1] << endl
            << "Side lengths = P0->P1" << Emag1 << endl
            << "Side lengths = P0->P2" << Emag2 << endl
            << "Side lengths = P1->P2" << Emag3;
//    UException("UTriangularFacet::UTriangularFacet()",
//                "GeomSolids1001", JustWarning, message);
    isDefined     = false;
    geometryType  = "UTriangularFacet";
    surfaceNormal = UVector3(0.0,0.0,0.0);
    a   = b   = c = 0.0;
    det = 0.0;
  }
  else
  {
    isDefined     = true;
    geometryType  = "UTriangularFacet";
	surfaceNormal = E[0].Cross(E[1]).Unit();
	a   = E[0].Mag2();
	b   = E[0].Dot(E[1]);
	c   = E[1].Mag2();
    det = std::fabs(a*c - b*b);
    
    sMin = -0.5*kCarTolerance/std::sqrt(a);
    sMax = 1.0 - sMin;
    tMin = -0.5*kCarTolerance/std::sqrt(c);
    
	area = 0.5 * (E[0].Cross(E[1])).Mag();

//    UVector3 vtmp = 0.25 * (E[0] + E[1]);
    double lambda0 = (a-b) * c / (8.0*area*area);
    double lambda1 = (c-b) * a / (8.0*area*area);
    circumcentre     = P0 + lambda0*E[0] + lambda1*E[1];
	radiusSqr        = (circumcentre-P0).Mag2();
    radius           = std::sqrt(radiusSqr);
  
    for (int i=0; i<3; i++) { I.push_back(0); }
  }
}

///////////////////////////////////////////////////////////////////////////////
//
// ~UTriangularFacet
//
// A pretty boring destructor indeed!
//
UTriangularFacet::~UTriangularFacet ()
{
  P.clear();
  E.clear();
  I.clear();
}

///////////////////////////////////////////////////////////////////////////////
//
// Copy constructor
//
UTriangularFacet::UTriangularFacet (const UTriangularFacet &rhs)
  : UFacet(rhs), a(rhs.a), b(rhs.b), c(rhs.c), det(rhs.det),
    sMin(rhs.sMin), sMax(rhs.sMax), tMin(rhs.tMin), sqrDist(rhs.sqrDist)
{
  tGeomAlg = UTessellatedGeometryAlgorithms::GetInstance();
}

///////////////////////////////////////////////////////////////////////////////
//
// Assignment operator
//
const UTriangularFacet &UTriangularFacet::operator=(UTriangularFacet &rhs)
{
   // Check assignment to self
   //
   if (this == &rhs)  { return *this; }

   // Copy base class data
   //
   UFacet::operator=(rhs);

   // Copy data
   //
   a = rhs.a; b = rhs.b; c = rhs.c; det = rhs.det;
   sMin = rhs.sMin; sMax = rhs.sMax; tMin = rhs.tMin; sqrDist = rhs.sqrDist;
   tGeomAlg = UTessellatedGeometryAlgorithms::GetInstance();

   return *this;
}

///////////////////////////////////////////////////////////////////////////////
//
// GetClone
//
// Simple member function to generate a diplicate of the triangular facet.
//
UFacet *UTriangularFacet::GetClone ()
{
  UTriangularFacet *fc = new UTriangularFacet (P0, P[0], P[1], UABSOLUTE);
  UFacet *cc         = 0;
  cc                   = fc;
  return cc;
}

///////////////////////////////////////////////////////////////////////////////
//
// GetFlippedFacet
//
// Member function to generate an identical facet, but with the normal vector
// pointing at 180 degrees.
//
UTriangularFacet *UTriangularFacet::GetFlippedFacet ()
{
  UTriangularFacet *flipped = new UTriangularFacet (P0, P[1], P[0], UABSOLUTE);
  return flipped;
}

///////////////////////////////////////////////////////////////////////////////
//
// Distance (UVector3)
//
// Determines the vector between p and the closest point on the facet to p.
// This is based on the algorithm published in "Geometric Tools for Computer
// Graphics," Philip J Scheider and David H Eberly, Elsevier Science (USA),
// 2003.  at the time of writing, the algorithm is also available in a
// technical note "Distance between point and triangle in 3D," by David Eberly
// at http://www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
//
// The by-product is the square-distance sqrDist, which is retained
// in case needed by the other "Distance" member functions.
//
UVector3 UTriangularFacet::Distance (const UVector3 &p)
{
  UVector3 D  = P0 - p;
  double d       = E[0].Dot(D);
  double e       = E[1].Dot(D);
  double f       = D.Mag2();
  double q       = b*e - c*d;
  double t       = b*d - a*e;

  sqrDist          = 0.0;

  if (q+t <= det)
  {
    if (q < 0.0)
    {
      if (t < 0.0)
      {
  //
  // We are in region 4.
  //
        if (d < 0.0)
        {
          t = 0.0;
          if (-d >= a) {q = 1.0; sqrDist = a + 2.0*d + f;}
          else         {q = -d/a; sqrDist = d*q + f;}
        }
        else
        {
          q = 0.0;
          if       (e >= 0.0) {t = 0.0; sqrDist = f;}
          else if (-e >= c)   {t = 1.0; sqrDist = c + 2.0*e + f;}
          else                {t = -e/c; sqrDist = e*t + f;}
        }
      }
      else
      {
   //
   // We are in region 3.
   //
        q = 0.0;
        if      (e >= 0.0) {t = 0.0; sqrDist = f;}
        else if (-e >= c)  {t = 1.0; sqrDist = c + 2.0*e + f;}
        else               {t = -e/c; sqrDist = e*t + f;}
      }
    }
    else if (t < 0.0)
    {
   //
   // We are in region 5.
   //
      t = 0.0;
      if      (d >= 0.0) {q = 0.0; sqrDist = f;}
      else if (-d >= a)  {q = 1.0; sqrDist = a + 2.0*d + f;}
      else               {q = -d/a; sqrDist = d*q + f;}
    }
    else
    {
   //
   // We are in region 0.
   //
      q       = q / det;
      t       = t / det;
      sqrDist = q*(a*q + b*t + 2.0*d) + t*(b*q + c*t + 2.0*e) + f;
    }
  }
  else
  {
    if (q < 0.0)
    {
   //
   // We are in region 2.
   //
      double tmp0 = b + d;
      double tmp1 = c + e;
      if (tmp1 > tmp0)
      {
        double numer = tmp1 - tmp0;
        double denom = a - 2.0*b + c;
        if (numer >= denom) {q = 1.0; t = 0.0; sqrDist = a + 2.0*d + f;}
        else
        {
          q       = numer/denom;
          t       = 1.0 - q;
          sqrDist = q*(a*q + b*t +2.0*d) + t*(b*q + c*t + 2.0*e) + f;
        }
      }
      else
      {
        q = 0.0;
        if      (tmp1 <= 0.0) {t = 1.0; sqrDist = c + 2.0*e + f;}
        else if (e >= 0.0)    {t = 0.0; sqrDist = f;}
        else                  {t = -e/c; sqrDist = e*t + f;}
      }
    }
    else if (t < 0.0)
    {
   //
   // We are in region 6.
   //
      double tmp0 = b + e;
      double tmp1 = a + d;
      if (tmp1 > tmp0)
      {
        double numer = tmp1 - tmp0;
        double denom = a - 2.0*b + c;
        if (numer >= denom) {t = 1.0; q = 0.0; sqrDist = c + 2.0*e + f;}
        else
        {
          t       = numer/denom;
          q       = 1.0 - t;
          sqrDist = q*(a*q + b*t +2.0*d) + t*(b*q + c*t + 2.0*e) + f;
        }
      }
      else
      {
        t = 0.0;
        if      (tmp1 <= 0.0) {q = 1.0; sqrDist = a + 2.0*d + f;}
        else if (d >= 0.0)    {q = 0.0; sqrDist = f;}
        else                  {q = -d/a; sqrDist = d*q + f;}
      }
    }
    else
   //
   // We are in region 1.
   //
    {
      double numer = c + e - b - d;
      if (numer <= 0.0)
      {
        q       = 0.0;
        t       = 1.0;
        sqrDist = c + 2.0*e + f;
      }
      else
      {
        double denom = a - 2.0*b + c;
        if (numer >= denom) {q = 1.0; t = 0.0; sqrDist = a + 2.0*d + f;}
        else
        {
          q       = numer/denom;
          t       = 1.0 - q;
          sqrDist = q*(a*q + b*t + 2.0*d) + t*(b*q + c*t + 2.0*e) + f;
        }
      }
    }
  } 
//
//
// Do a check for rounding errors in the distance-squared.  It appears that
// the conventional methods for calculating sqrDist breaks down when very near
// to or at the surface (as required by transport).  We'll therefore also use
// the magnitude-squared of the vector displacement.  (Note that I've also
// tried to get around this problem by using the existing equations for
//
//    sqrDist = function(a,b,c,d,q,t)
//
// and use a more accurate addition process which minimises errors and
// breakdown of cummutitivity [where (A+B)+C != A+(B+C)] but this still
// doesn't work.
// Calculation from u = D + q*E[0] + t*E[1] is less efficient, but appears
// more robust.
//
  if (sqrDist < 0.0) { sqrDist = 0.0; }
  UVector3 u = D + q*E[0] + t*E[1];
  double u2     = u.Mag2();
//
//
// The following (part of the roundoff error check) is from Oliver Merle'q
// updates.
//
  if ( sqrDist > u2 ) sqrDist = u2;

  return u;
}

///////////////////////////////////////////////////////////////////////////////
//
// Distance (UVector3, double)
//
// Determines the closest distance between point p and the facet.  This makes
// use of UVector3 UTriangularFacet::Distance, which stores the
// square of the distance in variable sqrDist.  If approximate methods show 
// the distance is to be greater than minDist, then forget about further
// computation and return a very large number.
//
double UTriangularFacet::Distance (const UVector3 &p,
  const double minDist)
{
//
//
// Start with quicky test to determine if the surface of the sphere enclosing
// the triangle is any closer to p than minDist.  If not, then don't bother
// about more accurate test.
//
  double dist = UUtils::kInfinity;
  if ((p-circumcentre).Mag()-radius < minDist)
  {
//
//
// It's possible that the triangle is closer than minDist, so do more accurate
// assessment.
//
	  dist = Distance(p).Mag();
//    dist = std::sqrt(sqrDist);
  }

  return dist;
}

///////////////////////////////////////////////////////////////////////////////
//
// Distance (UVector3, double, bool)
//
// Determine the distance to point p.  UUtils::kInfinity is returned if either:
// (1) outgoing is TRUE and the dot product of the normal vector to the facet
//     and the displacement vector from p to the triangle is negative.
// (2) outgoing is FALSE and the dot product of the normal vector to the facet
//     and the displacement vector from p to the triangle is positive.
// If approximate methods show the distance is to be greater than minDist, then
// forget about further computation and return a very large number.
//
// This method has been heavily modified thanks to the valuable comments and 
// corrections of Rickard Holmberg.
//
double UTriangularFacet::Distance (const UVector3 &p,
  const double minDist, const bool outgoing)
{
//
//
// Start with quicky test to determine if the surface of the sphere enclosing
// the triangle is any closer to p than minDist.  If not, then don't bother
// about more accurate test.
//
  double dist = UUtils::kInfinity;
  if ((p-circumcentre).Mag()-radius < minDist)
  {
//
//
// It's possible that the triangle is closer than minDist, so do more accurate
// assessment.
//
    UVector3 v  = Distance(p);
    double dist1   = std::sqrt(sqrDist);
    double dir     = v.Dot(surfaceNormal);
    bool wrongSide = (dir > 0.0 && !outgoing) || (dir < 0.0 && outgoing);
    if (dist1 <= kCarTolerance)
    {
//
//
// Point p is very close to triangle.  Check if it's on the wrong side, in
// which case return distance of 0.0 otherwise .
//
      if (wrongSide) dist = 0.0;
      else           dist = dist1;
    }
    else if (!wrongSide) dist = dist1;
  }

  return dist;
}

///////////////////////////////////////////////////////////////////////////////
//
// Extent
//
// Calculates the furthest the triangle extends in a particular direction
// defined by the vector axis.
//
double UTriangularFacet::Extent (const UVector3 axis)
{
  double s  = P0.Dot(axis);
  double sp = P[0].Dot(axis);
  if (sp > s) s = sp;
  sp = P[1].Dot(axis);
  if (sp > s) s = sp;

  return s;
}

///////////////////////////////////////////////////////////////////////////////
//
// Intersect
//
// Member function to find the next intersection when going from p in the
// direction of v.  If:
// (1) "outgoing" is TRUE, only consider the face if we are going out through
//     the face.
// (2) "outgoing" is FALSE, only consider the face if we are going in through
//     the face.
// Member functions returns TRUE if there is an intersection, FALSE otherwise.
// Sets the distance (distance along w), distFromSurface (orthogonal distance)
// and normal.
//
// Also considers intersections that happen with negative distance for small
// distances of distFromSurface = 0.5*kCarTolerance in the wrong direction.
// This is to detect kSurface without doing a full Inside(p) in
// UTessellatedSolid::Distance(p,v) calculation.
//
// This member function is thanks the valuable work of Rickard Holmberg.  PT.
// However, "gotos" are the Work of the Devil have been exorcised with
// extreme prejudice!!
//
// IMPORTANT NOTE:  These calculations are predicated on v being a unit
// vector.  If UTessellatedSolid or other classes call this member function
// with |v| != 1 then there will be errors.
//
bool UTriangularFacet::Intersect (const UVector3 &p,
                   const UVector3 &v, bool outgoing, double &distance,
                         double &distFromSurface, UVector3 &normal)
{
//
//
// Check whether the direction of the facet is consistent with the vector v
// and the need to be outgoing or ingoing.  If inconsistent, disregard and
// return false.
//
  double w = v.Dot(surfaceNormal);
  if ((outgoing && (w <-dirTolerance)) || (!outgoing && (w > dirTolerance)))
  {
    distance        = UUtils::kInfinity;
    distFromSurface = UUtils::kInfinity;
    normal          = UVector3(0.0,0.0,0.0);
    return false;
  }
//
//
// Calculate the orthogonal distance from p to the surface containing the
// triangle.  Then determine if we're on the right or wrong side of the
// surface (at a distance greater than kCarTolerance) to be consistent with
// "outgoing".
//
  UVector3 D  = P0 - p;
  distFromSurface  = D.Dot(surfaceNormal);
  bool wrongSide = (outgoing && (distFromSurface < -0.5*kCarTolerance)) ||
                    (!outgoing && (distFromSurface >  0.5*kCarTolerance));
  if (wrongSide)
  {
    distance        = UUtils::kInfinity;
    distFromSurface = UUtils::kInfinity;
    normal          = UVector3(0.0,0.0,0.0);
    return false;
  }

  wrongSide = (outgoing && (distFromSurface < 0.0)) ||
             (!outgoing && (distFromSurface > 0.0));
  if (wrongSide)
  {
//
//
// We're slightly on the wrong side of the surface.  Check if we're close
// enough using a precise distance calculation.
//
    UVector3 u = Distance(p);
    if ( sqrDist <= kCarTolerance*kCarTolerance )
    {
//
//
// We're very close.  Therefore return a small negative number to pretend
// we intersect.
//
//      distance = -0.5*kCarTolerance;
      distance = 0.0;
      normal   = surfaceNormal;
      return true;
    }
    else
    {
//
//
// We're close to the surface containing the triangle, but sufficiently
// far from the triangle, and on the wrong side compared to the directions
// of the surface normal and v.  There is no intersection.
//
      distance        = UUtils::kInfinity;
      distFromSurface = UUtils::kInfinity;
      normal          = UVector3(0.0,0.0,0.0);
      return false;
    }
  }
  if (w < dirTolerance && w > -dirTolerance)
  {
//
//
// The ray is within the plane of the triangle.  Project the problem into 2D
// in the plane of the triangle.  First try to create orthogonal unit vectors
// mu and nu, where mu is E[0]/|E[0]|.  This is kinda like
// the original algorithm due to Rickard Holmberg, but with better mathematical
// justification than the original method ... however, beware Rickard's was less
// time-consuming.
//
// Note that vprime is not a unit vector.  We need to keep it unnormalised
// since the values of distance along vprime (s0 and s1) for intersection with
// the triangle will be used to determine if we cut the plane at the same
// time.
//
    UVector3 mu = E[0].Unit();
    UVector3 nu = surfaceNormal.Cross(mu);
    UVector2 pprime(p.Dot(mu),p.Dot(nu));
    UVector2 vprime(v.Dot(mu),v.Dot(nu));
    UVector2 P0prime(P0.Dot(mu),P0.Dot(nu));
    UVector2 E0prime(E[0].Mag(),0.0);
    UVector2 E1prime(E[1].Dot(mu),E[1].Dot(nu));

    UVector2 loc[2];
    if ( tGeomAlg->IntersectLineAndTriangle2D(pprime,vprime,P0prime,
                                              E0prime,E1prime,loc) )
    {
//
//
// There is an intersection between the line and triangle in 2D.  Now check
// which part of the line intersects with the plane containing the triangle
// in 3D.
//
      double vprimemag = vprime.mag();
      double s0        = (loc[0] - pprime).mag()/vprimemag;
      double s1        = (loc[1] - pprime).mag()/vprimemag;
      double normDist0 = surfaceNormal.Dot(s0*v) - distFromSurface;
      double normDist1 = surfaceNormal.Dot(s1*v) - distFromSurface;

      if ((normDist0 < 0.0 && normDist1 < 0.0) ||
          (normDist0 > 0.0 && normDist1 > 0.0) ||
          (normDist0 == 0.0 && normDist1 == 0.0) ) 
      {
        distance        = UUtils::kInfinity;
        distFromSurface = UUtils::kInfinity;
        normal          = UVector3(0.0,0.0,0.0);
        return false;
      }
      else
      {
        double dnormDist = normDist1-normDist0;
        if (std::fabs(dnormDist) < DBL_EPSILON)
        {
          distance = s0;
          normal   = surfaceNormal;
          if (!outgoing) distFromSurface = -distFromSurface;
          return true;
        }
        else
        {
          distance = s0 - normDist0*(s1-s0)/dnormDist;
          normal   = surfaceNormal;
          if (!outgoing) distFromSurface = -distFromSurface;
          return true;
        }
      }

//      UVector3 dloc   = loc1 - loc0;
//      UVector3 dlocXv = dloc.cross(v);
//      double dlocXvmag   = dlocXv.mag();
//      if (dloc.mag() <= 0.5*kCarTolerance || dlocXvmag <= DBL_EPSILON)
//      {
//        distance = loc0.mag();
//        normal = surfaceNormal;
//        if (!outgoing) distFromSurface = -distFromSurface;
//        return true;
//      }

//      UVector3 loc0Xv   = loc0.cross(v);
//      UVector3 loc1Xv   = loc1.cross(v);
//      double sameDir       = -loc0Xv.dot(loc1Xv);
//      if (sameDir < 0.0)
//      {
//        distance        = UUtils::kInfinity;
//        distFromSurface = UUtils::kInfinity;
//        normal          = UVector3(0.0,0.0,0.0);
//        return false;
//      }
//      else
//      {
//        distance = loc0.mag() + loc0Xv.mag() * dloc.mag()/dlocXvmag;
//        normal   = surfaceNormal;
//        if (!outgoing) distFromSurface = -distFromSurface;
//        return true;
//      }
    }
    else
    {
      distance        = UUtils::kInfinity;
      distFromSurface = UUtils::kInfinity;
      normal          = UVector3(0.0,0.0,0.0);
      return false;
    }
  }
//
//
// Use conventional algorithm to determine the whether there is an
// intersection.  This involves determining the point of intersection of the
// line with the plane containing the triangle, and then calculating if the
// point is within the triangle.
//
  distance         = distFromSurface / w;
  UVector3 pp = p + v*distance;
  UVector3 DD = P0 - pp;
  double d       = E[0].Dot(DD);
  double e       = E[1].Dot(DD);
  double s       = b*e - c*d;
  double t       = b*d - a*e;

  double sTolerance = (std::fabs(b)+ std::fabs(c) + std::fabs(d)
                       + std::fabs(e)) *kCarTolerance;
  double tTolerance = (std::fabs(a)+ std::fabs(b) + std::fabs(d)
                       + std::fabs(e)) *kCarTolerance;
  double detTolerance = (std::fabs(a)+ std::fabs(c)
                       + 2*std::fabs(b) ) *kCarTolerance;

  //if (s < 0.0 || t < 0.0 || s+t > det)
  if (s < -sTolerance || t < -tTolerance || ( s+t - det ) > detTolerance)
  {
//
//
// The intersection is outside of the triangle.
//
    distance        = UUtils::kInfinity;
    distFromSurface = UUtils::kInfinity;
    normal          = UVector3(0.0,0.0,0.0);
    return false;
  }
  else
  {
//
//
// There is an intersection.  Now we only need to set the surface normal.
//
     normal = surfaceNormal;
     if (!outgoing) distFromSurface = -distFromSurface;
     return true;
  }
}

////////////////////////////////////////////////////////////////////////
//
// GetPointOnFace
//```
// Auxiliary method for get a random point on surface

UVector3 UTriangularFacet::GetPointOnFace() const
{
  double alpha = UUtils::RandomUniform(0, 1);
  double beta = UUtils::RandomUniform(0, 1);
  double lambda1=alpha*beta;
  double lambda0=alpha-lambda1;
  
  return (P0 + lambda0*E[0] + lambda1*E[1]);
}

////////////////////////////////////////////////////////////////////////
//
// GetArea
//
// Auxiliary method for returning the surface area

double UTriangularFacet::GetArea()
{
  return area;
}
////////////////////////////////////////////////////////////////////////
//
