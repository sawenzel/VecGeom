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
//
// $Id: UQuadrangularFacet.cc,v 1.9 2010-09-23 10:27:25 gcosmo Exp $
// GEANT4 tag $Name: not supported by cvs2svn $
//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// MODULE:              UQuadrangularFacet.cc
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
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#include "UQuadrangularFacet.hh"

#include "UUtils.hh"

using namespace std;

///////////////////////////////////////////////////////////////////////////////
//
// !!!THIS IS A FUDGE!!!  IT'S TWO ADJACENT G4TRIANGULARFACETS
// --- NOT EFFICIENT BUT PRACTICAL.
//
UQuadrangularFacet::UQuadrangularFacet (const UVector3 Pt0,
                 const UVector3 vt1, const UVector3 vt2,
                 const UVector3 vt3, UFacetVertexType vertexType)
  : UFacet(), facet1(0), facet2(0)
{
  P0        = Pt0;
  nVertices = 4;
  if (vertexType == UABSOLUTE)
  {
    P.push_back(vt1);
    P.push_back(vt2);
    P.push_back(vt3);
  
    E.push_back(vt1 - P0);
    E.push_back(vt2 - P0);
    E.push_back(vt3 - P0);
  }
  else
  {
    P.push_back(P0 + vt1);
    P.push_back(P0 + vt2);
    P.push_back(P0 + vt3);
  
    E.push_back(vt1);
    E.push_back(vt2);
    E.push_back(vt3);
  }

  double length1 = E[0].Mag();
  double length2 = (P[1]-P[0]).Mag();
  double length3 = (P[2]-P[1]).Mag();
  double length4 = E[2].Mag();
  
  UVector3 normal1 = E[0].Cross(E[1]).Unit();
  UVector3 normal2 = E[1].Cross(E[2]).Unit(); 
  
  if (length1 <= kCarTolerance || length2 <= kCarTolerance ||
      length3 <= kCarTolerance || length4 <= kCarTolerance ||
	  normal1.Dot(normal2) < 0.9999999999)
  {
//    G4Exception("UQuadrangularFacet::UQuadrangularFacet()", "InvalidSetup", JustWarning, "Length of sides of facet are too small or sides not planar.");


    cerr << endl;
    cerr << "P0 = " << P0   << endl;
    cerr << "P1 = " << P[0] << endl;
    cerr << "P2 = " << P[1] << endl;
    cerr << "P3 = " << P[2] << endl;
    cerr << "Side lengths = P0->P1" << length1 << endl;    
    cerr << "Side lengths = P1->P2" << length2 << endl;    
    cerr << "Side lengths = P2->P3" << length3 << endl;    
    cerr << "Side lengths = P3->P0" << length4 << endl;    
    cerr << endl;
    
    isDefined     = false;
    geometryType  = "UQuadragularFacet";
    surfaceNormal = UVector3(0.0,0.0,0.0);
  }
  else
  {
    isDefined     = true;
    geometryType  = "UQuadrangularFacet";
    
    facet1 = new UTriangularFacet(P0,P[0],P[1],UABSOLUTE);
    facet2 = new UTriangularFacet(P0,P[1],P[2],UABSOLUTE);
    surfaceNormal = normal1;
    
    UVector3 vtmp = 0.5 * (E[0] + E[1]);
    circumcentre       = P0 + vtmp;
	radiusSqr          = vtmp.Mag2();
    radius             = std::sqrt(radiusSqr);
  
    for (size_t i=0; i<4; i++) I.push_back(0);
  }
}

///////////////////////////////////////////////////////////////////////////////
//
UQuadrangularFacet::~UQuadrangularFacet ()
{
  delete facet1;
  delete facet2;
  
  P.clear();
  E.clear();
  I.clear();
}

///////////////////////////////////////////////////////////////////////////////
//
UQuadrangularFacet::UQuadrangularFacet (const UQuadrangularFacet &rhs)
  : UFacet(rhs)
{
  facet1 = new UTriangularFacet(*(rhs.facet1));
  facet2 = new UTriangularFacet(*(rhs.facet2));
}

///////////////////////////////////////////////////////////////////////////////
//
const UQuadrangularFacet &
UQuadrangularFacet::operator=(UQuadrangularFacet &rhs)
{
   // Check assignment to self
   //
   if (this == &rhs)  { return *this; }

   // Copy base class data
   //
   UFacet::operator=(rhs);

   // Copy data
   //
   delete facet1; facet1 = new UTriangularFacet(*(rhs.facet1));
   delete facet2; facet2 = new UTriangularFacet(*(rhs.facet2));

   return *this;
}

///////////////////////////////////////////////////////////////////////////////
//
UFacet *UQuadrangularFacet::GetClone ()
{
  UQuadrangularFacet *c =
    new UQuadrangularFacet (P0, P[0], P[1], P[2], UABSOLUTE);
  UFacet *cc         = 0;
  cc                   = c;
  return cc;
}

///////////////////////////////////////////////////////////////////////////////
//
UVector3 UQuadrangularFacet::Distance (const UVector3 &p)
{
  UVector3 v1 = facet1->Distance(p);
  UVector3 v2 = facet2->Distance(p);
  
  if (v1.Mag2() < v2.Mag2()) return v1;
  else return v2;
}

///////////////////////////////////////////////////////////////////////////////
//
double UQuadrangularFacet::Distance (const UVector3 &p,
  const double)
{
  /*UVector3 D  = P0 - p;
  double d       = E[0].dot(D);
  double e       = E[1].dot(D);
  double s       = b*e - c*d;
  double t       = b*d - a*e;*/
  double dist = UUtils::kInfinity;
  
  /*if (s+t > 1.0 || s < 0.0 || t < 0.0)
  {
    UVector3 D0 = P0 - p;
    UVector3 D1 = P[0] - p;
    UVector3 D2 = P[1] - p;
    
    double d0 = D0.mag();
    double d1 = D1.mag();
    double d2 = D2.mag();
    
    dist = min(d0, min(d1, d2));
    if (dist > minDist) return kInfinity;
  }*/
  
  dist = Distance(p).Mag();
  
  return dist;
}

///////////////////////////////////////////////////////////////////////////////
//
double UQuadrangularFacet::Distance (const UVector3 &p,
                                        const double, const bool outgoing)
{
  /*UVector3 D  = P0 - p;
  double d       = E[0].dot(D);
  double e       = E[1].dot(D);
  double s       = b*e - c*d;
  double t       = b*d - a*e;*/
  double dist = UUtils::kInfinity;
  
  /*if (s+t > 1.0 || s < 0.0 || t < 0.0)
  {
    UVector3 D0 = P0 - p;
    UVector3 D1 = P[0] - p;
    UVector3 D2 = P[1] - p;
    
    double d0 = D0.mag();
    double d1 = D1.mag();
    double d2 = D2.mag();
    
    dist = min(d0, min(d1, d2));
    if (dist > minDist ||
      (D0.dot(surfaceNormal) > 0.0 && !outgoing) ||
      (D0.dot(surfaceNormal) < 0.0 && outgoing)) return kInfinity;
  }*/
  
  UVector3 v = Distance(p);
  double dir    = v.Dot(surfaceNormal);
  if ((dir > dirTolerance && !outgoing) ||
      (dir <-dirTolerance && outgoing)) dist = UUtils::kInfinity;
  else dist = v.Mag();
  
  return dist;
}

///////////////////////////////////////////////////////////////////////////////
//
double UQuadrangularFacet::Extent (const UVector3 axis)
{
	double s  = P0.Dot(axis);
  for (vector<UVector3>::iterator it=P.begin(); it!=P.end(); it++)
  {
    double sp = it->Dot(axis);
    if (sp > s) s = sp;
  }

  return s;
}

///////////////////////////////////////////////////////////////////////////////
//
bool UQuadrangularFacet::Intersect (const UVector3 &p,
  const UVector3 &v, bool outgoing, double &distance,
  double &distFromSurface, UVector3 &normal)
{
  bool intersect =
    facet1->Intersect(p,v,outgoing,distance,distFromSurface,normal);
  if (!intersect)
  {
    intersect = facet2->Intersect(p,v,outgoing,distance,distFromSurface,normal);
  }
  
  if (!intersect)
  {
    distance        = UUtils::kInfinity;
    distFromSurface = UUtils::kInfinity;
    normal          = UVector3(0.0,0.0,0.0);
  }
  
  return intersect;
}

////////////////////////////////////////////////////////////////////////
//
// GetPointOnFace
//
// Auxiliary method for get a random point on surface

UVector3 UQuadrangularFacet::GetPointOnFace() const
{
  UVector3 pr;

  if ( UUtils::RandomUniform(0,1) < 0.5 )
  {
    pr = facet1->GetPointOnFace();
  }
  else
  {
    pr = facet2->GetPointOnFace();
  }

  return pr;
}

////////////////////////////////////////////////////////////////////////
//
// GetArea
//
// Auxiliary method for returning the surface area

double UQuadrangularFacet::GetArea()
{
  if (!area)  { area = facet1->GetArea() + facet2->GetArea(); }

  return area;
}
