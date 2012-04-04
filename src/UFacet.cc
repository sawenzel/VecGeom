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
// $Id: UFacet.cc,v 1.11 2010-09-23 10:30:07 gcosmo Exp $
// GEANT4 tag $Name: not supported by cvs2svn $
//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// MODULE:              UFacet.hh
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

#include "UFacet.hh"
#include "VUSolid.hh"

using namespace std;

//typedef vector<UVector3> vector<UVector3>;

///////////////////////////////////////////////////////////////////////////////
//
UFacet::UFacet ()
  : geometryType("UFacet"), isDefined(false), nVertices(0),
    radius(0.), radiusSqr(0.), dirTolerance(1.0E-14), area(0.)
{
	kCarTolerance = VUSolid::Tolerance();

  P.clear();
  E.clear();
    
  circumcentre = UVector3(0.0,0.0,0.0);
}

///////////////////////////////////////////////////////////////////////////////
//
UFacet::~UFacet ()
{
  P.clear();
  E.clear();
}

///////////////////////////////////////////////////////////////////////////////
//
UFacet::UFacet (const UFacet &rhs)
  : geometryType(rhs.geometryType), isDefined(rhs.isDefined),
    nVertices(rhs.nVertices), P0(rhs.P0), P(rhs.P), E(rhs.E), I(rhs.I),
    surfaceNormal(rhs.surfaceNormal), circumcentre(rhs.circumcentre),
    radius(rhs.radius), radiusSqr(rhs.radiusSqr),
    dirTolerance(rhs.dirTolerance), kCarTolerance(rhs.kCarTolerance),
    area(rhs.area)
{
}

///////////////////////////////////////////////////////////////////////////////
//
const UFacet &UFacet::operator=(UFacet &rhs)
{
   // Check assignment to self
   //
   if (this == &rhs)  { return *this; }

   // Copy data
   //
   geometryType = rhs.geometryType; isDefined = rhs.isDefined;
   nVertices = rhs.nVertices; P0 = rhs.P0; P = rhs.P; E = rhs.E; I = rhs.I;
   surfaceNormal = rhs.surfaceNormal; circumcentre = rhs.circumcentre;
   radius = rhs.radius; radiusSqr = rhs.radiusSqr;
   dirTolerance = rhs.dirTolerance; kCarTolerance = rhs.kCarTolerance;
   area = rhs.area;

   return *this;
}

///////////////////////////////////////////////////////////////////////////////
//
bool UFacet::operator== (const UFacet &right) const
{
  double tolerance = kCarTolerance*kCarTolerance/4.0;
  if (nVertices != right.GetNumberOfVertices())
    { return false; }
  else if ((circumcentre-right.GetCircumcentre()).Mag2() > tolerance)
    { return false; }
  else if (std::fabs((right.GetSurfaceNormal()).Dot(surfaceNormal)) < 0.9999999999)
    { return false; }

  bool coincident  = true;
  size_t i           = 0;
  do
  {
    coincident = false;
    size_t j   = 0;
    do
    {
		coincident = (GetVertex(i)-right.GetVertex(j)).Mag2() < tolerance;
    } while (!coincident && ++j < nVertices);
  } while (coincident && ++i < nVertices);
  
  return coincident;
}

///////////////////////////////////////////////////////////////////////////////
//
void UFacet::ApplyTranslation(const UVector3 v)
{
  P0 += v;
  for (vector<UVector3>::iterator it=P.begin(); it!=P.end(); it++)
  {
    (*it) += v;
  }
}

///////////////////////////////////////////////////////////////////////////////
//
std::ostream &UFacet::StreamInfo(std::ostream &os) const
{
  os << endl;
  os << "*********************************************************************"
     << endl;
  os << "FACET TYPE       = " << geometryType << endl;
  os << "ABSOLUTE VECTORS = " << endl;
  os << "P0               = " << P0 << endl;
  for (vector<UVector3>::const_iterator it=P.begin(); it!=P.end(); it++)
    { os << "P[" << it-P.begin()+1 << "]      = " << *it << endl; }

  os << "RELATIVE VECTORS = " << endl;
  for (vector<UVector3>::const_iterator it=E.begin(); it!=E.end(); it++)
    { os << "E[" << it-E.begin()+1 << "]      = " << *it << endl; }

  os << "*********************************************************************"
     << endl;
  
  return os;
}

///////////////////////////////////////////////////////////////////////////////
//
UFacet* UFacet::GetClone ()
  {return 0;}

///////////////////////////////////////////////////////////////////////////////
//
double UFacet::Distance (const UVector3&, const double)
{return UUtils::kInfinity;}

///////////////////////////////////////////////////////////////////////////////
//
double UFacet::Distance (const UVector3&, const double,
                                    const bool)
  {return UUtils::kInfinity;}

///////////////////////////////////////////////////////////////////////////////
//
double UFacet::Extent (const UVector3)
  {return 0.0;}

///////////////////////////////////////////////////////////////////////////////
//
bool UFacet::Intersect (const UVector3&, const UVector3 &,
                            const bool , double &, double &,
                                  UVector3 &)
  {return false;}
 