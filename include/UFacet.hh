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
// $Id: UFacet.hh,v 1.8 2010-09-23 10:27:25 gcosmo Exp $
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
//
// Class description:
//
//   Base class defining the facets which are components of a
//   G4TessellatedSolid shape.
//

///////////////////////////////////////////////////////////////////////////////
#ifndef UFacet_hh
#define UFacet_hh

//#include "G4VSolid.hh"
#include <iostream>
#include <vector>

#include "UVector3.hh"

enum UFacetVertexType {UABSOLUTE, URELATIVE};

class UFacet
{
  public:  // with description

    UFacet ();
    virtual ~UFacet ();

    bool operator== (const UFacet &right) const;

    inline int GetNumberOfVertices () const;
    inline UVector3 GetVertex (int i) const;
    
	inline std::string GetEntityType () const;
    inline UVector3 GetSurfaceNormal () const;
    inline bool IsInside(const UVector3 &p) const;
    inline bool IsDefined () const;
    inline void SetVertexIndex (const int i, const int j);
    inline int GetVertexIndex (const int i) const;
    inline UVector3 GetCircumcentre () const;
    inline double GetRadius () const;
    inline double GetRadiusSquared() const;
    
    void ApplyTranslation (const UVector3 v);
    
    std::ostream &StreamInfo(std::ostream &os) const;

    virtual UFacet *GetClone ();
    virtual double Distance (const UVector3&, const double);
    virtual double Distance (const UVector3&, const double,
                               const bool);
    virtual double Extent   (const UVector3);
    virtual bool Intersect  (const UVector3&, const UVector3 &,
                               const bool , double &, double &,
                                     UVector3 &);
    virtual double GetArea() = 0;
    virtual UVector3 GetPointOnFace() const = 0;

  public:  // without description

    UFacet (const UFacet &right);
    const UFacet &operator=(UFacet &right);

  protected:

    std::string geometryType;
    bool               isDefined;
    int               nVertices;
    UVector3        P0;
    std::vector<UVector3>    P;
    std::vector<UVector3>    E;
    std::vector<int>  I;
    UVector3        surfaceNormal;
    UVector3        circumcentre;
    double             radius;
    double             radiusSqr;

    double             dirTolerance;
    double             kCarTolerance;
    double             area;
};

typedef std::vector<UFacet*>::iterator       UFacetI;
typedef std::vector<UFacet*>::const_iterator UFacetCI;

///////////////////////////////////////////////////////////////////////////////
//
inline UVector3 UFacet::GetSurfaceNormal () const
  {return surfaceNormal;}

///////////////////////////////////////////////////////////////////////////////
//
inline std::string UFacet::GetEntityType () const
  {return geometryType;}

///////////////////////////////////////////////////////////////////////////////
//
inline bool UFacet::IsInside (const UVector3 &p) const
{
  UVector3 D       = p - P0;
  double displacement = D.Dot(surfaceNormal);
  bool inside         = (displacement <= 0.0);
  
  return inside;
}

///////////////////////////////////////////////////////////////////////////////
//
inline bool UFacet::IsDefined () const
  {return isDefined;}

///////////////////////////////////////////////////////////////////////////////
//
inline int UFacet::GetVertexIndex (const int i) const
{
  if (i < (int) I.size()) { return I[i]; }
  else              { return 999999999; }
}

///////////////////////////////////////////////////////////////////////////////
//
inline int UFacet::GetNumberOfVertices () const
{
  return nVertices;
}

///////////////////////////////////////////////////////////////////////////////
//
inline void UFacet::SetVertexIndex (const int i, const int j)
  {I[i] = j;}	

///////////////////////////////////////////////////////////////////////////////
//
inline UVector3 UFacet::GetVertex (int i) const
{
  if (i == 0)             { return P0; }
  else if (i < nVertices) { return P[i-1]; }
  else                    { return UVector3(0.0,0.0,0.0); }
}

///////////////////////////////////////////////////////////////////////////////
//
inline UVector3 UFacet::GetCircumcentre () const
  {return circumcentre;}

///////////////////////////////////////////////////////////////////////////////
//
inline double UFacet::GetRadius () const
  {return radius;}

///////////////////////////////////////////////////////////////////////////////
//
inline double UFacet::GetRadiusSquared () const
  {return radiusSqr;}

#endif
