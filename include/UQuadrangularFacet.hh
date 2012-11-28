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
// $Id: UQuadrangularFacet.hh,v 1.6 2008-12-18 12:57:24 gunter Exp $
// GEANT4 tag $Name: not supported by cvs2svn $
//
// Class description:
//
//   The UQuadrangularFacet class is used for the contruction of
//   G4TessellatedSolid.
//   It is defined by four fVertices, which shall be in the same plane and be
//   supplied in anti-clockwise order looking from the outsider of the solid
//   where it belongs. Its constructor
//   
//     UQuadrangularFacet (const UVector3 Pt0, const UVector3 vt1,
//                          const UVector3 vt2, const UVector3 vt3,
//                          G4FacetVertexType);
//
//   takes 5 parameters to define the four fVertices:
//     1) G4FacetvertexType = "ABSOLUTE": in this case Pt0, vt1, vt2 and vt3
//        are the four fVertices required in anti-clockwise order when looking
//        from the outsider.
//     2) G4FacetvertexType = "RELATIVE": in this case the first vertex is Pt0,
//        the second vertex is Pt0+vt, the third vertex is Pt0+vt2 and 
//        the fourth vertex is Pt0+vt3, in anti-clockwise order when looking 
//        from the outsider.
//
// Author: Marek Gayer, created from original implementation by P R Truscott, 2004
//
// History:
// 17.10.12 Marek Gayer, initial ver
// --------------------------------------------------------------------
//


#ifndef UQuadrangularFacet_HH
#define UQuadrangularFacet_HH 1

#include "VUFacet.hh"
#include "UTriangularFacet.hh"
#include "UVector3.hh"
#include "UQuadrangularFacet.hh"

class UQuadrangularFacet : public VUFacet
{
public:  // with description

	UQuadrangularFacet (const UVector3 &Pt0, const UVector3 &vt1,
		const UVector3 &vt2, const UVector3 &vt3,
		UFacetVertexType);
	virtual ~UQuadrangularFacet ();

	UQuadrangularFacet (const UQuadrangularFacet &right);
	UQuadrangularFacet &operator=(const UQuadrangularFacet &right);    

	VUFacet *GetClone ();

	UVector3 Distance (const UVector3 &p);
	double Distance (const UVector3 &p, const double minDist);
	double Distance (const UVector3 &p, const double minDist,
		const bool outgoing);
	double Extent   (const UVector3 axis);
	bool Intersect  (const UVector3 &p, const UVector3 &v,
		const bool outgoing, double &distance,
		double &distFromSurface, UVector3 &normal);

	double GetArea ();
	UVector3 GetPointOnFace () const;

	virtual UGeometryType GetEntityType () const;

	inline int GetNumberOfVertices () const
	{
		return 4;
	}

	UVector3 GetVertex (int i) const
	{
		return i == 3 ? fFacet2.GetVertex(2) : fFacet1.GetVertex(i);
	}

	UVector3 GetSurfaceNormal () const;

	inline double GetRadius () const
	{
		return fRadius;
	}

	inline UVector3 GetCircumcentre () const
	{
		return fCircumcentre;
	}

	inline void SetVertex (int i, const UVector3 &val)
	{
		switch (i)
		{
			case 0:
				fFacet1.SetVertex(0, val);
				fFacet2.SetVertex(0, val);
				break;
			case 1:
				fFacet1.SetVertex(1, val);
				break;
			case 2:
				fFacet1.SetVertex(2, val);
				fFacet2.SetVertex(1, val);
				break;
			case 3:
				fFacet2.SetVertex(2, val);
				break;
		}
	}

	inline void SetVertices(std::vector<UVector3> *v)
	{
		fFacet1.SetVertices(v);
		fFacet2.SetVertices(v);
	}

	inline bool IsDefined () const
	{
		return fFacet1.IsDefined();
	}

protected:
private:

	inline int GetVertexIndex (int i) const
	{
		return i == 3 ? fFacet2.GetVertexIndex(2) : fFacet1.GetVertexIndex(i);
	}

	inline void SetVertexIndex (int i, int val)
	{
		switch (i)
		{
			case 0:
				fFacet1.SetVertexIndex(0, val);
				fFacet2.SetVertexIndex(0, val);
				break;
			case 1:
				fFacet1.SetVertexIndex(1, val);
				break;
			case 2:
				fFacet1.SetVertexIndex(2, val);
				fFacet2.SetVertexIndex(1, val);
				break;
			case 3:
				fFacet2.SetVertexIndex(2, val);
				break;
		}
	}

	double fRadius;
	
	UVector3 fCircumcentre;

	int AllocatedMemory()
	{
		return sizeof(*this) + fFacet1.AllocatedMemory() + fFacet2.AllocatedMemory();
	}

	UTriangularFacet fFacet1, fFacet2;
};

#endif
