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
// $Id: G4TriangularFacet.hh,v 1.8 2007-12-10 16:30:18 gunter Exp $
// GEANT4 tag $Name: not supported by cvs2svn $
//
// Author: Marek Gayer, started from original implementation by P R Truscott, 2004
//
//
// Class description:
//
//   The G4TriangularFacet class is used for the contruction of
//   G4TessellatedSolid.
//   It is defined by three fVertices, which shall be supplied in anti-clockwise
//   order looking from the outsider of the solid where it belongs.
//   Its constructor:
//   
//      G4TriangularFacet (const UVector3 Pt0, const UVector3 vt1,
//                         const UVector3 vt2, G4FacetVertexType);
//
//   takes 4 parameters to define the three fVertices:
//      1) G4FacetvertexType = "ABSOLUTE": in this case Pt0, vt1 and vt2 are 
//         the 3 fVertices in anti-clockwise order looking from the outsider.
//      2) G4FacetvertexType = "RELATIVE": in this case the first vertex is Pt0,
//         the second vertex is Pt0+vt1 and the third vertex is Pt0+vt2, all  
//         in anti-clockwise order when looking from the outsider.

///////////////////////////////////////////////////////////////////////////////
#ifndef UTriangularFacet_hh
#define UTriangularFacet_hh 1

#include "VUFacet.hh"
#include "UVector3.hh"
#include "UTessellatedGeometryAlgorithms.hh"
#include "UTriangularFacet.hh"

class UTriangularFacet : public VUFacet
{

public:

	UTriangularFacet (const UVector3 &vt0, const UVector3 &vt1, const UVector3 &vt2, UFacetVertexType);

	UTriangularFacet ();

	~UTriangularFacet ();

	UTriangularFacet (const UTriangularFacet &right);

	UTriangularFacet &operator=(const UTriangularFacet &right);    

	VUFacet *GetClone ();
	UTriangularFacet *GetFlippedFacet ();

	UVector3 Distance (const UVector3 &p);
	double Distance (const UVector3 &p, const double minDist);
	double Distance (const UVector3 &p, const double minDist, const bool outgoing);
	double Extent   (const UVector3 axis);
	bool Intersect  (const UVector3 &p, const UVector3 &v, const bool outgoing, double &distance, double &distFromSurface, UVector3 &normal);
	double GetArea ();
	UVector3 GetPointOnFace () const;

	UVector3 GetSurfaceNormal () const;

	inline bool IsDefined () const
	{
		return fIsDefined;
	}

	UGeometryType GetEntityType () const;

	inline int GetNumberOfVertices () const
	{
		return 3;
	}

	UVector3 GetVertex (int i) const
	{		  
		int indice = fIndices[i];
		return indice < 0 ? (*fVertices)[i] : (*fVertices)[indice];
	}

	inline void SetVertex (int i, const UVector3 &val)
	{
		(*fVertices)[i] = val;
	}

	inline UVector3 GetCircumcentre () const
	{
		return fCircumcentre;
	}

	inline double GetRadius () const
	{
		return fRadius;
	}

	void SetSurfaceNormal (UVector3 normal);

	int AllocatedMemory()
	{
		int size = sizeof(*this);
		//		size += geometryType.length();
		size += GetNumberOfVertices() * sizeof(UVector3);
		//7		size += E.size() * sizeof(UVector3);
		return size;
	}

	inline int GetVertexIndex (int i) const
	{
		return fIndices[i];
	}

	inline void SetVertexIndex (int i, int j)
	{
		fIndices[i] = j;
	}

	inline void SetVertices(std::vector<UVector3> *v)
	{
		if (fIndices[0] < 0 && fVertices)
		{
			delete fVertices;
			fVertices = NULL;
		}
		fVertices = v;
	}

private:

	UVector3 fSurfaceNormal;
	double fArea;
	UVector3 fCircumcentre;
	double fRadius;
	int fIndices[3];

	std::vector<UVector3> *fVertices;

	void CopyFrom(const UTriangularFacet &rhs);

private:

	double fA, fB, fC;
	double fDet;
	double fSqrDist;
	UVector3 fE1, fE2;
	bool fIsDefined;
};

#endif
