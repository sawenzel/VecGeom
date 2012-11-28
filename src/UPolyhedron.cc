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
// $Id: UPolyhedron.cc,v 1.35 2010-12-07 09:36:59 allison Exp $
// GEANT4 tag $Name: not supported by cvs2svn $
//
// 
//
// U Polyhedron library
//
// History:
// 23.07.96 E.Chernyaev <Evgueni.Tcherniaev@cern.ch> - initial version
//
// 30.09.96 E.Chernyaev
// - added GetNextVertexIndex, GetVertex by Yasuhide Sawada
// - added GetNextUnitNormal, GetNextEdgeIndeces, GetNextEdge
//
// 15.12.96 E.Chernyaev
// - added GetNumberOfRotationSteps, RotateEdge, RotateAroundZ, SetReferences
// - rewritten UPolyhedronCons;
// - added UPolyhedronPara, ...Trap, ...Pgon, ...Pcon, ...Sphere, ...Torus
//
// 01.06.97 E.Chernyaev
// - modified RotateAroundZ, added SetSideFacets
//
// 19.03.00 E.Chernyaev
// - implemented boolean operations (add, subtract, intersect) on polyhedra;
//
// 25.05.01 E.Chernyaev
// - added GetSurfaceArea() and GetVolume();
//
// 05.11.02 E.Chernyaev
// - added createTwistedTrap() and createPolyhedron();
//
// 20.06.05 G.Cosmo
// - added UPolyhedronEllipsoid;
//
// 18.07.07 T.Nikitin
// - added UParaboloid;

/*
#include "UPolyhedron.h"
#include "UPhysicalConstants.hh"
#include "UVector3D.hh"
*/

#include <cstdlib>	// Required on some comUUtils::kPilers for std::abs(int) ...
#include <cmath>
#include <iostream>
#include "UPolyhedron.hh"
#include "VUSolid.hh"

static const double perMillion  = 0.000001;
static const double deg = (UUtils::kPi/180.0);

/*
using CLU::perMillion;
using CLU::deg;
using CLU::UUtils::kPi;
*/

/***********************************************************************
 *																																		 *
 * Name: UPolyhedron operator <<									 Date:		09.05.96 *
 * Author: E.Chernyaev (IU/Protvino)							 Revised:					*
 *																																		 *
 * Function: Print contents of U polyhedron													 *
 *																																		 *
 ***********************************************************************/
std::ostream & operator<<(std::ostream &ostr, const UFacet &facet) {
	for (int k=0; k<4; k++) {
		ostr << " " << facet.edge[k].v << "/" << facet.edge[k].f;
	}
	return ostr;
}

std::ostream & operator<<(std::ostream &ostr, const UPolyhedron &ph) {
	ostr << std::endl;
	ostr << "No. of vertices=" << ph.GetNoVertices() << ", No. of facets=" << ph.GetNoFacets() << std::endl;
	int i;
	for (i=1; i<=ph.GetNoVertices(); i++) {
		 ostr << "xyz(" << i << ")="
					<< ph.fVertices[i].x << ' ' << ph.fVertices[i].y << ' ' << ph.fVertices[i].z
					<< std::endl;
	}
	for (i=1; i<=ph.GetNoFacets(); i++) {
		ostr << "face(" << i << ")=" << ph.fFacets[i] << std::endl;
	}
	return ostr;
}

UPolyhedron::UPolyhedron(const UPolyhedron &from)
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron copy constructor						 Date:		23.07.96	*
 * Author: E.Chernyaev (IU/Protvino)							Revised:					 *
 *																																		 *
 ***********************************************************************/
: fVertices(0), fFacets(0)
{
	AllocateMemory(from.GetNoVertices(), from.GetNoFacets());
	int n = GetNoVertices();
	for (int i=1; i<=n; i++) fVertices[i] = from.fVertices[i];
	n = GetNoFacets();
	for (int k=1; k<=n; k++) fFacets[k] = from.fFacets[k];
}

UPolyhedron &UPolyhedron::operator=(const UPolyhedron &from)
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron operator =									 Date:		23.07.96	*
 * Author: E.Chernyaev (IU/Protvino)							Revised:					 *
 *																																		 *
 * Function: Copy contents of one polyhedron to another								*
 *																																		 *
 ***********************************************************************/
{
	if (this != &from) {
		AllocateMemory(from.GetNoVertices(), from.GetNoFacets());
		int n = GetNoVertices();
		for (int i=1; i<=n; i++) fVertices[i] = from.fVertices[i];
		int nface = GetNoFacets();
		for (int k=1; k<=nface; k++) fFacets[k] = from.fFacets[k];
	}
	return *this;
}

int
UPolyhedron::FindNeighbour(int iFace, int iNode, int iOrder) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::FindNeighbour								Date:		22.11.99 *
 * Author: E.Chernyaev															 Revised:					*
 *																																		 *
 * Function: Find neighbouring face																		*
 *																																		 *
 ***********************************************************************/
{
	int i;
	for (i=0; i<4; i++) {
		if (iNode == std::abs(fFacets[iFace].edge[i].v)) break;
	}
	if (i == 4) {
		std::cerr
			<< "UPolyhedron::FindNeighbour: face " << iFace
			<< " has no node " << iNode
			<< std::endl; 
		return 0;
	}
	if (iOrder < 0) {
		if ( --i < 0) i = 3;
		if (fFacets[iFace].edge[i].v == 0) i = 2;
	}
	return (fFacets[iFace].edge[i].v > 0) ? 0 : fFacets[iFace].edge[i].f;
}

UNormal3D UPolyhedron::FindNodeNormal(int iFace, int iNode) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::FindNodeNormal							 Date:		22.11.99 *
 * Author: E.Chernyaev															 Revised:					*
 *																																		 *
 * Function: Find normal at given node																 *
 *																																		 *
 ***********************************************************************/
{
	UNormal3D	 normal = GetUnitNormal(iFace);
	int					k = iFace, iOrder = 1, n = 1;

	while (k = FindNeighbour(k, iNode, iOrder), k != iFace) 
	{
		if (k > 0) {
			n++;
			normal += GetUnitNormal(k);
		}else{
			if (iOrder < 0) break;
			k = iFace;
			iOrder = -iOrder;
		}
	}
	return normal.Unit();
}

int UPolyhedron::GetNumberOfRotationSteps()
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetNumberOfRotationSteps		 Date:		24.06.97 *
 * Author: J.Allison (Manchester University)				 Revised:					*
 *																																		 *
 * Function: Get number of steps for whole circle											*
 *																																		 *
 ***********************************************************************/
{
	return fNumberOfRotationSteps;
}

void UPolyhedron::SetNumberOfRotationSteps(int n)
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::SetNumberOfRotationSteps		 Date:		24.06.97 *
 * Author: J.Allison (Manchester University)				 Revised:					*
 *																																		 *
 * Function: Set number of steps for whole circle											*
 *																																		 *
 ***********************************************************************/
{
	const int nMin = 3;
	if (n < nMin) {
		std::cerr 
			<< "UPolyhedron::SetNumberOfRotationSteps: attempt to Set the\n"
			<< "number of steps per circle < " << nMin << "; forced to " << nMin
			<< std::endl;
		fNumberOfRotationSteps = nMin;
	}else{
		fNumberOfRotationSteps = n;
	}		
}

void UPolyhedron::ResetNumberOfRotationSteps()
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetNumberOfRotationSteps		 Date:		24.06.97 *
 * Author: J.Allison (Manchester University)				 Revised:					*
 *																																		 *
 * Function: Reset number of steps for whole circle to default value	 *
 *																																		 *
 ***********************************************************************/
{
	fNumberOfRotationSteps = DEFAULT_NUMBER_OF_STEPS;
}

void UPolyhedron::AllocateMemory(int nvert, int nface)
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::AllocateMemory							 Date:		19.06.96 *
 * Author: E.Chernyaev (IU/Protvino)							 Revised: 05.11.02 *
 *																																		 *
 * Function: Allocate memory for GEANT4 polyhedron										 *
 *																																		 *
 * Input: Nvert - number of nodes																			*
 *				Nface - number of faces																			*
 *																																		 *
 ***********************************************************************/
{
	if (GetNoVertices() == nvert && nface == GetNoFacets()) return;
	if (nvert > 0 && nface > 0) {
		fVertices.resize(nvert+1);
		fFacets.resize(nface+1);
	}else{
		fVertices.resize(1); fFacets.resize(1);
	}
}

void UPolyhedron::CreatePrism()
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::CreatePrism									Date:		15.07.96 *
 * Author: E.Chernyaev (IU/Protvino)							 Revised:					*
 *																																		 *
 * Function: Set facets for a prism																		*
 *																																		 *
 ***********************************************************************/
{
	enum {DUMMY, BOTTOM, LEFT, BACK, RIGHT, FRONT, TOP};

	fFacets[BOTTOM] = UFacet(1,LEFT,	4,BACK,	3,RIGHT,	2,FRONT);
	fFacets[LEFT] = UFacet(5,TOP,	 8,BACK,	4,BOTTOM, 1,FRONT);
	fFacets[BACK] = UFacet(8,TOP,	 7,RIGHT, 3,BOTTOM, 4,LEFT);
	fFacets[RIGHT] = UFacet(7,TOP,	 6,FRONT, 2,BOTTOM, 3,BACK);
	fFacets[FRONT] = UFacet(6,TOP,	 5,LEFT,	1,BOTTOM, 2,RIGHT);
	fFacets[TOP] = UFacet(5,FRONT, 6,RIGHT, 7, BACK,8,LEFT);
}

void UPolyhedron::RotateEdge(int k1, int k2, double r1, double r2,
															int v1, int v2, int vEdge,
															bool ifWholeCircle, int nds, int &kface)
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::RotateEdge									 Date:		05.12.96 *
 * Author: E.Chernyaev (IU/Protvino)							 Revised:					*
 *																																		 *
 * Function: Create Set of facets by rotation of an edge around Z-axis *
 *																																		 *
 * Input: k1, k2 - end vertices of the edge														*
 *				r1, r2 - radiuses of the end vertices												*
 *				v1, v2 - visibility of edges produced by rotation of the end *
 *								 vertices																						*
 *				vEdge	- visibility of the edge															*
 *				ifWholeCircle - is true in case of whole circle rotation		 *
 *				nds		- number of discrete steps														*
 *				r[]		- r-coordinates																			 *
 *				kface	- current free cell in the fFacets array									 *
 *																																		 *
 ***********************************************************************/
{
	if (r1 == 0. && r2 == 0) return;

	int i;
	int i1	= k1;
	int i2	= k2;
	int ii1 = ifWholeCircle ? i1 : i1+nds;
	int ii2 = ifWholeCircle ? i2 : i2+nds;
	int vv	= ifWholeCircle ? vEdge : 1;

	if (nds == 1) {
		if (r1 == 0.) {
			fFacets[kface++]	 = UFacet(i1,0,		v2*i2,0, (i2+1),0);
		}else if (r2 == 0.) {
			fFacets[kface++]	 = UFacet(i1,0,		i2,0,		v1*(i1+1),0);
		}else{
			fFacets[kface++]	 = UFacet(i1,0,		v2*i2,0, (i2+1),0, v1*(i1+1),0);
		}
	}else{
		if (r1 == 0.) {
			fFacets[kface++]	 = UFacet(vv*i1,0,		v2*i2,0, vEdge*(i2+1),0);
			for (i2++,i=1; i<nds-1; i2++,i++) {
				fFacets[kface++] = UFacet(vEdge*i1,0, v2*i2,0, vEdge*(i2+1),0);
			}
			fFacets[kface++]	 = UFacet(vEdge*i1,0, v2*i2,0, vv*ii2,0);
		}else if (r2 == 0.) {
			fFacets[kface++]	 = UFacet(vv*i1,0,		vEdge*i2,0, v1*(i1+1),0);
			for (i1++,i=1; i<nds-1; i1++,i++) {
				fFacets[kface++] = UFacet(vEdge*i1,0, vEdge*i2,0, v1*(i1+1),0);
			}
			fFacets[kface++]	 = UFacet(vEdge*i1,0, vv*i2,0,		v1*ii1,0);
		}else{
			fFacets[kface++]	 = UFacet(vv*i1,0,		v2*i2,0, vEdge*(i2+1),0,v1*(i1+1),0);
			for (i1++,i2++,i=1; i<nds-1; i1++,i2++,i++) {
				fFacets[kface++] = UFacet(vEdge*i1,0, v2*i2,0, vEdge*(i2+1),0,v1*(i1+1),0);
			}	
			fFacets[kface++]	 = UFacet(vEdge*i1,0, v2*i2,0, vv*ii2,0,			v1*ii1,0);
		}
	}
}

void UPolyhedron::SetSideFacets(int ii[4], int vv[4], std::vector<int> &kk, std::vector<double> &r,
																 double dphi, int nds, int &kface)
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::SetSideFacets								Date:		20.05.97 *
 * Author: E.Chernyaev (IU/Protvino)							 Revised:					*
 *																																		 *
 * Function: Set side facets for the case of incomplete rotation			 *
 *																																		 *
 * Input: ii[4] - indeces of original verteces												 *
 *				vv[4] - visibility of edges																	*
 *				kk[]	- indeces of nodes																		 *
 *				r[]	 - radiuses																						 *
 *				dphi	- delta phi																						*
 *				nds		- number of discrete steps														*
 *				kface	- current free cell in the fFacets array									 *
 *																																		 *
 ***********************************************************************/
{
	int k1, k2, k3, k4;
	
	if (std::abs((double)(dphi-UUtils::kPi)) < perMillion) {					// half a circle
		for (int i=0; i<4; i++) {
			k1 = ii[i];
			k2 = (i == 3) ? ii[0] : ii[i+1];
			if (r[k1] == 0. && r[k2] == 0.) vv[i] = -1;			
		}
	}

	if (ii[1] == ii[2]) {
		k1 = kk[ii[0]];
		k2 = kk[ii[2]];
		k3 = kk[ii[3]];
		fFacets[kface++] = UFacet(vv[0]*k1,0, vv[2]*k2,0, vv[3]*k3,0);
		if (r[ii[0]] != 0.) k1 += nds;
		if (r[ii[2]] != 0.) k2 += nds;
		if (r[ii[3]] != 0.) k3 += nds;
		fFacets[kface++] = UFacet(vv[2]*k3,0, vv[0]*k2,0, vv[3]*k1,0);
	}else if (kk[ii[0]] == kk[ii[1]]) {
		k1 = kk[ii[0]];
		k2 = kk[ii[2]];
		k3 = kk[ii[3]];
		fFacets[kface++] = UFacet(vv[1]*k1,0, vv[2]*k2,0, vv[3]*k3,0);
		if (r[ii[0]] != 0.) k1 += nds;
		if (r[ii[2]] != 0.) k2 += nds;
		if (r[ii[3]] != 0.) k3 += nds;
		fFacets[kface++] = UFacet(vv[2]*k3,0, vv[1]*k2,0, vv[3]*k1,0);
	}else if (kk[ii[2]] == kk[ii[3]]) {
		k1 = kk[ii[0]];
		k2 = kk[ii[1]];
		k3 = kk[ii[2]];
		fFacets[kface++] = UFacet(vv[0]*k1,0, vv[1]*k2,0, vv[3]*k3,0);
		if (r[ii[0]] != 0.) k1 += nds;
		if (r[ii[1]] != 0.) k2 += nds;
		if (r[ii[2]] != 0.) k3 += nds;
		fFacets[kface++] = UFacet(vv[1]*k3,0, vv[0]*k2,0, vv[3]*k1,0);
	}else{
		k1 = kk[ii[0]];
		k2 = kk[ii[1]];
		k3 = kk[ii[2]];
		k4 = kk[ii[3]];
		fFacets[kface++] = UFacet(vv[0]*k1,0, vv[1]*k2,0, vv[2]*k3,0, vv[3]*k4,0);
		if (r[ii[0]] != 0.) k1 += nds;
		if (r[ii[1]] != 0.) k2 += nds;
		if (r[ii[2]] != 0.) k3 += nds;
		if (r[ii[3]] != 0.) k4 += nds;
		fFacets[kface++] = UFacet(vv[2]*k4,0, vv[1]*k3,0, vv[0]*k2,0, vv[3]*k1,0);
	}
}


void UPolyhedron::RotateAroundZ(int nstep, double phi, double dphi,
								int np1, int np2,
                                const std::vector<double> &z, std::vector<double> &r,
                                int nodeVis, int edgeVis)
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::RotateAroundZ								Date:		27.11.96 *
 * Author: E.Chernyaev (IU/Protvino)							 Revised:					*
 *																																		 *
 * Function: Create UPolyhedron for a solid produced by rotation of	*
 *					 two polylines around Z-axis															 *
 *																																		 *
 * Input: nstep - number of discrete steps, if 0 then default					*
 *				phi	 - starting phi angle																	 *
 *				dphi	- delta phi																						*
 *				np1	 - number of points in external polyline								*
 *								(must be negative in case of closed polyline)				*
 *				np2	 - number of points in internal polyline (may be 1)		 *
 *				z[]	 - z-coordinates (+z >>> -z for both polylines)				 *
 *				r[]	 - r-coordinates																				*
 *				nodeVis - how to Draw edges joing consecutive positions of	 *
 *									node during rotation															 *
 *				edgeVis - how to Draw edges																	*
 *																																		 *
 ***********************************************************************/
{
	static double wholeCircle	 = 2*UUtils::kPi;
		
	//	 S E T	 R O T A T I O N	 P A R A M E T E R S

	bool ifWholeCircle = (std::abs(dphi-wholeCircle) < perMillion) ? true : false;
	double	 delPhi	= ifWholeCircle ? wholeCircle : dphi;	
	int				nSphi		= (nstep > 0) ?
		nstep : int(delPhi*GetNumberOfRotationSteps()/wholeCircle+.5);
	if (nSphi == 0) nSphi = 1;
	int				nVphi		= ifWholeCircle ? nSphi : nSphi+1;
	bool ifClosed = np1 > 0 ? false : true;
	
	//	 C O U N T	 V E R T E C E S

	int absNp1 = std::abs(np1);
	int absNp2 = std::abs(np2);
	int i1beg = 0;
	int i1end = absNp1-1;
	int i2beg = absNp1;
	int i2end = absNp1+absNp2-1; 
	int i, j, k;

	for(i=i1beg; i<=i2end; i++) {
		if (std::abs(r[i]) < perMillion) r[i] = 0.;
	}

	j = 0;																								// external nodes
	for (i=i1beg; i<=i1end; i++) {
		j += (r[i] == 0.) ? 1 : nVphi;
	}

	bool ifSide1 = false;													 // internal nodes
	bool ifSide2 = false;

	if (r[i2beg] != r[i1beg] || z[i2beg] != z[i1beg]) {
		j += (r[i2beg] == 0.) ? 1 : nVphi;
		ifSide1 = true;
	}

	for(i=i2beg+1; i<i2end; i++) {
		j += (r[i] == 0.) ? 1 : nVphi;
	}
	
	if (r[i2end] != r[i1end] || z[i2end] != z[i1end]) {
		if (absNp2 > 1) j += (r[i2end] == 0.) ? 1 : nVphi;
		ifSide2 = true;
	}

	//	 C O U N T	 F A C E S

	k = ifClosed ? absNp1*nSphi : (absNp1-1)*nSphi;			 // external faces

	if (absNp2 > 1) {																		 // internal faces
		for(i=i2beg; i<i2end; i++) {
			if (r[i] > 0. || r[i+1] > 0.)			 k += nSphi;
		}

		if (ifClosed) {
			if (r[i2end] > 0. || r[i2beg] > 0.) k += nSphi;
		}
	}

	if (!ifClosed) {																			// side faces
		if (ifSide1 && (r[i1beg] > 0. || r[i2beg] > 0.)) k += nSphi;
		if (ifSide2 && (r[i1end] > 0. || r[i2end] > 0.)) k += nSphi;
	}

	if (!ifWholeCircle) {																 // phi_side faces
		k += ifClosed ? 2*absNp1 : 2*(absNp1-1);
	}

	//	 A L L O C A T E	 M E M O R Y

	AllocateMemory(j, k);

	//	 G E N E R A T E	 V E R T E C E S

	std::vector<int> kk(absNp1+absNp2);

	k = 1;
	for(i=i1beg; i<=i1end; i++) {
		kk[i] = k;
		if (r[i] == 0.)
		{ fVertices[k++] = UPoint3D(0, 0, z[i]); } else { k += nVphi; }
	}

	i = i2beg;
	if (ifSide1) {
		kk[i] = k;
		if (r[i] == 0.)
		{ fVertices[k++] = UPoint3D(0, 0, z[i]); } else { k += nVphi; }
	}else{
		kk[i] = kk[i1beg];
	}

	for(i=i2beg+1; i<i2end; i++) {
		kk[i] = k;
		if (r[i] == 0.)
		{ fVertices[k++] = UPoint3D(0, 0, z[i]); } else { k += nVphi; }
	}

	if (absNp2 > 1) {
		i = i2end;
		if (ifSide2) {
			kk[i] = k;
			if (r[i] == 0.) fVertices[k] = UPoint3D(0, 0, z[i]);
		}else{
			kk[i] = kk[i1end];
		}
	}

	double cosPhi, sinPhi;

	for(j=0; j<nVphi; j++) {
		cosPhi = std::cos(phi+j*delPhi/nSphi);
		sinPhi = std::sin(phi+j*delPhi/nSphi);
		for(i=i1beg; i<=i2end; i++) {
			if (r[i] != 0.)
				fVertices[kk[i]+j] = UPoint3D(r[i]*cosPhi,r[i]*sinPhi,z[i]);
		}
	}

	//	 G E N E R A T E	 E X T E R N A L	 F A C E S

	int v1,v2;

	k = 1;
	v2 = ifClosed ? nodeVis : 1;
	for(i=i1beg; i<i1end; i++) {
		v1 = v2;
		if (!ifClosed && i == i1end-1) {
			v2 = 1;
		}else{
			v2 = (r[i] == r[i+1] && r[i+1] == r[i+2]) ? -1 : nodeVis;
		}
		RotateEdge(kk[i], kk[i+1], r[i], r[i+1], v1, v2,
							 edgeVis, ifWholeCircle, nSphi, k);
	}
	if (ifClosed) {
		RotateEdge(kk[i1end], kk[i1beg], r[i1end],r[i1beg], nodeVis, nodeVis,
							 edgeVis, ifWholeCircle, nSphi, k);
	}

	//	 G E N E R A T E	 I N T E R N A L	 F A C E S

	if (absNp2 > 1) {
		v2 = ifClosed ? nodeVis : 1;
		for(i=i2beg; i<i2end; i++) {
			v1 = v2;
			if (!ifClosed && i==i2end-1) {
				v2 = 1;
			}else{
				v2 = (r[i] == r[i+1] && r[i+1] == r[i+2]) ? -1 :	nodeVis;
			}
			RotateEdge(kk[i+1], kk[i], r[i+1], r[i], v2, v1,
								 edgeVis, ifWholeCircle, nSphi, k);
		}
		if (ifClosed) {
			RotateEdge(kk[i2beg], kk[i2end], r[i2beg], r[i2end], nodeVis, nodeVis,
								 edgeVis, ifWholeCircle, nSphi, k);
		}
	}

	//	 G E N E R A T E	 S I D E	 F A C E S

	if (!ifClosed) {
		if (ifSide1) {
			RotateEdge(kk[i2beg], kk[i1beg], r[i2beg], r[i1beg], 1, 1,
								 -1, ifWholeCircle, nSphi, k);
		}
		if (ifSide2) {
			RotateEdge(kk[i1end], kk[i2end], r[i1end], r[i2end], 1, 1,
								 -1, ifWholeCircle, nSphi, k);
		}
	}

	//	 G E N E R A T E	 S I D E	 F A C E S	for the case of incomplete circle

	if (!ifWholeCircle) {

		int	ii[4], vv[4];

		if (ifClosed) {
			for (i=i1beg; i<=i1end; i++) {
				ii[0] = i;
				ii[3] = (i == i1end) ? i1beg : i+1;
				ii[1] = (absNp2 == 1) ? i2beg : ii[0]+absNp1;
				ii[2] = (absNp2 == 1) ? i2beg : ii[3]+absNp1;
				vv[0] = -1;
				vv[1] = 1;
				vv[2] = -1;
				vv[3] = 1;
				SetSideFacets(ii, vv, kk, r, dphi, nSphi, k);
			}
		}else{
			for (i=i1beg; i<i1end; i++) {
				ii[0] = i;
				ii[3] = i+1;
				ii[1] = (absNp2 == 1) ? i2beg : ii[0]+absNp1;
				ii[2] = (absNp2 == 1) ? i2beg : ii[3]+absNp1;
				vv[0] = (i == i1beg)	 ? 1 : -1;
				vv[1] = 1;
				vv[2] = (i == i1end-1) ? 1 : -1;
				vv[3] = 1;
				SetSideFacets(ii, vv, kk, r, dphi, nSphi, k);
			}
		}			
	}

	int nface = GetNoFacets();
	if (k-1 != nface) {
		std::cerr
			<< "Polyhedron::RotateAroundZ: number of generated faces ("
			<< k-1 << ") is not equal to the number of allocated faces ("
			<< nface << ")"
			<< std::endl;
	}
}

void UPolyhedron::SetReferences()
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::SetReferences								Date:		04.12.96 *
 * Author: E.Chernyaev (IU/Protvino)							 Revised:					*
 *																																		 *
 * Function: For each edge Set reference to neighbouring facet				 *
 *																																		 *
 ***********************************************************************/
{
	int nface = GetNoFacets();
	if (nface <= 0) return;

	struct edgeListMember {
		edgeListMember *next;
		int v2;
		int iface;
		int iedge;
	};
	
	//	 A L L O C A T E	 A N D	 I N I T I A T E	 L I S T S
	std::vector<edgeListMember> edgeList(2*nface);
	std::vector<edgeListMember *> headList(GetNoVertices());
	
	int i;
	edgeListMember *freeList = &edgeList[0];
	int nFace = GetNoFacets();
	for (i=0; i<2*nface-1; i++) {
		edgeList[i].next = &edgeList[i+1];
	}
	edgeList[2*nface-1].next = 0;

	//	 L O O P	 A L O N G	 E D G E S

	int iface, iedge, nedge, i1, i2, k1, k2;
	edgeListMember *prev, *cur;
	
	for(iface=1; iface<=nface; iface++) {
		nedge = (fFacets[iface].edge[3].v == 0) ? 3 : 4;
		for (iedge=0; iedge<nedge; iedge++) {
			i1 = iedge;
			i2 = (iedge < nedge-1) ? iedge+1 : 0;
			i1 = std::abs(fFacets[iface].edge[i1].v);
			i2 = std::abs(fFacets[iface].edge[i2].v);
			k1 = (i1 < i2) ? i1 : i2;					// k1 = ::min(i1,i2);
			k2 = (i1 > i2) ? i1 : i2;					// k2 = ::max(i1,i2);
			
			// check head of the List corresponding to k1
			cur = headList[k1];
			if (cur == 0) {
				headList[k1] = freeList;
				freeList = freeList->next;
				cur = headList[k1];
				cur->next = 0;
				cur->v2 = k2;
				cur->iface = iface;
				cur->iedge = iedge;
				continue;
			}

			if (cur->v2 == k2) {
				headList[k1] = cur->next;
				cur->next = freeList;
				freeList = cur;			
				fFacets[iface].edge[iedge].f = cur->iface;
				fFacets[cur->iface].edge[cur->iedge].f = iface;
				i1 = (fFacets[iface].edge[iedge].v < 0) ? -1 : 1;
				i2 = (fFacets[cur->iface].edge[cur->iedge].v < 0) ? -1 : 1;
				if (i1 != i2) {
					std::cerr
						<< "Polyhedron::SetReferences: different edge visibility "
						<< iface << "/" << iedge << "/"
						<< fFacets[iface].edge[iedge].v << " and "
						<< cur->iface << "/" << cur->iedge << "/"
						<< fFacets[cur->iface].edge[cur->iedge].v
						<< std::endl;
				}
				continue;
			}

			// check List itself
			for (;;) {
				prev = cur;
				cur = prev->next;
				if (cur == 0) {
					prev->next = freeList;
					freeList = freeList->next;
					cur = prev->next;
					cur->next = 0;
					cur->v2 = k2;
					cur->iface = iface;
					cur->iedge = iedge;
					break;
				}

				if (cur->v2 == k2) {
					prev->next = cur->next;
					cur->next = freeList;
					freeList = cur;			
					fFacets[iface].edge[iedge].f = cur->iface;
					fFacets[cur->iface].edge[cur->iedge].f = iface;
					i1 = (fFacets[iface].edge[iedge].v < 0) ? -1 : 1;
					i2 = (fFacets[cur->iface].edge[cur->iedge].v < 0) ? -1 : 1;
						if (i1 != i2) {
							std::cerr
								<< "Polyhedron::SetReferences: different edge visibility "
								<< iface << "/" << iedge << "/"
								<< fFacets[iface].edge[iedge].v << " and "
								<< cur->iface << "/" << cur->iedge << "/"
								<< fFacets[cur->iface].edge[cur->iedge].v
								<< std::endl;
						}
					break;
				}
			}
		}
	}

	//	C H E C K	 T H A T	 A L L	 L I S T S	 A R E	 E M P T Y

	int n = GetNoVertices();
	for (i=0; i<n; i++) {
		if (headList[i] != 0) {
			std::cerr
				<< "Polyhedron::SetReferences: List " << i << " is not empty"
				<< std::endl;
		}
	}

	//	 F R E E	 M E M O R Y
}

void UPolyhedron::InvertFacets()
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::InvertFacets								Date:		01.12.99	*
 * Author: E.Chernyaev															Revised:					 *
 *																																		 *
 * Function: Invert the order of the nodes in the facets							 *
 *																																		 *
 ***********************************************************************/
{
	int nface = GetNoFacets();
	if (nface <= 0) return;
	int i, k, nnode, v[4],f[4];
	for (i=1; i<=nface; i++) {
		nnode =	(fFacets[i].edge[3].v == 0) ? 3 : 4;
		for (k=0; k<nnode; k++) {
			v[k] = (k+1 == nnode) ? fFacets[i].edge[0].v : fFacets[i].edge[k+1].v;
			if (v[k] * fFacets[i].edge[k].v < 0) v[k] = -v[k];
			f[k] = fFacets[i].edge[k].f;
		}
		for (k=0; k<nnode; k++) {
			fFacets[i].edge[nnode-1-k].v = v[k];
			fFacets[i].edge[nnode-1-k].f = f[k];
		}
	}
}

// UPolyhedron &UPolyhedron::Transform(const UTransform3D &t)
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::Transform										Date:		01.12.99	*
 * Author: E.Chernyaev															Revised:					 *
 *																																		 *
 * Function: Make transformation of the polyhedron										 *
 *																																		 *
 ***********************************************************************/

/*
{
    int size = GetNoVertices();
	if (size > 0) {
		for (int i=1; i<=size; i++) { fVertices[i] = t * fVertices[i]; }

		//	C H E C K	 D E T E R M I N A N T	 A N D
		//	I N V E R T	 F A C E T S	 I F	 I T	 I S	 N E G A T I V E

		UVector3D d = t * UVector3D(0,0,0);
		UVector3D x = t * UVector3D(1,0,0) - d;
		UVector3D y = t * UVector3D(0,1,0) - d;
		UVector3D z = t * UVector3D(0,0,1) - d;
		if ((x.Cross(y))*z < 0) InvertFacets();
	}
	return *this;
}
*/

bool UPolyhedron::GetNextVertexIndex(int &index, int &edgeFlag) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetNextVertexIndex					Date:		03.09.96	*
 * Author: Yasuhide Sawada													Revised:					 *
 *																																		 *
 * Function:																													 *
 *																																		 *
 ***********************************************************************/
{
	static int iFace = 1;
	static int iQVertex = 0;
	int vIndex = fFacets[iFace].edge[iQVertex].v;

	edgeFlag = (vIndex > 0) ? 1 : 0;
	index = std::abs(vIndex);

	if (iQVertex >= 3 || fFacets[iFace].edge[iQVertex+1].v == 0) {
		iQVertex = 0;
		int nface = GetNoFacets();
		if (++iFace > nface) iFace = 1;
		return false;	// Last Edge
	}else{
		++iQVertex;
		return true;	// not Last Edge
	}
}

UPoint3D UPolyhedron::GetVertex(int index) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetVertex									 Date:		03.09.96	*
 * Author: Yasuhide Sawada													Revised: 17.11.99	*
 *																																		 *
 * Function: Get vertex of the index.																	*
 *																																		 *
 ***********************************************************************/
{
	if (index <= 0 || index > GetNoVertices()) {
		std::cerr
			<< "UPolyhedron::GetVertex: irrelevant index " << index
			<< std::endl;
		return UPoint3D();
	}
	return fVertices[index];
}

bool
UPolyhedron::GetNextVertex(UPoint3D &vertex, int &edgeFlag) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetNextVertex							 Date:		22.07.96	*
 * Author: John Allison														 Revised:					 *
 *																																		 *
 * Function: Get vertices of the quadrilaterals in order for each			*
 *					 face in face order.	Returns false when finished each		 *
 *					 face.																										 *
 *																																		 *
 ***********************************************************************/
{
	int index;
	bool rep = GetNextVertexIndex(index, edgeFlag);
	vertex = fVertices[index];
	return rep;
}

bool UPolyhedron::GetNextVertex(UPoint3D &vertex, int &edgeFlag,
																	UNormal3D &normal) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetNextVertex							 Date:		26.11.99	*
 * Author: E.Chernyaev															Revised:					 *
 *																																		 *
 * Function: Get vertices with normals of the quadrilaterals in order	*
 *					 for each face in face order.															*
 *					 Returns false when finished each face.										*
 *																																		 *
 ***********************************************************************/
{
	static int iFace = 1;
	static int iNode = 0;

	int nface = GetNoFacets();
	if (nface == 0) return false;	// empty polyhedron

	int k = fFacets[iFace].edge[iNode].v;
	if (k > 0) { edgeFlag = 1; } else { edgeFlag = -1; k = -k; }
	vertex = fVertices[k];
	normal = FindNodeNormal(iFace,k);
	if (iNode >= 3 || fFacets[iFace].edge[iNode+1].v == 0) {
		iNode = 0;
		if (++iFace > nface) iFace = 1;
		return false;								// last node
	}else{
		++iNode;
		return true;								 // not last node
	}
}

bool UPolyhedron::GetNextEdgeIndeces(int &i1, int &i2, int &edgeFlag,
																			 int &iface1, int &iface2) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetNextEdgeIndeces					Date:		30.09.96	*
 * Author: E.Chernyaev															Revised: 17.11.99	*
 *																																		 *
 * Function: Get indeces of the next edge together with indeces of		 *
 *					 of the faces which share the edge.												*
 *					 Returns false when the last edge.												 *
 *																																		 *
 ***********************************************************************/
{
	static int iFace		= 1;
	static int iQVertex = 0;
	static int iOrder	 = 1;
	int	k1, k2, kflag, kface1, kface2;

	int nface = GetNoFacets();
	if (iFace == 1 && iQVertex == 0) {
		k2 = fFacets[nface].edge[0].v;
		k1 = fFacets[nface].edge[3].v;
		if (k1 == 0) k1 = fFacets[nface].edge[2].v;
		if (std::abs(k1) > std::abs(k2)) iOrder = -1;
	}

	do {
		k1		 = fFacets[iFace].edge[iQVertex].v;
		kflag	= k1;
		k1		 = std::abs(k1);
		kface1 = iFace; 
		kface2 = fFacets[iFace].edge[iQVertex].f;
		if (iQVertex >= 3 || fFacets[iFace].edge[iQVertex+1].v == 0) {
			iQVertex = 0;
			k2 = std::abs(fFacets[iFace].edge[iQVertex].v);
			iFace++;
		}else{
			iQVertex++;
			k2 = std::abs(fFacets[iFace].edge[iQVertex].v);
		}
	} while (iOrder*k1 > iOrder*k2);

	i1 = k1; i2 = k2; edgeFlag = (kflag > 0) ? 1 : 0;
	iface1 = kface1; iface2 = kface2; 

	if (iFace > nface) {
		iFace	= 1; iOrder = 1;
		return false;
	}else{
		return true;
	}
}

bool
UPolyhedron::GetNextEdgeIndeces(int &i1, int &i2, int &edgeFlag) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetNextEdgeIndeces					Date:		17.11.99	*
 * Author: E.Chernyaev															Revised:					 *
 *																																		 *
 * Function: Get indeces of the next edge.														 *
 *					 Returns false when the last edge.												 *
 *																																		 *
 ***********************************************************************/
{
	int kface1, kface2;
	return GetNextEdgeIndeces(i1, i2, edgeFlag, kface1, kface2);
}

bool UPolyhedron::GetNextEdge(UPoint3D &p1,
													 UPoint3D &p2,
													 int &edgeFlag) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetNextEdge								 Date:		30.09.96	*
 * Author: E.Chernyaev															Revised:					 *
 *																																		 *
 * Function: Get next edge.																						*
 *					 Returns false when the last edge.												 *
 *																																		 *
 ***********************************************************************/
{
	int i1,i2;
	bool rep = GetNextEdgeIndeces(i1,i2,edgeFlag);
	p1 = fVertices[i1];
	p2 = fVertices[i2];
	return rep;
}

bool
UPolyhedron::GetNextEdge(UPoint3D &p1, UPoint3D &p2,
													int &edgeFlag, int &iface1, int &iface2) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetNextEdge								 Date:		17.11.99	*
 * Author: E.Chernyaev															Revised:					 *
 *																																		 *
 * Function: Get next edge with indeces of the faces which share			 *
 *					 the edge.																								 *
 *					 Returns false when the last edge.												 *
 *																																		 *
 ***********************************************************************/
{
	int i1,i2;
	bool rep = GetNextEdgeIndeces(i1,i2,edgeFlag,iface1,iface2);
	p1 = fVertices[i1];
	p2 = fVertices[i2];
	return rep;
}

void UPolyhedron::GetFacet(int iFace, std::vector<int> &facet)
{
	facet.resize(4);
	int n;
	GetFacet(iFace, n, &facet[0]);
	facet.resize(n);
}

void UPolyhedron::GetFacet(int iFace, int &n, int *iNodes,
														int *edgeFlags, int *iFaces) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetFacet										Date:		15.12.99	*
 * Author: E.Chernyaev															Revised:					 *
 *																																		 *
 * Function: Get face by index																				 *
 *																																		 *
 ***********************************************************************/
{
	int nface = GetNoFacets();
	if (iFace < 1 || iFace > nface) {
		std::cerr 
			<< "UPolyhedron::GetFacet: irrelevant index " << iFace
			<< std::endl;
		n = 0;
	}else{
		int i, k;
		for (i=0; i<4; i++) { 
			const UFacet &facet = fFacets[iFace];
			const UEdge &edge = facet.edge[i];
			if (edge.v == 0) break;
			if (iFaces != 0) iFaces[i] = edge.f;
			if (edge.v > 0) { 
				iNodes[i] = edge.v;
				if (edgeFlags != 0) edgeFlags[i] = 1;
			}else{
				iNodes[i] = -edge.v;
				if (edgeFlags != 0) edgeFlags[i] = -1;
			}
		}
		n = i;
	}
}

void UPolyhedron::GetFacet(int index, int &n, UPoint3D *nodes,
														 int *edgeFlags, UNormal3D *normals) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetFacet										Date:		17.11.99	*
 * Author: E.Chernyaev															Revised:					 *
 *																																		 *
 * Function: Get face by index																				 *
 *																																		 *
 ***********************************************************************/
{
	int iNodes[4];
	GetFacet(index, n, iNodes, edgeFlags);
	if (n != 0) {
		for (int i=0; i<n; i++) {
			nodes[i] = fVertices[iNodes[i]];
			if (normals != 0) normals[i] = FindNodeNormal(index,iNodes[i]);
		}
	}
}

bool
UPolyhedron::GetNextFacet(int &n, UPoint3D *nodes,
													 int *edgeFlags, UNormal3D *normals) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetNextFacet								Date:		19.11.99	*
 * Author: E.Chernyaev															Revised:					 *
 *																																		 *
 * Function: Get next face with normals of Unit length at the nodes.	 *
 *					 Returns false when finished all faces.										*
 *																																		 *
 ***********************************************************************/
{
	static int iFace = 1;

	if (edgeFlags == 0) {
		GetFacet(iFace, n, nodes);
	}else if (normals == 0) {
		GetFacet(iFace, n, nodes, edgeFlags);
	}else{
		GetFacet(iFace, n, nodes, edgeFlags, normals);
	}

	int nface = GetNoFacets();
	if (++iFace > nface) {
		iFace	= 1;
		return false;
	}else{
		return true;
	}
}

UNormal3D UPolyhedron::GetNormal(int iFace) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetNormal										Date:		19.11.99 *
 * Author: E.Chernyaev															 Revised:					*
 *																																		 *
 * Function: Get normal of the face given by index										 *
 *																																		 *
 ***********************************************************************/
{
	int nface = GetNoFacets();
	if (iFace < 1 || iFace > nface) {
		std::cerr 
			<< "UPolyhedron::GetNormal: irrelevant index " << iFace 
			<< std::endl;
		return UNormal3D();
	}

	int i0	= std::abs(fFacets[iFace].edge[0].v);
	int i1	= std::abs(fFacets[iFace].edge[1].v);
	int i2	= std::abs(fFacets[iFace].edge[2].v);
	int i3	= std::abs(fFacets[iFace].edge[3].v);
	if (i3 == 0) i3 = i0;
	return (fVertices[i2] - fVertices[i0]).Cross(fVertices[i3] - fVertices[i1]);
}

UNormal3D UPolyhedron::GetUnitNormal(int iFace) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetNormal										Date:		19.11.99 *
 * Author: E.Chernyaev															 Revised:					*
 *																																		 *
 * Function: Get Unit normal of the face given by index								*
 *																																		 *
 ***********************************************************************/
{
	int nface = GetNoFacets();
	if (iFace < 1 || iFace > nface) {
		std::cerr 
			<< "UPolyhedron::GetUnitNormal: irrelevant index " << iFace
			<< std::endl;
		return UNormal3D();
	}

	int i0	= std::abs(fFacets[iFace].edge[0].v);
	int i1	= std::abs(fFacets[iFace].edge[1].v);
	int i2	= std::abs(fFacets[iFace].edge[2].v);
	int i3	= std::abs(fFacets[iFace].edge[3].v);
	if (i3 == 0) i3 = i0;
	return ((fVertices[i2] - fVertices[i0]).Cross(fVertices[i3] - fVertices[i1])).Unit();
}

bool UPolyhedron::GetNextNormal(UNormal3D &normal) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetNextNormal							 Date:		22.07.96	*
 * Author: John Allison														 Revised: 19.11.99	*
 *																																		 *
 * Function: Get normals of each face in face order.	Returns false		*
 *					 when finished all faces.																	*
 *																																		 *
 ***********************************************************************/
{
	static int iFace = 1;
	normal = GetNormal(iFace);
	int nface = GetNoFacets();
	if (++iFace > nface) {
		iFace = 1;
		return false;
	}else{
		return true;
	}
}

bool UPolyhedron::GetNextUnitNormal(UNormal3D &normal) const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetNextUnitNormal					 Date:		16.09.96	*
 * Author: E.Chernyaev															Revised:					 *
 *																																		 *
 * Function: Get normals of Unit length of each face in face order.		*
 *					 Returns false when finished all faces.										*
 *																																		 *
 ***********************************************************************/
{
	bool rep = GetNextNormal(normal);
	normal = normal.Unit();
	return rep;
}

double UPolyhedron::GetSurfaceArea() const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetSurfaceArea							Date:		25.05.01	*
 * Author: E.Chernyaev															Revised:					 *
 *																																		 *
 * Function: Returns area of the surface of the polyhedron.						*
 *																																		 *
 ***********************************************************************/
{
	double srf = 0.;
	int nface = GetNoFacets();
	for (int iFace=1; iFace<=nface; iFace++) {
		int i0 = std::abs(fFacets[iFace].edge[0].v);
		int i1 = std::abs(fFacets[iFace].edge[1].v);
		int i2 = std::abs(fFacets[iFace].edge[2].v);
		int i3 = std::abs(fFacets[iFace].edge[3].v);
		if (i3 == 0) i3 = i0;
		srf += ((fVertices[i2] - fVertices[i0]).Cross(fVertices[i3] - fVertices[i1])).Mag();
	}
	return srf/2.;
}

double UPolyhedron::GetVolume() const
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::GetVolume									 Date:		25.05.01	*
 * Author: E.Chernyaev															Revised:					 *
 *																																		 *
 * Function: Returns volume of the polyhedron.												 *
 *																																		 *
 ***********************************************************************/
{
	double v = 0.;
	int nface = GetNoFacets();
	for (int iFace=1; iFace<=nface; iFace++) {
		int i0 = std::abs(fFacets[iFace].edge[0].v);
		int i1 = std::abs(fFacets[iFace].edge[1].v);
		int i2 = std::abs(fFacets[iFace].edge[2].v);
		int i3 = std::abs(fFacets[iFace].edge[3].v);
		UPoint3D pt;
		if (i3 == 0) {
			i3 = i0;
			pt = (fVertices[i0]+fVertices[i1]+fVertices[i2]) * (1./3.);
		}else{
			pt = (fVertices[i0]+fVertices[i1]+fVertices[i2]+fVertices[i3]) * 0.25;
		}
		v += ((fVertices[i2] - fVertices[i0]).Cross(fVertices[i3] - fVertices[i1])).Dot(pt);
	}
	return v/6.;
}

int
UPolyhedron::createTwistedTrap(double Dz,
																 const double xy1[][2],
																 const double xy2[][2])
/***********************************************************************
 *																																		 *
 * Name: createTwistedTrap													 Date:		05.11.02 *
 * Author: E.Chernyaev (IU/Protvino)							 Revised:					*
 *																																		 *
 * Function: Creates polyhedron for twisted trapezoid									*
 *																																		 *
 * Input: Dz			 - half-length along Z						 8----7						*
 *				xy1[2,4] - quadrilateral at Z=-Dz			 5----6	!						*
 *				xy2[2,4] - quadrilateral at Z=+Dz			 !	4-!--3						*
 *																								1----2							 *
 *																																		 *
 ***********************************************************************/
{
	AllocateMemory(12,18);

	fVertices[ 1] = UPoint3D(xy1[0][0],xy1[0][1],-Dz);
	fVertices[ 2] = UPoint3D(xy1[1][0],xy1[1][1],-Dz);
	fVertices[ 3] = UPoint3D(xy1[2][0],xy1[2][1],-Dz);
	fVertices[ 4] = UPoint3D(xy1[3][0],xy1[3][1],-Dz);

	fVertices[ 5] = UPoint3D(xy2[0][0],xy2[0][1], Dz);
	fVertices[ 6] = UPoint3D(xy2[1][0],xy2[1][1], Dz);
	fVertices[ 7] = UPoint3D(xy2[2][0],xy2[2][1], Dz);
	fVertices[ 8] = UPoint3D(xy2[3][0],xy2[3][1], Dz);

	fVertices[ 9] = (fVertices[1]+fVertices[2]+fVertices[5]+fVertices[6])/4.;
	fVertices[10] = (fVertices[2]+fVertices[3]+fVertices[6]+fVertices[7])/4.;
	fVertices[11] = (fVertices[3]+fVertices[4]+fVertices[7]+fVertices[8])/4.;
	fVertices[12] = (fVertices[4]+fVertices[1]+fVertices[8]+fVertices[5])/4.;

	enum {DUMMY, BOTTOM,
				LEFT_BOTTOM,	LEFT_FRONT,	 LEFT_TOP,	LEFT_BACK,
				BACK_BOTTOM,	BACK_LEFT,		BACK_TOP,	BACK_RIGHT,
				RIGHT_BOTTOM, RIGHT_BACK,	 RIGHT_TOP, RIGHT_FRONT,
				FRONT_BOTTOM, FRONT_RIGHT,	FRONT_TOP, FRONT_LEFT,
				TOP};

	fFacets[ 1]=UFacet(1,LEFT_BOTTOM, 4,BACK_BOTTOM, 3,RIGHT_BOTTOM, 2,FRONT_BOTTOM);

	fFacets[ 2]=UFacet(4,BOTTOM,		 -1,LEFT_FRONT,	-12,LEFT_BACK,		0,0);
	fFacets[ 3]=UFacet(1,FRONT_LEFT, -5,LEFT_TOP,		-12,LEFT_BOTTOM,	0,0);
	fFacets[ 4]=UFacet(5,TOP,				-8,LEFT_BACK,	 -12,LEFT_FRONT,	 0,0);
	fFacets[ 5]=UFacet(8,BACK_LEFT,	-4,LEFT_BOTTOM, -12,LEFT_TOP,		 0,0);

	fFacets[ 6]=UFacet(3,BOTTOM,		 -4,BACK_LEFT,	 -11,BACK_RIGHT,	 0,0);
	fFacets[ 7]=UFacet(4,LEFT_BACK,	-8,BACK_TOP,		-11,BACK_BOTTOM,	0,0);
	fFacets[ 8]=UFacet(8,TOP,				-7,BACK_RIGHT,	-11,BACK_LEFT,		0,0);
	fFacets[ 9]=UFacet(7,RIGHT_BACK, -3,BACK_BOTTOM, -11,BACK_TOP,		 0,0);

	fFacets[10]=UFacet(2,BOTTOM,		 -3,RIGHT_BACK,	-10,RIGHT_FRONT,	0,0);
	fFacets[11]=UFacet(3,BACK_RIGHT, -7,RIGHT_TOP,	 -10,RIGHT_BOTTOM, 0,0);
	fFacets[12]=UFacet(7,TOP,				-6,RIGHT_FRONT, -10,RIGHT_BACK,	 0,0);
	fFacets[13]=UFacet(6,FRONT_RIGHT,-2,RIGHT_BOTTOM,-10,RIGHT_TOP,		0,0);

	fFacets[14]=UFacet(1,BOTTOM,		 -2,FRONT_RIGHT,	-9,FRONT_LEFT,	 0,0);
	fFacets[15]=UFacet(2,RIGHT_FRONT,-6,FRONT_TOP,		-9,FRONT_BOTTOM, 0,0);
	fFacets[16]=UFacet(6,TOP,				-5,FRONT_LEFT,	 -9,FRONT_RIGHT,	0,0);
	fFacets[17]=UFacet(5,LEFT_FRONT, -1,FRONT_BOTTOM, -9,FRONT_TOP,		0,0);
 
	fFacets[18]=UFacet(5,FRONT_TOP, 6,RIGHT_TOP, 7,BACK_TOP, 8,LEFT_TOP);

	return 0;
}

int
UPolyhedron::createPolyhedron(int Nnodes, int Nfaces,
   const double xyz[][3],
   const int	faces[][4])

   /***********************************************************************
 *																																		 *
 * Name: createPolyhedron														Date:		05.11.02 *
 * Author: E.Chernyaev (IU/Protvino)							 Revised:					*
 *																																		 *
 * Function: Creates user defined polyhedron													 *
 *																																		 *
 * Input: Nnodes	- number of nodes																		*
 *				Nfaces	- number of faces																		*
 *				nodes[][3] - node coordinates																*
 *				faces[][4] - faces																					 *
 *																																		 *
 ***********************************************************************/
{
	AllocateMemory(Nnodes, Nfaces);
	if (GetNoVertices() <= 0) return 1;

	for (int i=0; i<Nnodes; i++) {
		fVertices[i+1] = UPoint3D(xyz[i][0], xyz[i][1], xyz[i][2]);
	}
	for (int k=0; k<Nfaces; k++) {
		fFacets[k+1] = UFacet(faces[k][0],0,faces[k][1],0,faces[k][2],0,faces[k][3],0);
	}
	SetReferences();
	return 0;
}

UPolyhedronTrd2::UPolyhedronTrd2(double Dx1, double Dx2,
		double Dy1, double Dy2, double Dz)
/***********************************************************************
 *																																		 *
 * Name: UPolyhedronTrd2													 Date:		22.07.96 *
 * Author: E.Chernyaev (IU/Protvino)							 Revised:					*
 *																																		 *
 * Function: Create GEANT4 TRD2-trapezoid															*
 *																																		 *
 * Input: Dx1 - half-length along X at -Dz					 8----7						*
 *				Dx2 - half-length along X ay +Dz				5----6	!						*
 *				Dy1 - half-length along Y ay -Dz				!	4-!--3						*
 *				Dy2 - half-length along Y ay +Dz				1----2							 *
 *				Dz	- half-length along Z																		*
 *																																		 *
 ***********************************************************************/
{
	AllocateMemory(8,6);

	fVertices[1] = UPoint3D(-Dx1,-Dy1,-Dz);
	fVertices[2] = UPoint3D( Dx1,-Dy1,-Dz);
	fVertices[3] = UPoint3D( Dx1, Dy1,-Dz);
	fVertices[4] = UPoint3D(-Dx1, Dy1,-Dz);
	fVertices[5] = UPoint3D(-Dx2,-Dy2, Dz);
	fVertices[6] = UPoint3D( Dx2,-Dy2, Dz);
	fVertices[7] = UPoint3D( Dx2, Dy2, Dz);
	fVertices[8] = UPoint3D(-Dx2, Dy2, Dz);

	CreatePrism();
	SetReferences();
}

UPolyhedronTrd2::~UPolyhedronTrd2() {}

UPolyhedronTrd1::UPolyhedronTrd1(double Dx1, double Dx2,
								double Dy, double Dz)
	: UPolyhedronTrd2(Dx1, Dx2, Dy, Dy, Dz) {}

UPolyhedronTrd1::~UPolyhedronTrd1() {}

UPolyhedronBox::UPolyhedronBox(double Dx, double Dy, double Dz)
	: UPolyhedronTrd2(Dx, Dx, Dy, Dy, Dz) {}

UPolyhedronBox::~UPolyhedronBox() {}

UPolyhedronTrap::UPolyhedronTrap(double Dz,
								double Theta,
								double Phi,
								double Dy1,
								double Dx1,
								double Dx2,
								double Alp1,
								double Dy2,
								double Dx3,
								double Dx4,
								double Alp2)
/***********************************************************************
 *																																		 *
 * Name: UPolyhedronTrap													 Date:		20.11.96 *
 * Author: E.Chernyaev															 Revised:					*
 *																																		 *
 * Function: Create GEANT4 TRAP-trapezoid															*
 *																																		 *
 * Input: DZ	 - half-length in Z																			*
 *				Theta,Phi - polar angles of the line joining centres of the	*
 *										faces at Z=-Dz and Z=+Dz												 *
 *				Dy1	- half-length in Y of the face at Z=-Dz								 *
 *				Dx1	- half-length in X of low edge of the face at Z=-Dz		 *
 *				Dx2	- half-length in X of top edge of the face at Z=-Dz		 *
 *				Alp1 - angle between Y-axis and the median joining top and	 *
 *							 low edges of the face at Z=-Dz												*
 *				Dy2	- half-length in Y of the face at Z=+Dz								 *
 *				Dx3	- half-length in X of low edge of the face at Z=+Dz		 *
 *				Dx4	- half-length in X of top edge of the face at Z=+Dz		 *
 *				Alp2 - angle between Y-axis and the median joining top and	 *
 *							 low edges of the face at Z=+Dz												*
 *																																		 *
 ***********************************************************************/
{
	double DzTthetaCphi = Dz*std::tan(Theta)*std::cos(Phi);
	double DzTthetaSphi = Dz*std::tan(Theta)*std::sin(Phi);
	double Dy1Talp1 = Dy1*std::tan(Alp1);
	double Dy2Talp2 = Dy2*std::tan(Alp2);
	
	AllocateMemory(8,6);

	fVertices[1] = UPoint3D(-DzTthetaCphi-Dy1Talp1-Dx1,-DzTthetaSphi-Dy1,-Dz);
	fVertices[2] = UPoint3D(-DzTthetaCphi-Dy1Talp1+Dx1,-DzTthetaSphi-Dy1,-Dz);
	fVertices[3] = UPoint3D(-DzTthetaCphi+Dy1Talp1+Dx2,-DzTthetaSphi+Dy1,-Dz);
	fVertices[4] = UPoint3D(-DzTthetaCphi+Dy1Talp1-Dx2,-DzTthetaSphi+Dy1,-Dz);
	fVertices[5] = UPoint3D( DzTthetaCphi-Dy2Talp2-Dx3, DzTthetaSphi-Dy2, Dz);
	fVertices[6] = UPoint3D( DzTthetaCphi-Dy2Talp2+Dx3, DzTthetaSphi-Dy2, Dz);
	fVertices[7] = UPoint3D( DzTthetaCphi+Dy2Talp2+Dx4, DzTthetaSphi+Dy2, Dz);
	fVertices[8] = UPoint3D( DzTthetaCphi+Dy2Talp2-Dx4, DzTthetaSphi+Dy2, Dz);

	CreatePrism();
}

UPolyhedronTrap::~UPolyhedronTrap() {}

UPolyhedronPara::UPolyhedronPara(double Dx, double Dy, double Dz,
																		 double Alpha, double Theta,
																		 double Phi)
	: UPolyhedronTrap(Dz, Theta, Phi, Dy, Dx, Dx, Alpha, Dy, Dx, Dx, Alpha) {}

UPolyhedronPara::~UPolyhedronPara() {}

UPolyhedronParaboloid::UPolyhedronParaboloid(double r1,
	double r2,
	double dz,
	double sPhi,
	double dPhi) 
/***********************************************************************
 *																																		 *
 * Name: UPolyhedronParaboloid										 Date:		28.06.07 *
 * Author: L.Lindroos, T.Nikitina (CERN), July 2007	Revised: 28.06.07 *
 *																																		 *
 * Function: Constructor for paraboloid																*
 *																																		 *
 * Input: r1		- inside and outside radiuses at -Dz									 *
 *				r2		- inside and outside radiuses at +Dz									 *
 *				dz		- half length in Z																		 *
 *				sPhi	- starting angle of the segment												*
 *				dPhi	- segment range																				*
 *																																		 *
 ***********************************************************************/
{
	static double wholeCircle=2*UUtils::kPi;

	//	 C H E C K	 I N P U T	 P A R A M E T E R S

	int k = 0;
	if (r1 < 0. || r2 <= 0.)				k = 1;

	if (dz <= 0.) k += 2;

	double phi1, phi2, dphi;

	if(dPhi < 0.)
	{
		phi2 = sPhi; phi1 = phi2 + dPhi;
	}
	else if(dPhi == 0.) 
	{
		phi1 = sPhi; phi2 = phi1 + wholeCircle;
	}
	else
	{
		phi1 = sPhi; phi2 = phi1 + dPhi;
	}
	dphi = phi2 - phi1;

	if (std::abs(dphi-wholeCircle) < perMillion) dphi = wholeCircle;
	if (dphi > wholeCircle) k += 4; 

	if (k != 0) {
		std::cerr << "UPolyhedronParaboloid: error in input parameters";
		if ((k & 1) != 0) std::cerr << " (radiuses)";
		if ((k & 2) != 0) std::cerr << " (half-length)";
		if ((k & 4) != 0) std::cerr << " (angles)";
		std::cerr << std::endl;
		std::cerr << " r1=" << r1;
		std::cerr << " r2=" << r2;
		std::cerr << " dz=" << dz << " sPhi=" << sPhi << " dPhi=" << dPhi
							<< std::endl;
		return;
	}
	
	//	 P R E P A R E	 T W O	 P O L Y L I N E S

	int n = GetNumberOfRotationSteps();
	double dl = (r2 - r1) / n;
	double k1 = (r2*r2 - r1*r1) / 2 / dz;
	double k2 = (r2*r2 + r1*r1) / 2;

	std::vector<double> zz(n + 2), rr(n + 2);

	zz[0] = dz;
	rr[0] = r2;

	for(int i = 1; i < n - 1; i++)
	{
		rr[i] = rr[i-1] - dl;
		zz[i] = (rr[i]*rr[i] - k2) / k1;
		if(rr[i] < 0)
		{
			rr[i] = 0;
			zz[i] = 0;
		}
	}

	zz[n-1] = -dz;
	rr[n-1] = r1;

	zz[n] = dz;
	rr[n] = 0;

	zz[n+1] = -dz;
	rr[n+1] = 0;

	//	 R O T A T E		P O L Y L I N E S

	RotateAroundZ(0, phi1, dphi, n, 2, zz, rr, -1, -1); 
	SetReferences();
}

UPolyhedronParaboloid::~UPolyhedronParaboloid() {}

UPolyhedronHype::UPolyhedronHype(double r1,
																		 double r2,
																		 double sqrtan1,
																		 double sqrtan2,
																		 double halfZ) 
/***********************************************************************
 *																																		 *
 * Name: UPolyhedronHype													 Date:		14.04.08 *
 * Author: Tatiana Nikitina (CERN)									 Revised: 14.04.08 *
 *																																		 *
 * Function: Constructor for Hype																			*
 *																																		 *
 * Input: r1			 - inside radius at z=0															*
 *				r2			 - outside radiuses at z=0													 *
 *				sqrtan1	- sqr of tan of Inner Stereo Angle									*
 *				sqrtan2	- sqr of tan of Outer Stereo Angle									*
 *				halfZ		- half length in Z																	*
 *																																		 *
 ***********************************************************************/
{
	static double wholeCircle = 2*UUtils::kPi;

	//	 C H E C K	 I N P U T	 P A R A M E T E R S

	int k = 0;
	if (r2 < 0. || r1 < 0. )				k = 1;
	if (r1 > r2 )									 k = 1;
	if (r1 == r2)									 k = 1;

	if (halfZ <= 0.) k += 2;
 
	if (sqrtan1<0.||sqrtan2<0.) k += 4;	
 
	if (k != 0)
	{
		std::cerr << "UPolyhedronHype: error in input parameters";
		if ((k & 1) != 0) std::cerr << " (radiuses)";
		if ((k & 2) != 0) std::cerr << " (half-length)";
		if ((k & 4) != 0) std::cerr << " (angles)";
		std::cerr << std::endl;
		std::cerr << " r1=" << r1 << " r2=" << r2;
		std::cerr << " halfZ=" << halfZ << " sqrTan1=" << sqrtan1
							<< " sqrTan2=" << sqrtan2
							<< std::endl;
		return;
	}
	
	//	 P R E P A R E	 T W O	 P O L Y L I N E S

	int n = GetNumberOfRotationSteps();
	double dz = 2.*halfZ / n;
	double k1 = r1*r1;
	double k2 = r2*r2;
    
	std::vector<double> zz(n+n+1), rr(n+n+1);

	zz[0] = halfZ;
	rr[0] = std::sqrt(sqrtan2*halfZ*halfZ+k2);

	for(int i = 1; i < n-1; i++)
	{
		zz[i] = zz[i-1] - dz;
		rr[i] =std::sqrt(sqrtan2*zz[i]*zz[i]+k2);
	}

	zz[n-1] = -halfZ;
	rr[n-1] = rr[0];

	zz[n] = halfZ;
	rr[n] =	std::sqrt(sqrtan1*halfZ*halfZ+k1);

	for(int i = n+1; i < n+n; i++)
	{
		zz[i] = zz[i-1] - dz;
		rr[i] =std::sqrt(sqrtan1*zz[i]*zz[i]+k1);
	}
	zz[n+n] = -halfZ;
	rr[n+n] = rr[n];

	//	 R O T A T E		P O L Y L I N E S

	RotateAroundZ(0, 0., wholeCircle, n, n, zz, rr, -1, -1); 
	SetReferences();
}

UPolyhedronHype::~UPolyhedronHype() {}

UPolyhedronCons::UPolyhedronCons(double Rmn1,
																		 double Rmx1,
																		 double Rmn2,
																		 double Rmx2, 
																		 double Dz,
																		 double Phi1,
																		 double Dphi) 
/***********************************************************************
 *																																		 *
 * Name: UPolyhedronCons::UPolyhedronCons				Date:		15.12.96 *
 * Author: E.Chernyaev (IU/Protvino)							 Revised: 15.12.96 *
 *																																		 *
 * Function: Constructor for CONS, TUBS, CONE, TUBE										*
 *																																		 *
 * Input: Rmn1, Rmx1 - inside and outside radiuses at -Dz							*
 *				Rmn2, Rmx2 - inside and outside radiuses at +Dz							*
 *				Dz				 - half length in Z																*
 *				Phi1			 - starting angle of the segment									 *
 *				Dphi			 - segment range																	 *
 *																																		 *
 ***********************************************************************/
{
	static double wholeCircle=2*UUtils::kPi;

	//	 C H E C K	 I N P U T	 P A R A M E T E R S

	int k = 0;
	if (Rmn1 < 0. || Rmx1 < 0. || Rmn2 < 0. || Rmx2 < 0.)				k = 1;
	if (Rmn1 > Rmx1 || Rmn2 > Rmx2)															k = 1;
	if (Rmn1 == Rmx1 && Rmn2 == Rmx2)														k = 1;

	if (Dz <= 0.) k += 2;
 
	double phi1, phi2, dphi;
	if (Dphi < 0.) {
		phi2 = Phi1; phi1 = phi2 - Dphi;
	}else if (Dphi == 0.) {
		phi1 = Phi1; phi2 = phi1 + wholeCircle;
	}else{
		phi1 = Phi1; phi2 = phi1 + Dphi;
	}
	dphi	= phi2 - phi1;
	if (std::abs(dphi-wholeCircle) < perMillion) dphi = wholeCircle;
	if (dphi > wholeCircle) k += 4; 

	if (k != 0) {
		std::cerr << "UPolyhedronCone(s)/Tube(s): error in input parameters";
		if ((k & 1) != 0) std::cerr << " (radiuses)";
		if ((k & 2) != 0) std::cerr << " (half-length)";
		if ((k & 4) != 0) std::cerr << " (angles)";
		std::cerr << std::endl;
		std::cerr << " Rmn1=" << Rmn1 << " Rmx1=" << Rmx1;
		std::cerr << " Rmn2=" << Rmn2 << " Rmx2=" << Rmx2;
		std::cerr << " Dz=" << Dz << " Phi1=" << Phi1 << " Dphi=" << Dphi
							<< std::endl;
		return;
	}
	
	//	 P R E P A R E	 T W O	 P O L Y L I N E S

	std::vector<double> zz(4), rr(4);
	zz[0] =	Dz; 
	zz[1] = -Dz; 
	zz[2] =	Dz; 
	zz[3] = -Dz; 
	rr[0] =	Rmx2;
	rr[1] =	Rmx1;
	rr[2] =	Rmn2;
	rr[3] =	Rmn1;

	//	 R O T A T E		P O L Y L I N E S

	RotateAroundZ(0, phi1, dphi, 2, 2, zz, rr, -1, -1); 
	SetReferences();
}

UPolyhedronCons::~UPolyhedronCons() {}

UPolyhedronCone::UPolyhedronCone(double Rmn1, double Rmx1, 
																		 double Rmn2, double Rmx2,
																		 double Dz) :
	UPolyhedronCons(Rmn1, Rmx1, Rmn2, Rmx2, Dz, 0*deg, 360*deg) {}

UPolyhedronCone::~UPolyhedronCone() {}

UPolyhedronTubs::UPolyhedronTubs(double Rmin, double Rmax,
																		 double Dz, 
																		 double Phi1, double Dphi)
	:	 UPolyhedronCons(Rmin, Rmax, Rmin, Rmax, Dz, Phi1, Dphi) {}

UPolyhedronTubs::~UPolyhedronTubs() {}

UPolyhedronTube::UPolyhedronTube (double Rmin, double Rmax,
																			double Dz)
	: UPolyhedronCons(Rmin, Rmax, Rmin, Rmax, Dz, 0*deg, 360*deg) {}

UPolyhedronTube::~UPolyhedronTube () {}

UPolyhedronPgon::UPolyhedronPgon(double phi,
																		 double dphi,
																		 int		npdv,
																		 int		nz,
																		 const double *z,
																		 const double *rmin,
																		 const double *rmax)
/***********************************************************************
 *																																		 *
 * Name: UPolyhedronPgon													 Date:		09.12.96 *
 * Author: E.Chernyaev															 Revised:					*
 *																																		 *
 * Function: Constructor of polyhedron for PGON, PCON									*
 *																																		 *
 * Input: phi	- initial phi																					 *
 *				dphi - delta phi																						 *
 *				npdv - number of steps along phi														 *
 *				nz	 - number of z-planes (at least two)										 *
 *				z[]	- z coordinates of the slices													 *
 *				rmin[] - smaller r at the slices														 *
 *				rmax[] - bigger	r at the slices														 *
 *																																		 *
 ***********************************************************************/
{
	//	 C H E C K	 I N P U T	 P A R A M E T E R S

	if (dphi <= 0. || dphi > 2*UUtils::kPi) {
		std::cerr
			<< "UPolyhedronPgon/Pcon: wrong delta phi = " << dphi
			<< std::endl;
		return;
	}		
		
	if (nz < 2) {
		std::cerr
			<< "UPolyhedronPgon/Pcon: number of z-planes less than two = " << nz
			<< std::endl;
		return;
	}

	if (npdv < 0) {
		std::cerr
			<< "UPolyhedronPgon/Pcon: error in number of phi-steps =" << npdv
			<< std::endl;
		return;
	}

	int i;
	for (i=0; i<nz; i++) {
		if (rmin[i] < 0. || rmax[i] < 0. || rmin[i] > rmax[i]) {
			std::cerr
				<< "UPolyhedronPgon: error in radiuses rmin[" << i << "]="
				<< rmin[i] << " rmax[" << i << "]=" << rmax[i]
				<< std::endl;
			return;
		}
	}

	//	 P R E P A R E	 T W O	 P O L Y L I N E S

	std::vector<double> zz(2*nz), rr(2*nz);

	if (z[0] > z[nz-1]) {
		for (i=0; i<nz; i++) {
			zz[i]		= z[i];
			rr[i]		= rmax[i];
			zz[i+nz] = z[i];
			rr[i+nz] = rmin[i];
		}
	}else{
		for (i=0; i<nz; i++) {
			zz[i]		= z[nz-i-1];
			rr[i]		= rmax[nz-i-1];
			zz[i+nz] = z[nz-i-1];
			rr[i+nz] = rmin[nz-i-1];
		}
	}

	//	 R O T A T E		P O L Y L I N E S

	RotateAroundZ(npdv, phi, dphi, nz, nz, zz, rr, -1, (npdv == 0) ? -1 : 1); 
	SetReferences();
}

UPolyhedronPgon::~UPolyhedronPgon() {}

UPolyhedronPcon::UPolyhedronPcon(double phi, double dphi, int nz,
																		 const double *z,
																		 const double *rmin,
																		 const double *rmax)
	: UPolyhedronPgon(phi, dphi, 0, nz, z, rmin, rmax) {}

UPolyhedronPcon::~UPolyhedronPcon() {}

UPolyhedronSphere::UPolyhedronSphere(double rmin, double rmax,
																				 double phi, double dphi,
																				 double the, double dthe)
/***********************************************************************
 *																																		 *
 * Name: UPolyhedronSphere												 Date:		11.12.96 *
 * Author: E.Chernyaev (IU/Protvino)							 Revised:					*
 *																																		 *
 * Function: Constructor of polyhedron for SPHERE											*
 *																																		 *
 * Input: rmin - internal radius																			 *
 *				rmax - external radius																			 *
 *				phi	- initial phi																					 *
 *				dphi - delta phi																						 *
 *				the	- initial theta																				 *
 *				dthe - delta theta																					 *
 *																																		 *
 ***********************************************************************/
{
	//	 C H E C K	 I N P U T	 P A R A M E T E R S

	if (dphi <= 0. || dphi > 2*UUtils::kPi) {
		std::cerr
			<< "UPolyhedronSphere: wrong delta phi = " << dphi
			<< std::endl;
		return;
	}		

	if (the < 0. || the > UUtils::kPi) {
		std::cerr
			<< "UPolyhedronSphere: wrong theta = " << the
			<< std::endl;
		return;
	}		
	
	if (dthe <= 0. || dthe > UUtils::kPi) {
		std::cerr
			<< "UPolyhedronSphere: wrong delta theta = " << dthe
			<< std::endl;
		return;
	}		

	if (the+dthe > UUtils::kPi) {
		std::cerr
			<< "UPolyhedronSphere: wrong theta + delta theta = "
			<< the << " " << dthe
			<< std::endl;
		return;
	}		
	
	if (rmin < 0. || rmin >= rmax) {
		std::cerr
			<< "UPolyhedronSphere: error in radiuses"
			<< " rmin=" << rmin << " rmax=" << rmax
			<< std::endl;
		return;
	}

	//	 P R E P A R E	 T W O	 P O L Y L I N E S

	int nds = (GetNumberOfRotationSteps() + 1) / 2;
	int np1 = int(dthe*nds/UUtils::kPi+.5) + 1;
	if (np1 <= 1) np1 = 2;
	int np2 = rmin < perMillion ? 1 : np1;

	std::vector<double> zz(np1+np2), rr(np1+np2);

	double a = dthe/(np1-1);
	double cosa, sina;
	for (int i=0; i<np1; i++) {
		cosa	= std::cos(the+i*a);
		sina	= std::sin(the+i*a);
		zz[i] = rmax*cosa;
		rr[i] = rmax*sina;
		if (np2 > 1) {
			zz[i+np1] = rmin*cosa;
			rr[i+np1] = rmin*sina;
		}
	}
	if (np2 == 1) {
		zz[np1] = 0.;
		rr[np1] = 0.;
	}

	//	 R O T A T E		P O L Y L I N E S

	RotateAroundZ(0, phi, dphi, np1, np2, zz, rr, -1, -1); 
	SetReferences();
}

UPolyhedronSphere::~UPolyhedronSphere() {}

UPolyhedronTorus::UPolyhedronTorus(double rmin,
								double rmax,
								double rtor,
								double phi,
								double dphi)
/***********************************************************************
 *																																		 *
 * Name: UPolyhedronTorus													Date:		11.12.96 *
 * Author: E.Chernyaev (IU/Protvino)							 Revised:					*
 *																																		 *
 * Function: Constructor of polyhedron for TORUS											 *
 *																																		 *
 * Input: rmin - internal radius																			 *
 *				rmax - external radius																			 *
 *				rtor - radius of torus																			 *
 *				phi	- initial phi																					 *
 *				dphi - delta phi																						 *
 *																																		 *
 ***********************************************************************/
{
	//	 C H E C K	 I N P U T	 P A R A M E T E R S

	if (dphi <= 0. || dphi > 2*UUtils::kPi) {
		std::cerr
			<< "UPolyhedronTorus: wrong delta phi = " << dphi
			<< std::endl;
		return;
	}

	if (rmin < 0. || rmin >= rmax || rmax >= rtor) {
		std::cerr
			<< "UPolyhedronTorus: error in radiuses"
			<< " rmin=" << rmin << " rmax=" << rmax << " rtorus=" << rtor
			<< std::endl;
		return;
	}

	//	 P R E P A R E	 T W O	 P O L Y L I N E S

	int np1 = GetNumberOfRotationSteps();
	int np2 = rmin < perMillion ? 1 : np1;

	std::vector<double> zz(np1+np2), rr(np1+np2);

	double a = 2*UUtils::kPi/np1;
	double cosa, sina;
	for (int i=0; i<np1; i++) {
		cosa	= std::cos(i*a);
		sina	= std::sin(i*a);
		zz[i] = rmax*cosa;
		rr[i] = rtor+rmax*sina;
		if (np2 > 1) {
			zz[i+np1] = rmin*cosa;
			rr[i+np1] = rtor+rmin*sina;
		}
	}
	if (np2 == 1) {
		zz[np1] = 0.;
		rr[np1] = rtor;
		np2 = -1;
	}

	//	 R O T A T E		P O L Y L I N E S

	RotateAroundZ(0, phi, dphi, -np1, -np2, zz, rr, -1,-1); 
	SetReferences();
}

UPolyhedronTorus::~UPolyhedronTorus() {}

UPolyhedronEllipsoid::UPolyhedronEllipsoid(double ax, double by,
																							 double cz, double zCut1,
																							 double zCut2)
/***********************************************************************
 *																																		 *
 * Name: UPolyhedronEllipsoid											Date:		25.02.05 *
 * Author: G.Guerrieri															 Revised:					*
 *																																		 *
 * Function: Constructor of polyhedron for ELLIPSOID									 *
 *																																		 *
 * Input: ax - semiaxis x																							*
 *				by - semiaxis y																							*
 *				cz - semiaxis z																							*
 *				zCut1 - lower cut plane level (solid lies above this plane)	*
 *				zCut2 - upper cut plane level (solid lies below this plane)	*
 *																																		 *
 ***********************************************************************/
{
	//	 C H E C K	 I N P U T	 P A R A M E T E R S

	if (zCut1 >= cz || zCut2 <= -cz || zCut1 > zCut2) {
		std::cerr << "UPolyhedronEllipsoid: wrong zCut1 = " << zCut1
					 << " zCut2 = " << zCut2
					 << " for given cz = " << cz << std::endl;
		return;
	}
	if (cz <= 0.0) {
		std::cerr << "UPolyhedronEllipsoid: bad z semi-axis: cz = " << cz
			<< std::endl;
		return;
	}

	double dthe;
	double sthe;
	int cutflag;
	cutflag= 0;
	if (zCut2 >= cz)
		{
			sthe= 0.0;
		}
	else
		{
			sthe= std::acos(zCut2/cz);
			cutflag++;
		}
	if (zCut1 <= -cz)
		{
			dthe= UUtils::kPi - sthe;
		}
	else
		{
			dthe= std::acos(zCut1/cz)-sthe;
			cutflag++;
		}

	//	 P R E P A R E	 T W O	 P O L Y L I N E S
	//	 generate sphere of radius cz first, then rescale x and y later

	int nds = (GetNumberOfRotationSteps() + 1) / 2;
	int np1 = int(dthe*nds/UUtils::kPi) + 2 + cutflag;

	std::vector<double> zz(np1+1), rr(np1+1);

//	if (!zz || !rr)
		{
			// UException("UPolyhedronEllipsoid::UPolyhedronEllipsoid",
			//"greps1002", FatalException, "Out of memory");
		}

	double a = dthe/(np1-cutflag-1);
	double cosa, sina;
	int j=0;
	if (sthe > 0.0)
		{
			zz[j]= zCut2;
			rr[j]= 0.;
			j++;
		}
	for (int i=0; i<np1-cutflag; i++) {
		cosa	= std::cos(sthe+i*a);
		sina	= std::sin(sthe+i*a);
		zz[j] = cz*cosa;
		rr[j] = cz*sina;
		j++;
	}
	if (j < np1)
		{
			zz[j]= zCut1;
			rr[j]= 0.;
			j++;
		}
	if (j > np1)
		{
			std::cerr << "Logic error in UPolyhedronEllipsoid, memory corrupted!"
								<< std::endl;
		}
	if (j < np1)
		{
			std::cerr << "Warning: logic error in UPolyhedronEllipsoid."
								<< std::endl;
			np1= j;
		}
	zz[j] = 0.;
	rr[j] = 0.;

	
	//	 R O T A T E		P O L Y L I N E S

	RotateAroundZ(0, 0.0, 2*UUtils::kPi, np1, 1, zz, rr, -1, 1); 
	SetReferences();

	// rescale x and y vertex coordinates
	{
		int n = GetNoVertices();
		for (int i=0; i < n; i++) {
		    UPoint3D &p= fVertices[i];
			p.x *= ax/cz;
			p.y *= by/cz;
		}
	}
}

UPolyhedronEllipsoid::~UPolyhedronEllipsoid() {}

UPolyhedronEllipticalCone::UPolyhedronEllipticalCone(double ax,
																												 double ay,
																												 double h,
																												 double zTopCut) 
/***********************************************************************
 *																																		 *
 * Name: UPolyhedronEllipticalCone								 Date:		8.9.2005 *
 * Author: D.Anninos																 Revised: 9.9.2005 *
 *																																		 *
 * Function: Constructor for EllipticalCone														*
 *																																		 *
 * Input: ax, ay		 - X & Y semi axes at z = 0												*
 *				h					- height of full cone														 *
 *				zTopCut		- Top Cut in Z Axis															 *
 *																																		 *
 ***********************************************************************/
{
	//	 C H E C K	 I N P U T	 P A R A M E T E R S

	int k = 0;
	if ( (ax <= 0.) || (ay <= 0.) || (h <= 0.) || (zTopCut <= 0.) ) { k = 1; }

	if (k != 0) {
		std::cerr << "UPolyhedronCone: error in input parameters";
		std::cerr << std::endl;
		return;
	}
	
	//	 P R E P A R E	 T W O	 P O L Y L I N E S

	zTopCut = (h >= zTopCut ? zTopCut : h);

	std::vector<double> zz(4), rr(4);

	zz[0] =	 zTopCut; 
	zz[1] =	-zTopCut; 
	zz[2] =	 zTopCut; 
	zz[3] =	-zTopCut; 
	rr[0] =	(h-zTopCut);
	rr[1] =	(h+zTopCut);
	rr[2] =	0.;
	rr[3] =	0.;

	//	 R O T A T E		P O L Y L I N E S

	RotateAroundZ(0, 0., 2*UUtils::kPi, 2, 2, zz, rr, -1, -1); 
	SetReferences();

	// rescale x and y vertex coordinates
 {
	 int n = GetNoVertices();
	 for (int i=0; i < n; i++) {
		 UPoint3D &p= fVertices[i];
		 p.x *= ax;
		 p.y *= ay;
	 }
 }
}

UPolyhedronEllipticalCone::~UPolyhedronEllipticalCone() {}

int UPolyhedron::fNumberOfRotationSteps = DEFAULT_NUMBER_OF_STEPS;
/***********************************************************************
 *																																		 *
 * Name: UPolyhedron::fNumberOfRotationSteps			 Date:		24.06.97 *
 * Author: J.Allison (Manchester University)				 Revised:					*
 *																																		 *
 * Function: Number of steps for whole circle													*
 *																																		 *
 ***********************************************************************/
