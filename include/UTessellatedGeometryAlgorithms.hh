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
// $Id:
// GEANT4 tag $Name: not supported by cvs2svn $
//
// Author: Marek Gayer, started from original implementation by P R Truscott, 2007
//
//
// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//
// Class description:
//
//   The G4TessellatedGeometryAlgorithms class is used to contain standard
//   routines to determine whether (and if so where) simple geometric shapes
//   intersect.
//   
//   The constructor doesn't need to do anything, and neither does the
//   destructor.
//   
//   IntersectLineAndTriangle2D
//     Determines whether there is an intersection between a line defined
//     by r = p + s.v and a triangle defined by verticies P0, P0+E0 and P0+E1.
//     Here:
//        p = 2D vector
//        s = scaler on [0,infinity)
//        v = 2D vector
//        P0, E0 and E1 are 2D vectors
//     Information about where the intersection occurs is returned in the
//     variable location.
//
//   IntersectLineAndLineSegment2D
//     Determines whether there is an intersection between a line defined
//     by r = P0 + s.D0 and a line-segment with endpoints P1 and P1+D1.
//     Here:
//        P0 = 2D vector
//        s  = scaler on [0,infinity)
//        D0 = 2D vector
//        P1 and D1 are 2D vectors
//     Information about where the intersection occurs is returned in the
//     variable location.

///////////////////////////////////////////////////////////////////////////////
#ifndef UTessellatedGeometryAlgorithms_hh
#define UTessellatedGeometryAlgorithms_hh 1

#include "UVector2.hh"

class UTessellatedGeometryAlgorithms
{
public:

	static bool IntersectLineAndTriangle2D (const UVector2 &p,
		const UVector2 &v,
		const UVector2 &p0, 
		const UVector2 &e0,
		const UVector2 &e1,
		UVector2 location[2]);

	static int IntersectLineAndLineSegment2D (const UVector2 &p0,
		const UVector2 &d0,
		const UVector2 &p1,
		const UVector2 &d1,
		UVector2 location[2]);

	static double Cross(const UVector2 &v1, const UVector2 &v2);

};

///////////////////////////////////////////////////////////////////////////////
//
// CrossProduct
//
// This is just a ficticious "cross-product" function for two 2D vectors...
// "ficticious" because such an operation is not relevant to 2D space compared
// with 3D space.
//

#endif
