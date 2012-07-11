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
// * technical work of the GEANT4 collaboration.                      *
// * By using,  copying,  modifying or  distributing the software (or *
// * any work based  on the software)  you  agree  to acknowledge its *
// * use  in  resulting  scientific  publications,  and indicate your *
// * acceptance of all terms of the Geant4 Software license.          *
// ********************************************************************
//
//
// SBTperformance.hh
//
// Definition of the batch solid test
//

#ifndef G4Tools_hh
#define G4Tools_hh

#include <vector>

#include "G4VSolid.hh"
#include "G4Polyhedron.hh"
#include "UVector3.hh"

class G4Tools {

public:
    static inline int GetPolyhedra(const G4VSolid &solid, std::vector<UVector3> &vertices, std::vector<std::vector<int> > &nodes)
    {
        G4Polyhedron *p = solid.GetPolyhedron();
        int noVertices = p->GetNoVertices();
        int noFaces = p->GetNoFacets();
        vertices.resize(noVertices);
        nodes.resize(noFaces);

        int totalNodes = 0;

        UVector3 vertex;
        for (int i = 1; i <= noVertices; i++)
        {
            HepGeom::Point3D<double> point = p->GetVertex(i);
            vertex.Set (point.x(), point.y(), point.z());
            vertices[i-1] = vertex;
        }
        for (int i = 1; i <= noFaces; i++)
        {
            int n;
            std::vector<int> iNodes(4);
            p->GetFacet(i, n, &iNodes[0]);
            nodes[i-1] = iNodes;
            nodes[i-1].resize(n);
            totalNodes += n;
        }
        return totalNodes;
    }
};

#endif
