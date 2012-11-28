
#ifndef USTL_H
#define USTL_H

#include <iostream>
#include <string>
#include <vector>

#include "UTriangularFacet.hh"
#include "G4TriangularFacet.hh"
#include "G4TessellatedSolid.hh"
#include "G4LogicalVolume.hh"

class USTL
{
public:

	static bool ReadFromSTLBinaryFile(const std::string filename, std::vector<G4TriangularFacet *> &triangles);

	// 	static bool ReadFromBinaryFile(const std::string filename, std::vector<UTriangularFacet *> &triangles); 

	static G4TessellatedSolid *ReadFromSTLBinaryFile(const std::string filename, int maxVoxels=-1);

	static G4TessellatedSolid *TraverseGDML(G4LogicalVolume *root);

	static G4TessellatedSolid *ReadGDML(const std::string filename);

private:
	static const unsigned int headerSize = 80;

};

#endif
