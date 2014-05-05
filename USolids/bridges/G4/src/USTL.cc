
#include "USTL.hh"
#include <fstream>
#include <string>
#include "UUtils.hh"
#include <vector>
//#include <cstdint>

#include "UTriangularFacet.hh"
#include "G4TriangularFacet.hh"
#include "G4GDMLParser.hh"

using namespace std;

bool USTL::ReadFromSTLBinaryFile(const string filename, vector<G4TriangularFacet *> &triangles)
{
	triangles.clear();

	std::ifstream is(filename.c_str(), std::ios::binary);

	// Read the STL header
	char header[headerSize];
	is.read(header, sizeof(header));

	// Check that stream is OK, if not this may be an ASCII file
	if (!is.good())
	{
		std::cout << "Problem reading header, perhaps file is not binary" << std::endl;
		return false;
	}

	// Read the number of triangles in the STl file
	// (note: read as int so we can check whether >2^31)
	int nTris = 0;
	is.read((char*) &nTris, 4);

	for (int i = 0; i < nTris; i++)
	{
		float points[9], normal[3];
		char attribute[2];

		for (int j = 0; j < 3; j++) is.read((char *)(normal+j), 4);
		for (int j = 0; j < 9; j++) is.read((char *)(points+j), 4);
		is.read(attribute, 2);

		G4ThreeVector vt[3], n;
		for (int j = 0; j < 3; j++) vt[j].set(points[3*j], points[3*j+1], points[3*j+2]);
		n.set(normal[0], normal[1], normal[2]);

		G4ThreeVector e1 = vt[1] - vt[0];
		G4ThreeVector e2 = vt[2] - vt[0];
		G4ThreeVector surfaceNormal = e1.cross(e2).unit();
		G4ThreeVector dif = n - surfaceNormal;
		bool validNormal = (dif.mag() > 1 && n != G4ThreeVector());
		G4TriangularFacet *f = new G4TriangularFacet(vt[0], vt[1], vt[2], ABSOLUTE);

		if (!validNormal) cerr << "Warning, normal does not match right-hand rule.";
		else 
		{
			//		f->SetSurfaceNormal(G4ThreeVector(normal[0], normal[1], normal[2]));
		}
		triangles.push_back(f);
	}

	return true;
}

G4TessellatedSolid *USTL::ReadFromSTLBinaryFile(const std::string filename, int maxVoxels)
{
	G4TessellatedSolid &tessel = *new G4TessellatedSolid("aTessellatedSolid");
	tessel.SetMaxVoxels(maxVoxels);

	vector <G4TriangularFacet *> facets;
	USTL::ReadFromSTLBinaryFile(filename, facets);

	int size = facets.size();
	for (int i = 0; i < size; ++i)
	{
		G4TriangularFacet *facet = facets[i];
		tessel.AddFacet(facet);
	}
	tessel.SetSolidClosed(true);
	return &tessel;
}


G4TessellatedSolid *USTL::TraverseGDML(G4LogicalVolume *root)
{
	int n = root->GetNoDaughters();
//	G4VSolid *lsolid2 = root->GetSolid();
	for (int i = 0; i < n; i++)
	{
		G4VPhysicalVolume *pvolume = root->GetDaughter(i);
		G4LogicalVolume *lvolume = pvolume->GetLogicalVolume();
		if (lvolume->GetNoDaughters())
		{
			G4TessellatedSolid * res = TraverseGDML(lvolume);
			if (res) return res;
		}
		else
		{
			G4VSolid *lsolid = lvolume->GetSolid();
			G4String name = lsolid->GetEntityType();
			if (name == "G4TessellatedSolid")
			{
				G4TessellatedSolid *tessel = (G4TessellatedSolid *) lsolid;
				return tessel;
			}
		}
	}
	return NULL;
}

G4TessellatedSolid *USTL::ReadGDML(const std::string filename)
{
	G4GDMLParser parser;

	parser.Read(filename, false);
   
	G4VPhysicalVolume *pvolume = parser.GetWorldVolume();
	G4LogicalVolume *root = pvolume->GetLogicalVolume();

	return TraverseGDML(root);
}
