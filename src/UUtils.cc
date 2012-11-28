///////////////////////////////////////////////////////////////////////////////
//
//  UUtils - Utility namespace providing common constants and mathematical
//      utilities.
//
////////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>

#include "UVector3.hh"
#include "UTransform3D.hh"
#include "UUtils.hh"
#include "VUSolid.hh"

using namespace std;

//______________________________________________________________________________
void UUtils::TransformLimits(UVector3 &min, UVector3 &max, const UTransform3D &transformation)
{
	// The goal of this method is to convert the quantities min and max (representing the
	// bounding box of a given solid in its local frame) to the main frame, using
	// "transformation"
	UVector3 vertices[8] = {   // Detemination of the vertices thanks to the extension of each solid:
		UVector3(min.x, min.y, min.z), // 1st vertice:
		UVector3(min.x, max.y, min.z), // 2nd vertice:
		UVector3(max.x, max.y, min.z),
		UVector3(max.x, min.y, min.z),
		UVector3(min.x, min.y, max.z),
		UVector3(min.x, max.y, max.z),
		UVector3(max.x, max.y, max.z),
		UVector3(max.x, min.y, max.z)
	};

	min.Set(kInfinity); max.Set(-kInfinity);

	// Loop on th vertices
	int limit = sizeof(vertices) / sizeof(UVector3);
	for(int i = 0 ; i < limit; i++)
	{
		// From local frame to the gobal one:
		// Current positions on the three axis:         
		UVector3 current = transformation.GlobalPoint(vertices[i]);

		// If need be, replacement of the min & max values:
		if (current.x > max.x) max.x = current.x;
		if (current.x < min.x) min.x = current.x;

		if (current.y > max.y) max.y = current.y;
		if (current.y < min.y) min.y = current.y;  

		if (current.z > max.z) max.z = current.z;
		if (current.z < min.z) min.z = current.z;                             
	}
}

double UUtils::Random(double min, double max)
{
	// srand((unsigned)time(NULL));
	double number = (double) rand() / RAND_MAX;
	double res = min + number * (max - min);
	return res;
}
 
int UUtils::SaveVectorToMatlabFile(vector<double> &vector, string filename)
{
	ofstream file(filename.c_str());

	// NEW: set precision, use exponential, precision 4 digits
	if (file.is_open())
	{
		int size = vector.size();
		file.precision(16);
		for (int i = 0; i < size; i++)
		{
			double value = vector[i];
			file << value << "\n";
		}
		return 0;
	}
	return 1;
}


int UUtils::SaveVectorToMatlabFile(vector<int> &vector, string filename)
{
	ofstream file(filename.c_str());

	if (file.is_open())
	{
		int size = vector.size();
		for (int i = 0; i < size; i++)
		{
			int value = vector[i];
			file << value << "\n";
		}
		return 0;
	}
	return 1;
}

int UUtils::SaveVectorToMatlabFile(vector<UVector3> &vector, string filename)
{
	ofstream file(filename.c_str());

	if (file.is_open())
	{
		int size = vector.size();
		for (int i = 0; i < size; i++) 
		{
			UVector3 &vec = vector[i];
			file << vec.x << "\t" << vec.y << "\t" << vec.z << "\n";
		}
		return 0;
	}
	return 1;
}

string UUtils::ToString(int number)
{
	std::stringstream ss; 
	ss << number;
	return ss.str();
}

string UUtils::ToString(double number)
{
	std::stringstream ss; 
	ss << number;
	return ss.str();
}

int UUtils::FileSize(std::string filePath)
{
	std::streampos fsize = 0;
	std::ifstream file(filePath.c_str(), std::ios::binary);

	fsize = file.tellg();
	file.seekg(0, std::ios::end);
	fsize = file.tellg() - fsize;
	file.close();

	return fsize;
}


int UUtils::StrPos(const string &haystack, const string &needle)
{
    int sleng = haystack.length();
    int nleng = needle.length();

    if (sleng==0 || nleng==0)
        return -1;

    for(int i=0, j=0; i<sleng; j=0, i++ )
    {
        while (i+j<sleng && j<nleng && haystack[i+j]==needle[j])
            j++;
        if (j==nleng)
            return i;
    }
    return -1;
}

/*
void UTessellatedSolid::ImportFromSTLFile(std::string filename)
{
vector <UTriangularFacet *> fFacets;

USTL::ReadFromBinaryFile(filename, fFacets);

int size = fFacets.size();
for (int i = 0; i < size; ++i)
{
UTriangularFacet *facet = fFacets[i];
AddFacet(facet);
}
SetSolidClosed(true);
}
*/

/*
int size = fFacets.size();
for (int j = 0; j < 100; ++j) //2.418 , 2.511
for (int i = 0; i < size; ++i)
{
UFacet &facet = *facetsi[j];
a += facet.GetNumberOfVertices();
}
if (a % rand() == -1) cout << a;
*/

/*
for (int j = 0; j < 100; ++j) //2.917 3.01
{
int size = fFacets.size();
for (int i = 0; i < size; ++i)
{
UFacet &facet = *fFacets[i];
a += facet.GetNumberOfVertices();
}
}
*/

/*
for (int j = 0; j < 100; ++j) // 2.589
{
std::vector<UFacet *>::const_iterator i, begin = fFacets.begin(), end = fFacets.end();
for (i = begin; i < end; ++i)
{
UFacet &facet = *(*i);
a += facet.GetNumberOfVertices();
}
}

return location;
*/
