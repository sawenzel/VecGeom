/*
 * GeoManager.h
 *
 *  Created on: Nov 11, 2013
 *      Author: swenzel
 */

#ifndef GEOMANAGER_H_
#define GEOMANAGER_H_

#include "PhysicalVolume.h"
#include "PhysicalBox.h"
#include "PhysicalTube.h"
#include "TransformationMatrix.h"
#include <list>

class PhysicalVolume;
// the GeoManager manages the geometry hierarchy. It keeps a pointer to the top volume
// etc.

// is a singleton class
class GeoManager
{
//private:
	//PhysicalVolume const *top;

public:
	// a couple of useful static matrix instances
//	static TransformationMatrix const * IdentityTransformationMatrix;

	//void Export(){}
	//void Import(){}
	//void ExportToDatabase(){}

	//PhysicalVolume const * GetTop() const {return top;}

	//int GetNumberOfPlacedVolumes(){return 0;}
	//int GetNumberOfLogicalVolumes(){return 0;}

	//
	//void GetListOfLogicalVolumes(){;}


	// it would be nice to have something like an SQL query here
	//

	// static factory methods
	static
	PhysicalVolume * MakePlacedBox( BoxParameters const *, TransformationMatrix const * );

	static
	PhysicalVolume * MakePlacedTube( TubeParameters<> const *, TransformationMatrix const * );

	// we need to solve problem of pointer ownership --> let's use smart pointers and move semantics
};

// static const GeoManager::IdentityTransformationMatrix=new TransformationMatrix(0,0,0,0,0,0);


#endif /* GEOMANAGER_H_ */
