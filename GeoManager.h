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
#include <list>

// the GeoManager manages the geometry hierarchy. It keeps a pointer to the top volume
// etc.

class GeoManager
{
private:
	PhysicalVolume const *top;

public:
	void Export();
	void Import();
	void ExportToDatabase();

	PhysicalVolume const * GetTop() const {return top;}

	int GetNumberOfPlacedVolumes();
	int GetNumberOfLogicalVolumes();

	//
	void GetListOfLogicalVolumes();


	// it would be nice to have something like an SQL query here
	//


	// static factory methods
	static
	PhysicalVolume * MakePlacedBox( BoxParameters const *, TransformationMatrix const * );


	// we need to solve problem of pointer ownership --> let's use smart pointers and move semantics

};


#endif /* GEOMANAGER_H_ */
