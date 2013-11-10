/*
 * TransformationMatrix.h
 *
 *  Created on: Nov 8, 2013
 *      Author: swenzel
 */

#ifndef TRANSFORMATIONMATRIX_H_
#define TRANSFORMATIONMATRIX_H_

#include "Vector3D.h"

typedef int PlacementIdType;

class TransformationMatrix
{
private:
    // this can be varied depending on template specialization
	double const *trans;
	double const *rot;

	template<PlacementIdType id>
	void foo() const;

	friend class PhysicalBox;

public:
	inline
	void MasterToLocal(Vector3D const &, Vector3D &) const;

	// why not ??
	inline Vector3D MasterToLocal(Vector3D const &) const;

	// for a Vc vector: not so easy
	// inline MasterToLocal(Vc::double) const;


	friend class PhysicalVolume;

};

#endif /* TRANSFORMATIONMATRIX_H_ */
