/*
 * SimpleVecNavigator.h
 *
 *  Created on: Dec 12, 2013
 *      Author: swenzel
 */

#ifndef SIMPLEVECNAVIGATOR_H_
#define SIMPLEVECNAVIGATOR_H_

#include "PhysicalVolume.h"
#include "Vector3D.h"

class SimpleVecNavigator {
private:
	double * workspace;

public:
	SimpleVecNavigator(int);
	virtual ~SimpleVecNavigator();


	void
	DistToNextBoundary( PhysicalVolume const *, Vectors3DSOA const & /*points*/,
												Vectors3DSOA const & /*dirs*/,
												double const * /*steps*/,
												double * /*distance*/,
												PhysicalVolume ** nextnode, int np ) const;
};

#endif /* SIMPLEVECNAVIGATOR_H_ */
