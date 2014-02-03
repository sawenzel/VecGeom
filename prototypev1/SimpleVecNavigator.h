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
	// for transformed points and dirs
	Vectors3DSOA transformedpoints;
	Vectors3DSOA transformeddirs;

public:
	SimpleVecNavigator(int);
	virtual ~SimpleVecNavigator();


	void
	DistToNextBoundary( PhysicalVolume const *, Vectors3DSOA const & /*points*/,
												Vectors3DSOA const & /*dirs*/,
												double const * /*steps*/,
												double * /*distance*/,
												PhysicalVolume ** nextnode, int np ) const;

	void
	DistToNextBoundaryUsingUnplacedVolumes( PhysicalVolume const *, Vectors3DSOA const & /*points*/,
												Vectors3DSOA const & /*dirs*/,
												double const * /*steps*/,
												double * /*distance*/,
												PhysicalVolume ** nextnode, int np ) const;

	void
	DistToNextBoundaryUsingUnplacedVolumesButSpecializedMatrices( PhysicalVolume const *, Vectors3DSOA const & /*points*/,
													Vectors3DSOA const & /*dirs*/,
													double const * /*steps*/,
													double * /*distance*/,
													PhysicalVolume ** nextnode, int np ) const;


	void
	DistToNextBoundaryUsingROOT( PhysicalVolume const *,
								 double const * /*points*/,
								 double const * /*dirs*/,
								 double const * /*steps*/,
								 double * /*distance*/,
								 PhysicalVolume ** nextnode, int np ) const;

	void
	DistToNextBoundaryUsingUSOLIDS( PhysicalVolume const *, Vectors3DSOA const & /*points*/,
									Vectors3DSOA const & /*dirs*/,
									double const * /*steps*/,
									double * /*distance*/,
									PhysicalVolume ** nextnode, int np ) const;


	static
	PhysicalVolume const *
	LocateGlobalPoint( PhysicalVolume const *, Vector3D const & globalpoint, bool top=true );

};

#endif /* SIMPLEVECNAVIGATOR_H_ */
