/*
 * navigationstate.h
 *
 *  Created on: Mar 12, 2014
 *      Author: swenzel
 */

#ifndef NAVIGATIONSTATE_H_
#define NAVIGATIONSTATE_H_

#include "placed_volume.h"


/**
 * a class describing a current geometry state
 * likely there will be such an object for each
 * particle/track currently treated
 */

class VPlacedVolume;

class NavigationState
{
private:
	int fmaxlevel;
	int fcurrentlevel;
	VPlacedVolume const * * path;
	TransformationMatrix const * * cache_of_global_matrices;

	// add other navigation state here, stuff like:

public:
	NavigationState( int );
	NavigationState( NavigationState const & rhs );
	inline NavigationState & operator=( NavigationState const & rhs );

	// what else: operator new etc...

	inline int GetMaxLevel() const {return fmaxlevel;}
	inline int GetCurrentLevel() const {return fcurrentlevel;}

	inline VPlacedVolume const * At(int i) const;
	inline void SetAt(int i, VPlacedVolume const *);

	// better to use pop and push
	inline void Push(VPlacedVolume const *);
	inline VPlacedVolume const * Top() const;
	inline void Pop();

	int Distance(NavigationState const &) const;
	// clear all information
	void Clear();

	void Print() const;

	void GetGlobalMatrixFromPath( TransformationMatrix * m ) const;
};


#endif /* NAVIGATIONSTATE_H_ */
