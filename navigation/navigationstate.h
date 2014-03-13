/*
 * navigationstate.h
 *
 *  Created on: Mar 12, 2014
 *      Author: swenzel
 */

#ifndef NAVIGATIONSTATE_H_
#define NAVIGATIONSTATE_H_

#include "volumes/placed_volume.h"
#include "base/transformation_matrix.h"
#include "string.h"
#include <iostream>

namespace vecgeom
{
/**
 * a class describing a current geometry state
 * likely there will be such an object for each
 * particle/track currently treated
 */
class NavigationState
{
private:
	int maxlevel_;
	int currentlevel_;
	VPlacedVolume const * * path_;
	TransformationMatrix const * * cache_of_global_matrices_;

	// add other navigation state here, stuff like:

	// some private management methods
	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	void InitInternalStorage();

public:
	// constructors and assignment operators
	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	NavigationState( int );

	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	NavigationState( NavigationState const & rhs );

	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	NavigationState & operator=( NavigationState const & rhs );

	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	~NavigationState( );


	// what else: operator new etc...

	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	int GetMaxLevel() const {return maxlevel_;}

	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	int GetCurrentLevel() const {return currentlevel_;}

	// better to use pop and push
	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	void
	Push(VPlacedVolume const *);

	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	VPlacedVolume const *
	Top() const;

	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	void Pop();

//	int Distance(NavigationState const &) const;

	// clear all information
	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	void Clear();

	void Print() const;

	//void GetGlobalMatrixFromPath( TransformationMatrix *const m ) const;
	//TransformationMatrix const * GetGlobalMatrixFromPath() const;
};



NavigationState & NavigationState::operator=( NavigationState const & rhs )
{

	return *this;
}


NavigationState::NavigationState( NavigationState const & rhs ) : maxlevel_(rhs.maxlevel_), currentlevel_(rhs.currentlevel_)
{
	InitInternalStorage();
	std::memcpy(path_, rhs.path_, sizeof(path_)*currentlevel_ );
}


// implementations follow
NavigationState::NavigationState( int maxlevel ) : maxlevel_(maxlevel), currentlevel_(0)
{
	InitInternalStorage();
}

void
NavigationState::InitInternalStorage()
{
	path_ = new VPlacedVolume const *[maxlevel_];
}


NavigationState::~NavigationState()
{
	delete[] path_;
}


void
NavigationState::Pop()
{
	if(currentlevel_ > 0) path_[currentlevel_--]=0;
}

void
NavigationState::Clear()
{
	currentlevel_=0;
}

void
NavigationState::Push( VPlacedVolume const * v )
{
#ifdef DEBUG
	assert( currentlevel_ < maxlevel_ )
#endif
	path_[currentlevel_++]=v;
}

VPlacedVolume const *
NavigationState::Top() const
{
	return (currentlevel_ > 0 )? path_[currentlevel_-1] : 0;
}

void NavigationState::Print() const
{
	std::cerr << "maxlevel " << maxlevel_ << std::endl;
	std::cerr << "currentlevel " << currentlevel_ << std::endl;
	std::cerr << "deepest volume " << Top() << std::endl;
}



}

#endif /* NAVIGATIONSTATE_H_ */
