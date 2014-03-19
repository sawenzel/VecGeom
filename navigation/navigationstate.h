/*
 * navigationstate.h
 *
 *  Created on: Mar 12, 2014
 *      Author: swenzel
 */

#ifndef NAVIGATIONSTATE_H_
#define NAVIGATIONSTATE_H_

#include <string>
#include <iostream>

#include "backend.h"
#include "base/transformation_matrix.h"
#include "volumes/placed_volume.h"

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
	TransformationMatrix global_matrix_;

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
	TransformationMatrix const &
	TopMatrix();

	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	Vector3D<Precision>
	GlobalToLocal(Vector3D<Precision> const &);

	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	void Pop();

	VECGEOM_INLINE
	VECGEOM_CUDA_HEADER_BOTH
	int Distance( NavigationState const & ) const;
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
	currentlevel_=rhs.currentlevel_;
	maxlevel_ = rhs.maxlevel_;
	std::memcpy(path_, rhs.path_, sizeof(path_)*currentlevel_);
	return *this;
}


NavigationState::NavigationState( NavigationState const & rhs ) : maxlevel_(rhs.maxlevel_), currentlevel_(rhs.currentlevel_)
{
	InitInternalStorage();
	std::memcpy(path_, rhs.path_, sizeof(path_)*rhs.currentlevel_ );
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

VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
TransformationMatrix const &
NavigationState::TopMatrix()
{
// this could be actually cached in case the path does not change ( particle stays inside a volume )
	global_matrix_.CopyFrom( *(path_[0]->matrix()) );
	for(int i=1;i<currentlevel_;++i)
	{
		global_matrix_.MultiplyFromRight( *(path_[i]->matrix()) );
	}
	return global_matrix_;
}

/**
 * function that transforms a global point to local point in reference frame of deepest volume in current navigation state
 * ( equivalent to using a global matrix )
 */
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
Vector3D<Precision>
NavigationState::GlobalToLocal(Vector3D<Precision> const & globalpoint)
{
	Vector3D<Precision> tmp=globalpoint;
	Vector3D<Precision> current;
	for(int level=0;level<currentlevel_;++level)
	{
		TransformationMatrix const *m = path_[level]->matrix();
		current = m->Transform<1,0,Precision>( tmp );
		tmp = current;
	}
	return tmp;
}

void NavigationState::Print() const
{
	std::cerr << "maxlevel " << maxlevel_ << std::endl;
	std::cerr << "currentlevel " << currentlevel_ << std::endl;
	std::cerr << "deepest volume " << Top() << std::endl;
}

/**
 * calculates if other navigation state takes a different branch in geometry path or is on same branch
 * ( two states are on same branch if one can connect the states just by going upwards or downwards ( or do nothing ))
 */
VECGEOM_INLINE
VECGEOM_CUDA_HEADER_BOTH
int NavigationState::Distance( NavigationState const & other ) const
{
	int lastcommonlevel=0;
	int maxlevel = Max( GetCurrentLevel() , other.GetCurrentLevel() );

	//  algorithm: start on top and go down until paths split
	for(int i=0; i < maxlevel; i++)
	{
		VPlacedVolume const *v1 = this->path_[i];
		VPlacedVolume const *v2 = other.path_[i];
		if( v1 == v2 )
		{
			lastcommonlevel = i;
		}
		else
		{
			break;
		}
	}
	return (GetCurrentLevel()-lastcommonlevel) + ( other.GetCurrentLevel() - lastcommonlevel ) - 2;
}


}

#endif /* NAVIGATIONSTATE_H_ */
