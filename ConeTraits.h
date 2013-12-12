/*
 * ConeTraits.h
 *
 *  Created on: Nov 29, 2013
 *      Author: swenzel
 */

#ifndef CONETRAITS_H_
#define CONETRAITS_H_

// CONE FACTORY CAN BE COPIED

namespace ConeTraits
{

// define here different kinds of cones
// for the moment take same specialization as Cone

	// a Cone not having rmin nor phi section
	struct NonHollowCone {};
	// a Cone not having at least one rmin but a non-2pi phi section
	struct NonHollowConeWithPhi {};
	// a Cone not having rmin and special case phi = 180^\circ = PI section
	struct NonHollowConeWithPhiEqualsPi {};

	// a Cone having rmin but no phi section
	struct HollowCone {};
	// a Cone having rmin and phi section
	struct HollowConeWithPhi {};

	struct HollowConeWithPhiEqualsPi {};

	// give a traits template
	// this maps cone types to certain characteristics

template <typename T>
struct NeedsPhiTreatment
{
	static const bool value=true;
};
// specializations
template <>
struct NeedsPhiTreatment<NonHollowCone>
{
	static const bool value=false;
};
template <>
struct NeedsPhiTreatment<HollowCone>
{
	static const bool value=false;
};

// *** asking for rmin treatment ***** //
template <typename T>
struct NeedsRminTreatment
{
	static const bool value=true;
};
template <>
struct NeedsRminTreatment<NonHollowCone>
{
	static const bool value=false;
};
template <>
struct NeedsRminTreatment<NonHollowConeWithPhi>
{
	static const bool value=false;
};
template <>
struct NeedsRminTreatment<NonHollowConeWithPhiEqualsPi>
{
	static const bool value=false;
};


template <typename T>
struct IsPhiEqualsPiCase
{
	static const bool value = false;
};

template <>
struct IsPhiEqualsPiCase<NonHollowConeWithPhiEqualsPi>
{
	static const bool value = true;
};

template <>
struct IsPhiEqualsPiCase<HollowConeWithPhiEqualsPi>
{
	static const bool value = true;
};


};
#endif /* CONETRAITS_H_ */
