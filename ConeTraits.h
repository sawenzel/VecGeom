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
// for the moment take same specialization as tube


	// a tube not having rmin nor phi section
	struct NonHollowCone {};
	// a tube not having at least one rmin but a non-2pi phi section
	struct NonHollowConeWithPhi {};

	// a tube having rmin but no phi section
	struct HollowCone {};
	// a tube having rmin and phi section
	struct HollowConeWithPhi {};


	// give a traits template
	// this maps cone types to certain characteristics

	// *** asking for Phi treatment **** //
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
struct NeedsRminTreatment<NonHollowConeWithPhi>
{
	static const bool value=false;
};

};
#endif /* CONETRAITS_H_ */
