/*
 * TubeTraits.h
 *
 *  Created on: Nov 29, 2013
 *      Author: swenzel
 */

#ifndef TUBETRAITS_H_
#define TUBETRAITS_H_

namespace TubeTraits
{

// define here different kinds of tubes


// a tube not having rmin nor phi section
struct NonHollowTube {};
// a tube not having rmin but a non-2pi phi section
struct NonHollowTubeWithPhi {};

// a tube having rmin but no phi section
struct HollowTube {};
// a tube having rmin and phi section
struct HollowTubeWithPhi {};


// give a traits template
// this maps tube types to certain characteristics

// *** asking for Phi treatment **** //

template <typename T>
struct NeedsPhiTreatment
{
	static const bool value=true;
};
// specializations
template <>
struct NeedsPhiTreatment<NonHollowTube>
{
	static const bool value=false;
};
template <>
struct NeedsPhiTreatment<HollowTube>
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
struct NeedsRminTreatment<NonHollowTube>
{
	static const bool value=false;
};
template <>
struct NeedsRminTreatment<NonHollowTubeWithPhi>
{
	static const bool value=false;
};

};
#endif /* TUBETRAITS_H_ */
