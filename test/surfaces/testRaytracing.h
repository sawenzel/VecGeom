#ifndef SURF_TEST_RAYTRACING
#define SURF_TEST_RAYTRACING

#include <vector>

#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/surfaces/Model.h>

using Real_t = double;

/// @brief Structure holding boundary crossing info for a single ray
struct CrossingSeq {
  double fStart[6];                              ///< Start position and direction components
  vecgeom::NavigationState fStartState;          ///< State in the start point
  std::vector<double> fSteps;                    ///< Step lengths to next crossing
  std::vector<vecgeom::NavigationState> fStates; ///< States after each crossing

  VECCORE_ATT_HOST_DEVICE
  size_t GetNsteps() const { return fSteps.size(); }

  /// @brief Initialize crossing object
  /// @param x x start pos
  /// @param y y start pos
  /// @param z z start pos
  /// @param dx x start dir
  /// @param dy y start dir
  /// @param dz z start dir
  VECCORE_ATT_HOST_DEVICE
  void Init(double x, double y, double z, double dx, double dy, double dz)
  {
    fSteps.clear();
    fStates.clear();
    fStart[0] = x;
    fStart[1] = y;
    fStart[2] = z;
    fStart[3] = dx;
    fStart[4] = dy;
    fStart[5] = dz;
  }

  /// @brief Set distance and next state for the current crossing
  /// @param distance Distance to next crossing
  /// @param next_state Next state
  /// @return Crossing index
  int SetNextCrossing(double distance, vecgeom::NavigationState const &next_state)
  {
    int istep = fSteps.size();
    fSteps.push_back(distance);
    fStates.push_back(next_state);
    return istep;
  }

  /// @brief Compares two crossing sequences
  /// @param other Sequence to compare to
  /// @param istep_err Index of the divergent step if comparison fails
  /// @return Are the sequences identical
  VECCORE_ATT_HOST_DEVICE
  bool IsEqual(CrossingSeq const &other, int &istep_err, bool accept_zeros = false)
  {
    auto kTolerance    = vecgeom::kToleranceDist<Real_t>;
    size_t istep       = 0;
    size_t istep_other = 0;
    istep_err          = 0;
    while (istep < fSteps.size()) {
      if (accept_zeros) {
        if (istep_other >= other.GetNsteps()) {
          istep_err = istep;
          return false;
        }
        if (vecCore::math::Abs(fSteps[istep]) < kTolerance) {
          istep++;
          continue;
        }
        if (vecCore::math::Abs(other.fSteps[istep_other]) < kTolerance) {
          istep_other++;
          continue;
        }
      }
      // Now the steps must be in sync
      if (fStates[istep].GetState() != other.fStates[istep_other].GetState() ||
          vecCore::math::Abs(fSteps[istep] - other.fSteps[istep_other]) >
              vgbrep::RoundingError(static_cast<Real_t>(other.fSteps[istep_other]), 100 * kTolerance)) {
        istep_err = istep;
        return false;
      }
      istep++;
      istep_other++;
    }
    return true;
  }
};

template <typename InputPrecision, typename OutputPrecision>
vecgeom::Vector3D<OutputPrecision> *convertVectorArray(vecgeom::Vector3D<InputPrecision> *inputArray, int size)
{
  vecgeom::Vector3D<OutputPrecision> *outputArray = new vecgeom::Vector3D<OutputPrecision>[size];
  for (int i = 0; i < size; ++i) {
    // Constructing vecgeom::Vector3D<OutputPrecision> from vecgeom::Vector3D<InputPrecision>
    outputArray[i] = vecgeom::Vector3D<OutputPrecision>(inputArray[i]);
  }
  return outputArray;
}

#endif