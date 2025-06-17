#ifndef SURF_TEST_RAYTRACING
#define SURF_TEST_RAYTRACING

#include <cstdint>
#include <vector>

#include <VecGeom/navigation/NavigationState.h>
#include <VecGeom/surfaces/Model.h>

using Real_t = double;

struct TestConfig {
  int nrays{10000};
  int debug{0};
  int verbosity{0};
  int min_per_scene{1000};
  int ongpu{1};
  int max_cross{vecgeom::kMaximumInt};
  bool detect_overlaps{false};
  bool accept_zeros{false};
  bool test_bvh{false};
  bool bvh_single_step{false};
  bool bvh_split_step{false};
  bool validate_results{true};
  bool only_surf{false};
  bool use_surf{true};
  bool use_TB_gun{false};
  bool on_boundary{false};
  bool test_relocate{false};
  double mmunit{1.};
  double safety_ratio{0.};
  double step_limit{vecgeom::kInfLength};
  vecgeom::NavigationState input_state; ///< Force input state (only works when a point is provided)
  vecgeom::NavigationState next_state;  ///< Triggers computation of normal for the input state
  int compute_normal{0};
};

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
  /// @param istep_err_other Index of the diverging step of the sequence to compare to
  /// @param accept_zeros whether 0 steps should throw an error or not
  /// @return Are the sequences identical
  VECCORE_ATT_HOST_DEVICE
  bool IsEqual(CrossingSeq const &other, int &istep_err, int &istep_err_other, bool accept_zeros = false)
  {
    auto kTolerance         = 10 * vecgeom::kToleranceStrict<Real_t>;
    size_t istep            = 0;
    size_t istep_other      = 0;
    size_t istep_next       = 0;
    size_t istep_next_other = 0;
    istep_err               = 0;
    while (istep < fSteps.size()) {
      if (accept_zeros) {
        if (istep_other >= other.GetNsteps()) {
          istep_err = istep;
          return false;
        }
        if (vecCore::math::Abs(fSteps[istep]) < 1000 * kTolerance) {
          if (vecCore::math::Abs(fSteps[istep] - other.fSteps[istep_other]) < 100 * kTolerance) istep_other++;
          istep++;
          continue;
        }
        if (vecCore::math::Abs(other.fSteps[istep_other]) < 1000 * kTolerance) {
          if (vecCore::math::Abs(fSteps[istep] - other.fSteps[istep_other]) < 100 * kTolerance) istep++;
          istep_other++;
          continue;
        }
      }
      // Now the steps must be in sync
      if (vecCore::math::Abs(fSteps[istep] - other.fSteps[istep_other]) >
          vgbrep::RoundingError(static_cast<Real_t>(other.fSteps[istep_other]), 1000 * kTolerance)) {
        istep_err       = istep;
        istep_err_other = istep_other;

        return false;
      }

      // However, the states must not be in sync, because if the next step was a zero step, the states only need to be
      // in sync after the next zero step. Thus, if the states disagree, check for the next steps, until the next step
      // is not a zero step.
      if (fStates[istep].GetState() != other.fStates[istep_other].GetState()) {

        istep_next       = istep;
        istep_next_other = istep_other;
        // loop over next steps until the next step is not a 0 step
        while (accept_zeros && (istep < fSteps.size() - 1) &&
               vecCore::math::Abs(fSteps[istep_next + 1]) < 1000 * kTolerance) {
          istep_next++;
        }
        while (accept_zeros && (istep_other < other.GetNsteps() - 1) &&
               vecCore::math::Abs(other.fSteps[istep_next_other + 1]) < 1000 * kTolerance) {
          istep_next_other++;
        }
        // check states again, if they still differ, this is an error
        if (fStates[istep_next].GetState() != other.fStates[istep_next_other].GetState()) {
          istep_err       = istep - 1;
          istep_err_other = istep_other - 1;
          return false;
        } else {
          istep       = istep_next;
          istep_other = istep_next_other;
        }
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