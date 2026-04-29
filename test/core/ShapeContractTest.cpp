/**
 * @file ShapeContractTest.cpp
 * @brief CLI runner for sampled helper families and curated manual edge cases.
 *
 * The executable is the developer-facing entry point for the ShapeTester
 * redesign. It exposes family/tier/case selection, list and help commands,
 * replay of sampled failures, and manual hand-picked reproductions while
 * reusing one in-memory sample cache per solid/tier run.
 *
 * See docs/shape_testing.md for the user-facing contract catalog, tier model,
 * replay workflow, and extension procedure.
 */

#undef NDEBUG

#include "VecGeom/base/FpeEnable.h"
#include "VecGeom/base/Stopwatch.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "VecGeom/base/Assert.h"
#include "VecGeom/volumes/PlacedVolume.h"
#include "VecGeomTest/ShapeContractChecks.h"
#include "VecGeomTest/TestCaseManualEdgeCases.h"
#include "VecGeomTest/TestCaseSolids.h"

using vecgeom::Precision;
using Vec_t = vecgeom::Vector3D<Precision>;

namespace {

enum class ShapeContractTier { kFast, kMedium, kSlow };
enum class ShapeContractTestFamily {
  kContracts,
  kNormals,
  kSurface,
  kDistanceToOut,
  kDistanceToIn,
  kSafeties,
  kHitConsistency,
  kManualEdgeCases,
  kAll
};

enum class ShapeBenchmarkOperation {
  kInside,
  kNormal,
  kDistanceToIn,
  kDistanceToOut,
  kSafetyToIn,
  kSafetyToOut
};

// Parsed CLI state shared by the list, sampled-family, and manual-edge-case
// execution paths.
struct ShapeContractOptions {
  std::string case_name        = "all";
  std::string family_name      = "all";
  std::string tier_name        = "fast";
  std::string test_family_name = "contracts";
  std::string manual_case_name = "all";
  std::string manual_method    = "all";
  std::string manual_topology  = "all";
  int npoints                  = -1;
  int seed                     = -1;
  int stream_id                = -1;
  int replay_index             = -1;
  int benchmark_repetitions    = 9;
  int benchmark_warmup         = 3;
  int benchmark_target_calls   = 100000;
  Precision grazing_tolerance  = static_cast<Precision>(-1.);
  bool benchmark_mode          = false;
  bool show_help               = false;
  bool list_cases              = false;
  bool list_families           = false;
  bool list_test_families      = false;
  bool list_manual_cases       = false;
  bool used_defaults           = true;
};

struct ShapeBenchmarkWorkload {
  std::string label;
  ShapeBenchmarkOperation operation = ShapeBenchmarkOperation::kInside;
  int offset                        = 0;
  int count                         = 0;
};

struct ShapeBenchmarkSummary {
  std::string label;
  int samples                    = 0;
  int loops_per_repeat           = 0;
  int warmup_repetitions         = 0;
  int measured_repetitions       = 0;
  std::uint64_t calls_per_repeat = 0;
  Precision mean_seconds         = 0.;
  Precision stddev_seconds       = 0.;
  Precision min_seconds          = 0.;
  Precision max_seconds          = 0.;
  Precision ns_per_call          = 0.;
  Precision rel_sigma_percent    = 0.;
};

struct ShapeContractOutcome {
  vecgeom::test::ShapeContractCheckSummary summary;
  vecgeom::test::ShapeCheckResult result;
};

struct ShapeNormalOutcome {
  vecgeom::test::ShapeNormalCheckSummary summary;
  vecgeom::test::ShapeCheckResult result;
};

struct ShapeSurfaceOutcome {
  vecgeom::test::ShapeSurfaceCheckSummary summary;
  vecgeom::test::ShapeCheckResult result;
};

struct ShapeDistanceToOutOutcome {
  vecgeom::test::ShapeDistanceToOutCheckSummary summary;
  vecgeom::test::ShapeCheckResult result;
};

struct ShapeDistanceToInOutcome {
  vecgeom::test::ShapeDistanceToInCheckSummary summary;
  vecgeom::test::ShapeCheckResult result;
};

struct ShapeSafetyOutcome {
  vecgeom::test::ShapeSafetyCheckSummary summary;
  vecgeom::test::ShapeCheckResult result;
};

struct ShapeHitConsistencyOutcome {
  vecgeom::test::ShapeHitConsistencyCheckSummary summary;
  vecgeom::test::ShapeCheckResult result;
};

// Translate enum selections back into the stable user-facing CLI spellings.
const char *TierLabel(ShapeContractTier tier)
{
  switch (tier) {
  case ShapeContractTier::kFast:
    return "fast";
  case ShapeContractTier::kMedium:
    return "medium";
  case ShapeContractTier::kSlow:
    return "slow";
  }
  return "fast";
}

const char *TierDescription(ShapeContractTier tier)
{
  switch (tier) {
  case ShapeContractTier::kFast:
    return "Fast shape contract test";
  case ShapeContractTier::kMedium:
    return "Medium shape contract test";
  case ShapeContractTier::kSlow:
    return "Slow shape contract test";
  }
  return "Fast shape contract test";
}

const char *TestFamilyLabel(ShapeContractTestFamily family)
{
  switch (family) {
  case ShapeContractTestFamily::kContracts:
    return "contracts";
  case ShapeContractTestFamily::kNormals:
    return "normals";
  case ShapeContractTestFamily::kSurface:
    return "surface";
  case ShapeContractTestFamily::kDistanceToOut:
    return "distance_to_out";
  case ShapeContractTestFamily::kDistanceToIn:
    return "distance_to_in";
  case ShapeContractTestFamily::kSafeties:
    return "safeties";
  case ShapeContractTestFamily::kHitConsistency:
    return "hit_consistency";
  case ShapeContractTestFamily::kManualEdgeCases:
    return "manual_edge_cases";
  case ShapeContractTestFamily::kAll:
    return "all";
  }
  return "contracts";
}

const char *TestFamilyDescription(ShapeContractTier tier, ShapeContractTestFamily family)
{
  switch (family) {
  case ShapeContractTestFamily::kContracts:
    return tier == ShapeContractTier::kFast
               ? "Fast shape contract test"
               : (tier == ShapeContractTier::kMedium ? "Medium shape contract test" : "Slow shape contract test");
  case ShapeContractTestFamily::kNormals:
    return tier == ShapeContractTier::kFast
               ? "Fast shape normal test"
               : (tier == ShapeContractTier::kMedium ? "Medium shape normal test" : "Slow shape normal test");
  case ShapeContractTestFamily::kSurface:
    return tier == ShapeContractTier::kFast
               ? "Fast shape surface test"
               : (tier == ShapeContractTier::kMedium ? "Medium shape surface test" : "Slow shape surface test");
  case ShapeContractTestFamily::kDistanceToOut:
    return tier == ShapeContractTier::kFast ? "Fast shape DistanceToOut test"
                                            : (tier == ShapeContractTier::kMedium ? "Medium shape DistanceToOut test"
                                                                                  : "Slow shape DistanceToOut test");
  case ShapeContractTestFamily::kDistanceToIn:
    return tier == ShapeContractTier::kFast ? "Fast shape DistanceToIn test"
                                            : (tier == ShapeContractTier::kMedium ? "Medium shape DistanceToIn test"
                                                                                  : "Slow shape DistanceToIn test");
  case ShapeContractTestFamily::kSafeties:
    return tier == ShapeContractTier::kFast
               ? "Fast shape safety test"
               : (tier == ShapeContractTier::kMedium ? "Medium shape safety test" : "Slow shape safety test");
  case ShapeContractTestFamily::kHitConsistency:
    return tier == ShapeContractTier::kFast ? "Fast shape hit-consistency test"
                                            : (tier == ShapeContractTier::kMedium ? "Medium shape hit-consistency test"
                                                                                  : "Slow shape hit-consistency test");
  case ShapeContractTestFamily::kManualEdgeCases:
    return "Manual shape edge-case test";
  case ShapeContractTestFamily::kAll:
    return tier == ShapeContractTier::kFast ? "Fast grouped shape helper test"
                                            : (tier == ShapeContractTier::kMedium ? "Medium grouped shape helper test"
                                                                                  : "Slow grouped shape helper test");
  }
  return TierDescription(tier);
}

ShapeContractTestFamily ParseTestFamilySelection(const std::string &family_name)
{
  if (family_name == "contracts") return ShapeContractTestFamily::kContracts;
  if (family_name == "normals") return ShapeContractTestFamily::kNormals;
  if (family_name == "surface") return ShapeContractTestFamily::kSurface;
  if (family_name == "distance_to_out") return ShapeContractTestFamily::kDistanceToOut;
  if (family_name == "distance_to_in") return ShapeContractTestFamily::kDistanceToIn;
  if (family_name == "safeties") return ShapeContractTestFamily::kSafeties;
  if (family_name == "hit_consistency") return ShapeContractTestFamily::kHitConsistency;
  if (family_name == "manual_edge_cases") return ShapeContractTestFamily::kManualEdgeCases;
  if (family_name == "all") return ShapeContractTestFamily::kAll;
  VECGEOM_VALIDATE(false, << "Unknown shape contract test family '" << family_name
                          << "'. Use -test_family contracts, -test_family normals, -test_family surface, "
                          << "-test_family distance_to_out, -test_family distance_to_in, -test_family safeties, "
                          << "-test_family hit_consistency, -test_family manual_edge_cases, or -test_family all.");
  return ShapeContractTestFamily::kContracts;
}

ShapeContractTier ParseTierSelection(const std::string &tier_name)
{
  if (tier_name == "fast") return ShapeContractTier::kFast;
  if (tier_name == "medium") return ShapeContractTier::kMedium;
  if (tier_name == "slow") return ShapeContractTier::kSlow;
  VECGEOM_VALIDATE(false, << "Unknown shape contract tier '" << tier_name
                          << "'. Use -tier fast, -tier medium, or -tier " << "slow.");
  return ShapeContractTier::kFast;
}

std::string JoinConfiguredCaseNames(const std::string &family_name = "all")
{
  std::ostringstream out;
  bool first = true;
  for (auto const &solid_case : vecgeom::test::GetTestCaseSolids()) {
    if (family_name != "all" && vecgeom::test::GetTestCaseFamilyName(solid_case.name) != family_name) continue;
    if (!first) out << ", ";
    out << solid_case.name;
    first = false;
  }
  return out.str();
}

std::string JoinConfiguredFamilyNames()
{
  std::ostringstream out;
  bool first = true;
  for (auto const &family : vecgeom::test::GetTestCaseFamilyNames()) {
    if (!first) out << ", ";
    out << family;
    first = false;
  }
  return out.str();
}

std::string JoinConfiguredTestFamilyNames()
{
  return "contracts, normals, surface, distance_to_out, distance_to_in, safeties, hit_consistency, "
         "manual_edge_cases, all";
}

// Print the full executable contract so `-help` is enough to discover every
// runner feature without reading the source.
void PrintUsage(const char *argv0)
{
  std::cout
      << "Tiered shape contract, normal, and manual edge-case checks for the configured reusable test solids.\n\n";
  std::cout << "Usage:\n";
  std::cout
      << "  " << argv0
      << " [-tier <fast|medium|slow>] "
         "[-test_family "
         "<contracts|normals|surface|distance_to_out|distance_to_in|safeties|hit_consistency|manual_edge_cases|all>] "
         "[-case_name <name|all>] [-family <name|all>]\n"
      << "             [-npoints <count>] [-seed <seed>] [-stream_id <id>] [-replay_index <index>]\n"
      << "             [-benchmark] [-benchmark_repetitions <count>] [-benchmark_warmup <count>]\n"
      << "             [-benchmark_target_calls <count>]\n"
      << "             [-manual_case_name <name|all>] "
         "[-manual_method <contracts|normals|surface|distance_to_out|distance_to_in|safeties|hit_consistency|all>]\n"
      << "             [-manual_topology <inside|surface|edge|outside|all>]\n"
      << "             [-grazing_tolerance <value>]\n"
      << "             [-list_cases] [-list_families] [-list_test_families] [-list_manual_cases] [-help]\n\n";
  std::cout << "Behavior:\n";
  std::cout << "  With no options, runs the fast contracts family for every configured solid.\n";
  std::cout << "  Use -tier fast for the smaller smoke-style sampling profile.\n";
  std::cout << "  Use -tier medium for the heavier seeded sampling profile.\n";
  std::cout << "  Use -tier slow for the largest nightly-style sampling profile.\n";
  std::cout << "  Use -test_family contracts to run only the extracted contract checks.\n";
  std::cout << "  Use -test_family normals to run only the extracted normal checks.\n";
  std::cout << "  Use -test_family surface to run only the extracted surface-point checks.\n";
  std::cout << "  Use -test_family distance_to_out to run only the extracted inside exit-distance checks.\n";
  std::cout << "  Use -test_family distance_to_in to run only the extracted outside entry-distance checks.\n";
  std::cout << "  Use -test_family safeties to run only the extracted safety-sphere checks.\n";
  std::cout << "  Use -test_family hit_consistency to run only the extracted propagated-hit consistency checks.\n";
  std::cout << "  Use -test_family manual_edge_cases to run curated hand-picked solid/ray reproductions.\n";
  std::cout << "  Use -test_family all to run every available helper family for the selected solid and tier.\n";
  std::cout << "  Use -family <name> to run all configured solids in one shape family.\n";
  std::cout << "  Use -replay_index <index> with a single case and a single test family to replay one sampled\n";
  std::cout << "  point/direction from that family's shared sample cache.\n";
  std::cout << "  Use -benchmark to time the geometry APIs touched by the selected helper family on the same\n";
  std::cout << "  sampled cache, replayed sample, or manual edge case. Benchmark warmup passes are discarded so\n";
  std::cout << "  old/new comparisons run on hot cache for both versions.\n";
  std::cout << "  Manual edge cases are already explicit replays; use -manual_case_name, -manual_method, and\n";
  std::cout << "  -manual_topology to choose a specific solid, ray topology, and helper method to reproduce.\n";
  std::cout << "  The replayed sample set is determined by the pair (seed, stream_id): seed chooses the base\n";
  std::cout << "  deterministic sequence, while stream_id selects the logical deterministic sub-stream.\n";
  std::cout << "  Use -grazing_tolerance <value> to tilt the surface family's grazing direction along the\n";
  std::cout << "  surface normal, matching the legacy ShapeTester option. Default: 0 (exact grazing).\n\n";
  std::cout << "Options:\n";
  std::cout << "  -tier <fast|medium|slow> Sampling tier to run. Default: fast.\n";
  std::cout << "  -test_family <name>    Helper family to run. Default: contracts.\n";
  std::cout << "  -case_name <name|all>  Solid case to run. Default: all.\n";
  std::cout << "  -family <name|all>     Shape family to run. Default: all.\n";
  std::cout << "  -npoints <count>       Override the configured sample count for the selected tier.\n";
  std::cout << "  -seed <seed>           Override the configured base deterministic seed.\n";
  std::cout << "  -stream_id <id>        Override the logical deterministic sub-stream id paired with the seed.\n";
  std::cout << "  -replay_index <index>  Replay one sampled ray for the selected case.\n";
  std::cout << "  -benchmark             Run hot-cache timing instead of correctness validation.\n";
  std::cout << "  -benchmark_repetitions <n> Number of measured timing repetitions. Default: 9.\n";
  std::cout << "  -benchmark_warmup <n>  Number of discarded hot-cache warmup repetitions. Default: 3.\n";
  std::cout << "  -benchmark_target_calls <n> Target geometry calls per measured repetition. Default: 100000.\n";
  std::cout << "  -manual_case_name <n>  Curated manual edge-case name to run. Default: all.\n";
  std::cout << "  -manual_method <name>  Filter manual edge cases by helper method. Default: all.\n";
  std::cout << "  -manual_topology <t>   Filter manual edge cases by topology. Default: all.\n";
  std::cout << "  -grazing_tolerance <v> Tilt the surface family's grazing direction by v along the normal.\n";
  std::cout << "  -list_cases            Print the configured solid case names and exit.\n";
  std::cout << "                        Combine with -family <name> to print only that family.\n";
  std::cout << "  -list_families         Print the configured family names and exit.\n";
  std::cout << "  -list_test_families    Print the configured helper test families and exit.\n";
  std::cout << "  -list_manual_cases     Print the configured manual edge cases and exit.\n";
  std::cout << "  -help, --help, -h      Print this usage message and exit.\n\n";
  std::cout << "Examples:\n";
  std::cout << "  " << argv0 << " -tier fast -test_family contracts -family tube\n";
  std::cout << "  " << argv0 << " -tier fast -test_family normals -case_name box\n";
  std::cout << "  " << argv0 << " -tier fast -test_family surface -case_name box\n";
  std::cout << "  " << argv0 << " -tier fast -test_family distance_to_out -case_name box\n";
  std::cout << "  " << argv0 << " -tier fast -test_family distance_to_in -case_name box\n";
  std::cout << "  " << argv0 << " -tier fast -test_family safeties -case_name box\n";
  std::cout << "  " << argv0 << " -tier fast -test_family hit_consistency -case_name box\n";
  std::cout << "  " << argv0
            << " -test_family manual_edge_cases -case_name box -manual_method surface -manual_topology surface\n";
  std::cout << "  " << argv0
            << " -test_family manual_edge_cases -case_name box -manual_case_name box_inside_exit_positive_x\n";
  std::cout << "  " << argv0 << " -tier fast -test_family surface -case_name box -grazing_tolerance 1e-6\n";
  std::cout << "  " << argv0 << " -benchmark -tier fast -test_family surface -case_name cone_narrow_phi\n";
  std::cout << "  " << argv0
            << " -benchmark -tier slow -test_family surface -case_name cone_narrow_phi -seed 57 -stream_id 35 "
               "-npoints 10000000 -replay_index 4594302\n";
  std::cout << "  " << argv0 << " -tier medium -test_family contracts -case_name cone_section\n";
  std::cout << "  " << argv0
            << " -tier medium -test_family contracts -case_name cone_section -npoints 1000 -seed 42 -stream_id 7\n";
  std::cout << "  " << argv0 << " -tier slow -test_family all -case_name box\n";
  std::cout << "  " << argv0
            << " -tier medium -test_family contracts -case_name polycone_cms_like -seed 315 -stream_id 15 -npoints 360"
               " -replay_index 100\n\n";
  std::cout << "Configured helper test families:\n";
  std::cout << "  " << JoinConfiguredTestFamilyNames() << "\n\n";
  std::cout << "Configured solid families:\n";
  std::cout << "  " << JoinConfiguredFamilyNames() << "\n\n";
  std::cout << "Configured solid cases:\n";
  std::cout << "  " << JoinConfiguredCaseNames() << "\n";
}

bool ParseIntArgument(const char *option_name, const char *value, int &parsed)
{
  char *end            = nullptr;
  const long candidate = std::strtol(value, &end, 10);
  VECGEOM_VALIDATE(end != value && end != nullptr && *end == '\0',
                   << "Invalid integer value '" << value << "' for option " << option_name << ".");
  parsed = static_cast<int>(candidate);
  return true;
}

bool ParsePrecisionArgument(const char *option_name, const char *value, Precision &parsed)
{
  char *end              = nullptr;
  const double candidate = std::strtod(value, &end);
  VECGEOM_VALIDATE(end != value && end != nullptr && *end == '\0',
                   << "Invalid floating-point value '" << value << "' for option " << option_name << ".");
  parsed = static_cast<Precision>(candidate);
  return true;
}

// Parse every CLI option into one normalized structure so later execution code
// can focus on selection and validation rather than argument decoding.
ShapeContractOptions ParseOptions(int argc, char *argv[])
{
  ShapeContractOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "-help" || argument == "--help" || argument == "-h") {
      options.show_help = true;
      return options;
    }
    if (argument == "-list_cases") {
      options.list_cases    = true;
      options.used_defaults = false;
      continue;
    }
    if (argument == "-list_families") {
      options.list_families = true;
      options.used_defaults = false;
      continue;
    }
    if (argument == "-list_test_families") {
      options.list_test_families = true;
      options.used_defaults      = false;
      continue;
    }
    if (argument == "-list_manual_cases") {
      options.list_manual_cases = true;
      options.used_defaults     = false;
      continue;
    }
    if (argument == "-tier") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -tier.");
      options.tier_name     = argv[++i];
      options.used_defaults = false;
      continue;
    }
    if (argument == "-test_family") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -test_family.");
      options.test_family_name = argv[++i];
      options.used_defaults    = false;
      continue;
    }
    if (argument == "-case_name") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -case_name.");
      options.case_name     = argv[++i];
      options.used_defaults = false;
      continue;
    }
    if (argument == "-family") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -family.");
      options.family_name   = argv[++i];
      options.used_defaults = false;
      continue;
    }
    if (argument == "-npoints") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -npoints.");
      ParseIntArgument("-npoints", argv[++i], options.npoints);
      options.used_defaults = false;
      continue;
    }
    if (argument == "-seed") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -seed.");
      ParseIntArgument("-seed", argv[++i], options.seed);
      options.used_defaults = false;
      continue;
    }
    if (argument == "-stream_id") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -stream_id.");
      ParseIntArgument("-stream_id", argv[++i], options.stream_id);
      options.used_defaults = false;
      continue;
    }
    if (argument == "-replay_index") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -replay_index.");
      ParseIntArgument("-replay_index", argv[++i], options.replay_index);
      options.used_defaults = false;
      continue;
    }
    if (argument == "-benchmark") {
      options.benchmark_mode = true;
      options.used_defaults  = false;
      continue;
    }
    if (argument == "-benchmark_repetitions") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -benchmark_repetitions.");
      ParseIntArgument("-benchmark_repetitions", argv[++i], options.benchmark_repetitions);
      options.used_defaults = false;
      continue;
    }
    if (argument == "-benchmark_warmup") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -benchmark_warmup.");
      ParseIntArgument("-benchmark_warmup", argv[++i], options.benchmark_warmup);
      options.used_defaults = false;
      continue;
    }
    if (argument == "-benchmark_target_calls") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -benchmark_target_calls.");
      ParseIntArgument("-benchmark_target_calls", argv[++i], options.benchmark_target_calls);
      options.used_defaults = false;
      continue;
    }
    if (argument == "-manual_case_name") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -manual_case_name.");
      options.manual_case_name = argv[++i];
      options.used_defaults    = false;
      continue;
    }
    if (argument == "-manual_method") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -manual_method.");
      options.manual_method = argv[++i];
      options.used_defaults = false;
      continue;
    }
    if (argument == "-manual_topology") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -manual_topology.");
      options.manual_topology = argv[++i];
      options.used_defaults   = false;
      continue;
    }
    if (argument == "-grazing_tolerance") {
      VECGEOM_VALIDATE(i + 1 < argc, << "Missing value for option -grazing_tolerance.");
      ParsePrecisionArgument("-grazing_tolerance", argv[++i], options.grazing_tolerance);
      options.used_defaults = false;
      continue;
    }
    VECGEOM_VALIDATE(false, << "Unknown option '" << argument << "'. Run with -help for usage.");
  }
  return options;
}

void ValidateFamilySelection(const std::string &family_name)
{
  if (family_name == "all") return;
  VECGEOM_VALIDATE(!vecgeom::test::FindTestCaseSolidsByFamily(family_name).empty(),
                   << "Unknown shape test family '" << family_name << "'. Use -list_families to inspect the "
                   << "configured families.");
}

void PrintDefaultRunSummary(const ShapeContractOptions &options)
{
  if (!options.used_defaults) return;
  std::cout << "Running the fast contracts family for all configured solids. Use -help for usage, -tier medium "
               "for the heavier seeded tier, -tier slow for the nightly-style tier, -test_family normals, "
               "-test_family surface, "
               "-test_family distance_to_out, -test_family distance_to_in, -test_family safeties, "
               "-test_family hit_consistency, -test_family manual_edge_cases, -test_family all, or -benchmark for "
               "hot-cache timing on the selected helper workloads."
               " Use grouped family "
               "execution, -list_cases to inspect the configured cases, or -list_families to inspect the available "
               "families."
            << std::endl;
}

void ValidateManualMethodSelection(const std::string &manual_method)
{
  if (manual_method == "all") return;
  const auto parsed = ParseTestFamilySelection(manual_method);
  VECGEOM_VALIDATE(parsed != ShapeContractTestFamily::kManualEdgeCases && parsed != ShapeContractTestFamily::kAll,
                   << "Use -manual_method contracts, normals, surface, distance_to_out, distance_to_in, safeties, "
                   << "hit_consistency, or all.");
}

void ValidateManualTopologySelection(const std::string &manual_topology)
{
  VECGEOM_VALIDATE(manual_topology == "all" || manual_topology == "inside" || manual_topology == "surface" ||
                       manual_topology == "edge" || manual_topology == "outside",
                   << "Unknown manual topology '" << manual_topology
                   << "'. Use -manual_topology inside, surface, edge, outside, or all.");
}

bool UsesManualSelection(const ShapeContractOptions &options)
{
  return options.manual_case_name != "all" || options.manual_method != "all" || options.manual_topology != "all" ||
         options.list_manual_cases;
}

void ValidateManualOptionUsage(const ShapeContractOptions &options, ShapeContractTestFamily test_family)
{
  ValidateManualMethodSelection(options.manual_method);
  ValidateManualTopologySelection(options.manual_topology);
  VECGEOM_VALIDATE(!(UsesManualSelection(options) && !options.list_manual_cases &&
                     test_family != ShapeContractTestFamily::kManualEdgeCases),
                   << "Manual edge-case filters require -test_family manual_edge_cases or -list_manual_cases.");
  VECGEOM_VALIDATE(!(test_family == ShapeContractTestFamily::kManualEdgeCases && options.replay_index >= 0),
                   << "Manual edge cases are already explicit replays; rerun with -manual_case_name instead of "
                   << "-replay_index.");
  VECGEOM_VALIDATE(!(test_family == ShapeContractTestFamily::kManualEdgeCases &&
                     (options.npoints > 0 || options.seed >= 0 || options.stream_id >= 0)),
                   << "Manual edge cases do not use sampled npoints/seed/stream_id overrides.");
}

bool MatchesManualCaseSolidSelection(const vecgeom::test::ManualEdgeCase &manual_case,
                                     const ShapeContractOptions &options)
{
  if (options.case_name != "all") return options.case_name == manual_case.solid_case_name;
  if (options.family_name == "all") return true;
  return vecgeom::test::GetTestCaseFamilyName(manual_case.solid_case_name) == options.family_name;
}

bool MatchesManualCaseFilters(const vecgeom::test::ManualEdgeCase &manual_case, const ShapeContractOptions &options)
{
  if (!MatchesManualCaseSolidSelection(manual_case, options)) return false;
  if (options.manual_case_name != "all" && options.manual_case_name != manual_case.name) return false;
  if (options.manual_method != "all" && options.manual_method != manual_case.target_family_name) return false;
  if (options.manual_topology != "all" &&
      options.manual_topology != vecgeom::test::ShapeSampleCategoryLabel(manual_case.topology)) {
    return false;
  }
  return true;
}

// Reuse the same manual-case filtering logic for both `-list_manual_cases` and
// actual `manual_edge_cases` execution so the two views stay consistent.
std::vector<const vecgeom::test::ManualEdgeCase *> ResolveSelectedManualEdgeCases(const ShapeContractOptions &options)
{
  std::vector<const vecgeom::test::ManualEdgeCase *> matches;
  for (auto const &manual_case : vecgeom::test::GetManualEdgeCases()) {
    if (MatchesManualCaseFilters(manual_case, options)) matches.push_back(&manual_case);
  }
  return matches;
}

std::string DescribeManualEdgeCase(const vecgeom::test::ManualEdgeCase &manual_case)
{
  std::ostringstream out;
  out << manual_case.name << " solid=" << manual_case.solid_case_name << " method=" << manual_case.target_family_name
      << " topology=" << vecgeom::test::ShapeSampleCategoryLabel(manual_case.topology)
      << " point=" << vecgeom::test::FormatVec(manual_case.point);
  if (manual_case.uses_target_point) {
    out << " target_point=" << vecgeom::test::FormatVec(manual_case.target_point)
        << " direction=" << vecgeom::test::FormatVec(vecgeom::test::EffectiveManualEdgeCaseDirection(manual_case));
  } else {
    out << " direction=" << vecgeom::test::FormatVec(manual_case.direction);
  }
  if (std::string(manual_case.target_family_name) == "surface") {
    out << " grazing_tolerance=" << manual_case.grazing_tolerance;
  }
  out << " description=\"" << manual_case.description << "\"";
  return out.str();
}

void PrintManualEdgeCases(const ShapeContractOptions &options)
{
  auto selected_cases = ResolveSelectedManualEdgeCases(options);
  VECGEOM_VALIDATE(!selected_cases.empty(),
                   << "No manual edge cases matched the requested filters. Use -list_manual_cases without filters to "
                   << "inspect the configured manual cases.");
  for (auto const *manual_case : selected_cases) {
    std::cout << DescribeManualEdgeCase(*manual_case) << std::endl;
  }
}

vecgeom::test::ShapeSamplingConfig ResolveSamplingConfig(const vecgeom::test::TestCaseSolid &solid_case,
                                                         ShapeContractTier tier, const ShapeContractOptions &options)
{
  auto config = solid_case.fast_contract_sampling;
  if (tier == ShapeContractTier::kMedium) {
    if (solid_case.use_independent_medium_sampling) {
      config = solid_case.medium_contract_sampling;
    } else {
      // Keep fast and medium on the same deterministic sample stream unless a
      // solid explicitly opts into an independent medium profile. Medium then
      // differs only by statistics, which makes fast failures naturally
      // reproducible inside the larger medium sample set. Guard that policy so
      // registry data cannot silently drift away from the effective behavior.
      VECGEOM_VALIDATE(vecgeom::test::SamplingConfigsMatchExceptMaxPoints(solid_case.fast_contract_sampling,
                                                                          solid_case.medium_contract_sampling),
                       << "Medium sampling for solid '" << solid_case.name
                       << "' must match the fast profile except for max_points unless "
                          "use_independent_medium_sampling is set.");
      config.max_points = solid_case.medium_contract_sampling.max_points;
    }
  } else if (tier == ShapeContractTier::kSlow) {
    if (solid_case.use_independent_slow_sampling) {
      config = solid_case.slow_contract_sampling;
    } else {
      // Slow follows the same rule as medium: reuse the fast profile unless a
      // solid explicitly needs a dedicated slow-only configuration. By default
      // it only raises statistics to make nightly runs a strict superset of
      // the fast and medium sampled rays.
      config.max_points = 1000000;
    }
  }
  if (options.npoints > 0) config.max_points = options.npoints;
  if (options.seed >= 0) config.seed = static_cast<unsigned long>(options.seed);
  if (options.stream_id >= 0) config.stream_id = static_cast<unsigned long>(options.stream_id);
  return config;
}

Precision ResolveGrazingTolerance(const ShapeContractOptions &options)
{
  return options.grazing_tolerance >= static_cast<Precision>(0.) ? options.grazing_tolerance
                                                                 : static_cast<Precision>(0.);
}

Precision ResolveSolidTolerance(const vecgeom::test::TestCaseSolid &solid_case)
{
  return solid_case.solid_tolerance >= vecgeom::kTolerance ? solid_case.solid_tolerance : vecgeom::kTolerance;
}

void ValidateBenchmarkOptions(const ShapeContractOptions &options)
{
  VECGEOM_VALIDATE(options.benchmark_repetitions > 0, << "Use -benchmark_repetitions with a positive integer.");
  VECGEOM_VALIDATE(options.benchmark_warmup >= 0, << "Use -benchmark_warmup with a non-negative integer.");
  VECGEOM_VALIDATE(options.benchmark_target_calls > 0, << "Use -benchmark_target_calls with a positive integer.");
}

int ViolationDisplayLimit(ShapeContractTier tier)
{
  if (tier == ShapeContractTier::kFast) return 1;
  if (tier == ShapeContractTier::kMedium) return 2;
  return 3;
}

auto MakeDistanceToOutCaller()
{
  return [](vecgeom::VPlacedVolume const *shape, const Vec_t &point, const Vec_t &direction, Vec_t &normal) {
    // The helper interface expects a DistanceToOut callback that can also
    // provide an exit normal when one is available. Reconstruct that normal
    // from the sampled exit point so the same callback works for every
    // configured placed solid used by the contract tests.
    Precision distance = shape->DistanceToOut(point, direction);
    if (distance >= 0. && distance < vecgeom::kInfLength) {
      const Vec_t hit_point = point + distance * direction;
      shape->Normal(hit_point, normal);
    } else {
      normal = Vec_t(0., 0., 0.);
    }
    return distance;
  };
}

// Build the shared failure footer that points developers straight at a
// copy-pasteable replay command and the geometry functions involved.
std::string MakeHowToDebugHint(const std::string &executable_path, const std::string &case_name,
                               const vecgeom::test::ShapeCheckResult &result,
                               const vecgeom::test::ShapeSamplingConfig &config,
                               const vecgeom::test::TestCaseSolid &solid_case, ShapeContractTier tier,
                               ShapeContractTestFamily test_family, Precision grazing_tolerance)
{
  if (result.Violations().empty()) return "";
  auto const &first_violation = result.Violations().front();
  if (first_violation.displayed_occurrences.empty()) return "";

  auto const &first_occurrence = first_violation.displayed_occurrences.front();
  auto const &context          = first_occurrence.context;
  std::ostringstream out;
  out << "\nHow to debug:\n";
  out << "  first_recorded_occurrence: " << vecgeom::test::FormatShapeCheckContext(context) << "\n";
  out << "  replay_command: " << executable_path << " -tier " << TierLabel(tier) << " -test_family "
      << TestFamilyLabel(test_family) << " -case_name " << case_name << " -seed " << config.seed << " -stream_id "
      << config.stream_id << " -npoints " << config.max_points;
  if (test_family == ShapeContractTestFamily::kSurface && grazing_tolerance > static_cast<Precision>(0.)) {
    out << " -grazing_tolerance " << grazing_tolerance;
  }
  out << " -replay_index " << context.sample_index << "\n";
  out << "  replay_function: " << vecgeom::test::ShapeContractReplayFunctionName(context) << "\n";
  out << "  evaluator_function: " << vecgeom::test::ShapeContractEvaluatorFunctionName(context) << "\n";
  out << "  geometry_function: " << vecgeom::test::ShapeContractGeometryFunctionName(context) << "\n";
  const auto implementation_function =
      vecgeom::test::ShapeContractImplementationFunctionName(solid_case.implementation_debug_type, context);
  if (!implementation_function.empty()) {
    out << "  implementation_function: " << implementation_function << "\n";
  }
  const char *support = vecgeom::test::ShapeContractSupportFunctionName(context);
  if (support[0] != '\0') out << "  support_function: " << support << "\n";
  return out.str();
}

std::uint64_t MakeSampleFingerprint(const vecgeom::test::ShapeSampleSet &samples)
{
  auto mix_hash = [](std::uint64_t state, std::uint64_t value) {
    constexpr std::uint64_t kFnvPrime = 1099511628211ull;
    return (state ^ value) * kFnvPrime;
  };

  auto hash_scalar = [&mix_hash](std::uint64_t state, Precision value) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(value));
    return mix_hash(state, bits);
  };

  // Persist a compact sample fingerprint so failures can report the exact
  // generated sample set without dumping every sampled point and direction.
  std::uint64_t fingerprint = 1469598103934665603ull;
  fingerprint               = mix_hash(fingerprint, static_cast<std::uint64_t>(samples.TotalPoints()));
  fingerprint               = mix_hash(fingerprint, static_cast<std::uint64_t>(samples.max_points_inside));
  fingerprint               = mix_hash(fingerprint, static_cast<std::uint64_t>(samples.max_points_surface));
  fingerprint               = mix_hash(fingerprint, static_cast<std::uint64_t>(samples.max_points_outside));
  for (int i = 0; i < samples.TotalPoints(); ++i) {
    fingerprint = hash_scalar(fingerprint, samples.points[i].x());
    fingerprint = hash_scalar(fingerprint, samples.points[i].y());
    fingerprint = hash_scalar(fingerprint, samples.points[i].z());
    fingerprint = hash_scalar(fingerprint, samples.directions[i].x());
    fingerprint = hash_scalar(fingerprint, samples.directions[i].y());
    fingerprint = hash_scalar(fingerprint, samples.directions[i].z());
  }
  return fingerprint;
}

// Cache one sampled solid run in memory so grouped family execution can reuse
// the same generated rays instead of resampling independently per family.
class ShapeContractExecutionCache {
public:
  ShapeContractExecutionCache(const vecgeom::test::TestCaseSolid &solid_case, ShapeContractTier tier,
                              const ShapeContractOptions &options)
      : fSolidCase(solid_case), fTier(tier), fConfig(ResolveSamplingConfig(solid_case, tier, options)),
        fSolidTolerance(ResolveSolidTolerance(solid_case)), fGrazingTolerance(ResolveGrazingTolerance(options)),
        fShape(solid_case.make_shape())
  {
  }

  const vecgeom::test::TestCaseSolid &SolidCase() const { return fSolidCase; }
  ShapeContractTier Tier() const { return fTier; }
  const vecgeom::test::ShapeSamplingConfig &Config() const { return fConfig; }
  Precision SolidTolerance() const { return fSolidTolerance; }
  Precision GrazingTolerance() const { return fGrazingTolerance; }
  vecgeom::VPlacedVolume const *Shape() const { return fShape.get(); }
  std::uint64_t Fingerprint()
  {
    EnsureSamples();
    return fFingerprint;
  }

  const vecgeom::test::ShapeSampleSet &Samples()
  {
    EnsureSamples();
    return fSamples;
  }

  const ShapeContractOutcome &ContractOutcome()
  {
    if (!fContractOutcomeReady) {
      auto view = vecgeom::test::MakeShapeContractSampleView(Samples());
      vecgeom::test::ShapeContractViolationSink sink(fContractOutcome.result, ViolationDisplayLimit(fTier));
      fContractOutcome.summary = vecgeom::test::RunShapeConventionChecks(fShape.get(), view, fSolidTolerance,
                                                                         MakeDistanceToOutCaller(), sink);
      fContractOutcomeReady    = true;
    }
    return fContractOutcome;
  }

  const ShapeNormalOutcome &NormalOutcome()
  {
    if (!fNormalOutcomeReady) {
      auto view = vecgeom::test::MakeShapeContractSampleView(Samples());
      vecgeom::test::ShapeContractViolationSink sink(fNormalOutcome.result, ViolationDisplayLimit(fTier));
      fNormalOutcome.summary =
          vecgeom::test::RunShapeNormalChecks(fShape.get(), view, fSolidTolerance, MakeDistanceToOutCaller(), sink);
      fNormalOutcomeReady = true;
    }
    return fNormalOutcome;
  }

  const ShapeSurfaceOutcome &SurfaceOutcome()
  {
    if (!fSurfaceOutcomeReady) {
      auto view = vecgeom::test::MakeShapeContractSampleView(Samples());
      vecgeom::test::ShapeContractViolationSink sink(fSurfaceOutcome.result, ViolationDisplayLimit(fTier));
      fSurfaceOutcome.summary = vecgeom::test::RunShapeSurfaceChecks(
          fShape.get(), view, fSolidTolerance, fGrazingTolerance, MakeDistanceToOutCaller(), sink);
      fSurfaceOutcomeReady = true;
    }
    return fSurfaceOutcome;
  }

  const ShapeDistanceToOutOutcome &DistanceToOutOutcome()
  {
    if (!fDistanceToOutOutcomeReady) {
      auto view = vecgeom::test::MakeShapeContractSampleView(Samples());
      vecgeom::test::ShapeContractViolationSink sink(fDistanceToOutOutcome.result, ViolationDisplayLimit(fTier));
      fDistanceToOutOutcome.summary = vecgeom::test::RunShapeDistanceToOutChecks(
          fShape.get(), view, fSolidTolerance, MakeDistanceToOutCaller(), sink);
      fDistanceToOutOutcomeReady = true;
    }
    return fDistanceToOutOutcome;
  }

  const ShapeDistanceToInOutcome &DistanceToInOutcome()
  {
    if (!fDistanceToInOutcomeReady) {
      auto view = vecgeom::test::MakeShapeContractSampleView(Samples());
      vecgeom::test::ShapeContractViolationSink sink(fDistanceToInOutcome.result, ViolationDisplayLimit(fTier));
      fDistanceToInOutcome.summary =
          vecgeom::test::RunShapeDistanceToInChecks(fShape.get(), view, fSolidTolerance, sink);
      fDistanceToInOutcomeReady = true;
    }
    return fDistanceToInOutcome;
  }

  const ShapeSafetyOutcome &SafetyOutcome()
  {
    if (!fSafetyOutcomeReady) {
      auto view = vecgeom::test::MakeShapeContractSampleView(Samples());
      vecgeom::test::ShapeContractViolationSink sink(fSafetyOutcome.result, ViolationDisplayLimit(fTier));
      fSafetyOutcome.summary =
          vecgeom::test::RunShapeSafetyChecks(fShape.get(), view, fSolidTolerance, MakeDistanceToOutCaller(), sink);
      fSafetyOutcomeReady = true;
    }
    return fSafetyOutcome;
  }

  const ShapeHitConsistencyOutcome &HitConsistencyOutcome()
  {
    if (!fHitConsistencyOutcomeReady) {
      auto view = vecgeom::test::MakeShapeContractSampleView(Samples());
      vecgeom::test::ShapeContractViolationSink sink(fHitConsistencyOutcome.result, ViolationDisplayLimit(fTier));
      fHitConsistencyOutcome.summary = vecgeom::test::RunShapeHitConsistencyChecks(
          fShape.get(), view, fSolidTolerance, MakeDistanceToOutCaller(), sink);
      fHitConsistencyOutcomeReady = true;
    }
    return fHitConsistencyOutcome;
  }

private:
  void EnsureSamples()
  {
    if (fSamplesReady) return;
    vecgeom::RNG rng;
    vecgeom::test::ShapeSampler sampler(rng);
    // Generate the sampled inside/surface/outside cache once, then let every
    // requested family derive its own views and summaries from the same data.
    fSamples      = sampler.Generate(fShape.get(), fConfig);
    fFingerprint  = MakeSampleFingerprint(fSamples);
    fSamplesReady = true;
  }

  const vecgeom::test::TestCaseSolid &fSolidCase;
  ShapeContractTier fTier;
  vecgeom::test::ShapeSamplingConfig fConfig;
  Precision fSolidTolerance   = vecgeom::kTolerance;
  Precision fGrazingTolerance = 0.;
  std::unique_ptr<vecgeom::VPlacedVolume> fShape;
  vecgeom::test::ShapeSampleSet fSamples;
  std::uint64_t fFingerprint = 0;
  ShapeContractOutcome fContractOutcome;
  ShapeNormalOutcome fNormalOutcome;
  ShapeSurfaceOutcome fSurfaceOutcome;
  ShapeDistanceToOutOutcome fDistanceToOutOutcome;
  ShapeDistanceToInOutcome fDistanceToInOutcome;
  ShapeSafetyOutcome fSafetyOutcome;
  ShapeHitConsistencyOutcome fHitConsistencyOutcome;
  bool fSamplesReady               = false;
  bool fContractOutcomeReady       = false;
  bool fNormalOutcomeReady         = false;
  bool fSurfaceOutcomeReady        = false;
  bool fDistanceToOutOutcomeReady  = false;
  bool fDistanceToInOutcomeReady   = false;
  bool fSafetyOutcomeReady         = false;
  bool fHitConsistencyOutcomeReady = false;
};

struct ShapeBenchmarkSlice {
  const char *label = "";
  int offset        = 0;
  int count         = 0;
};

ShapeBenchmarkSlice MakeAllBenchmarkSlice(const vecgeom::test::ShapeContractSampleView &samples)
{
  return {"all_samples", 0, samples.TotalPoints()};
}

ShapeBenchmarkSlice MakeInsideBenchmarkSlice(const vecgeom::test::ShapeContractSampleView &samples)
{
  return {"inside_samples", samples.offset_inside, samples.max_points_inside};
}

ShapeBenchmarkSlice MakeSurfaceEdgeBenchmarkSlice(const vecgeom::test::ShapeContractSampleView &samples)
{
  return {"surface_edge_samples", samples.offset_surface, samples.max_points_surface + samples.max_points_edge};
}

ShapeBenchmarkSlice MakeOutsideBenchmarkSlice(const vecgeom::test::ShapeContractSampleView &samples)
{
  return {"outside_samples", samples.offset_outside, samples.max_points_outside};
}

ShapeBenchmarkSlice MakeSingleBenchmarkSlice(int sample_index, const char *label) { return {label, sample_index, 1}; }

const char *ShapeBenchmarkOperationLabel(ShapeBenchmarkOperation operation)
{
  switch (operation) {
  case ShapeBenchmarkOperation::kInside:
    return "Inside";
  case ShapeBenchmarkOperation::kNormal:
    return "Normal";
  case ShapeBenchmarkOperation::kDistanceToIn:
    return "DistanceToIn";
  case ShapeBenchmarkOperation::kDistanceToOut:
    return "DistanceToOut";
  case ShapeBenchmarkOperation::kSafetyToIn:
    return "SafetyToIn";
  case ShapeBenchmarkOperation::kSafetyToOut:
    return "SafetyToOut";
  }
  return "Inside";
}

std::string MakeBenchmarkWorkloadLabel(ShapeBenchmarkOperation operation, const ShapeBenchmarkSlice &slice)
{
  std::ostringstream out;
  out << ShapeBenchmarkOperationLabel(operation) << "(" << slice.label << ")";
  return out.str();
}

void AppendBenchmarkWorkload(std::vector<ShapeBenchmarkWorkload> &workloads, ShapeBenchmarkOperation operation,
                             const ShapeBenchmarkSlice &slice)
{
  if (slice.count <= 0) return;
  const auto label = MakeBenchmarkWorkloadLabel(operation, slice);
  for (auto const &existing : workloads) {
    if (existing.label == label && existing.offset == slice.offset && existing.count == slice.count &&
        existing.operation == operation) {
      return;
    }
  }
  workloads.push_back({label, operation, slice.offset, slice.count});
}

std::vector<ShapeBenchmarkWorkload> BuildBenchmarkWorkloadsForFamily(
    const vecgeom::test::ShapeContractSampleView &samples, ShapeContractTestFamily test_family)
{
  std::vector<ShapeBenchmarkWorkload> workloads;
  const auto all_samples     = MakeAllBenchmarkSlice(samples);
  const auto inside_samples  = MakeInsideBenchmarkSlice(samples);
  const auto surface_samples = MakeSurfaceEdgeBenchmarkSlice(samples);
  const auto outside_samples = MakeOutsideBenchmarkSlice(samples);

  auto add_contracts = [&]() {
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kInside, all_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToIn, surface_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToOut, surface_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToIn, surface_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToOut, surface_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToOut, inside_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToOut, inside_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToIn, outside_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToIn, outside_samples);
  };

  switch (test_family) {
  case ShapeContractTestFamily::kContracts:
    add_contracts();
    break;
  case ShapeContractTestFamily::kNormals:
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kNormal, surface_samples);
    break;
  case ShapeContractTestFamily::kSurface:
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kNormal, surface_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToIn, surface_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToOut, surface_samples);
    break;
  case ShapeContractTestFamily::kDistanceToOut:
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToOut, inside_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToOut, inside_samples);
    break;
  case ShapeContractTestFamily::kDistanceToIn:
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToIn, outside_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToIn, outside_samples);
    break;
  case ShapeContractTestFamily::kSafeties:
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToOut, inside_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToIn, outside_samples);
    break;
  case ShapeContractTestFamily::kHitConsistency:
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kNormal, surface_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToOut, inside_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToOut, inside_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToIn, outside_samples);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToIn, outside_samples);
    break;
  case ShapeContractTestFamily::kAll:
    add_contracts();
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kNormal, surface_samples);
    break;
  case ShapeContractTestFamily::kManualEdgeCases:
    VECGEOM_VALIDATE(false, << "manual_edge_cases should be benchmarked through their selected target family.");
    break;
  }
  return workloads;
}

std::vector<ShapeBenchmarkWorkload> BuildReplayBenchmarkWorkloads(ShapeContractTestFamily test_family, int sample_index)
{
  std::vector<ShapeBenchmarkWorkload> workloads;
  const auto replay_sample = MakeSingleBenchmarkSlice(sample_index, "replay_sample");
  switch (test_family) {
  case ShapeContractTestFamily::kContracts:
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kInside, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToIn, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToOut, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToIn, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToOut, replay_sample);
    break;
  case ShapeContractTestFamily::kNormals:
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kNormal, replay_sample);
    break;
  case ShapeContractTestFamily::kSurface:
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kNormal, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToIn, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToOut, replay_sample);
    break;
  case ShapeContractTestFamily::kDistanceToOut:
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToOut, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToOut, replay_sample);
    break;
  case ShapeContractTestFamily::kDistanceToIn:
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToIn, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToIn, replay_sample);
    break;
  case ShapeContractTestFamily::kSafeties:
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToIn, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToOut, replay_sample);
    break;
  case ShapeContractTestFamily::kHitConsistency:
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kNormal, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToIn, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToOut, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToIn, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToOut, replay_sample);
    break;
  case ShapeContractTestFamily::kAll:
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kInside, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kNormal, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToIn, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kDistanceToOut, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToIn, replay_sample);
    AppendBenchmarkWorkload(workloads, ShapeBenchmarkOperation::kSafetyToOut, replay_sample);
    break;
  case ShapeContractTestFamily::kManualEdgeCases:
    VECGEOM_VALIDATE(false, << "manual_edge_cases do not use -replay_index.");
    break;
  }
  return workloads;
}

Precision SanitizeBenchmarkValue(Precision value)
{
  if (!std::isfinite(static_cast<double>(value)) || value >= vecgeom::kInfLength * static_cast<Precision>(0.5)) {
    return static_cast<Precision>(0.);
  }
  return value;
}

Precision ExecuteBenchmarkWorkload(vecgeom::VPlacedVolume const *shape,
                                   const vecgeom::test::ShapeContractSampleView &view,
                                   const ShapeBenchmarkWorkload &workload)
{
  Precision accumulator = 0.;
  for (int i = 0; i < workload.count; ++i) {
    const int sample_index = workload.offset + i;
    const Vec_t &point     = view.Point(sample_index);
    const Vec_t &direction = view.Direction(sample_index);
    switch (workload.operation) {
    case ShapeBenchmarkOperation::kInside:
      accumulator += static_cast<Precision>(static_cast<int>(shape->Inside(point)));
      break;
    case ShapeBenchmarkOperation::kNormal: {
      Vec_t normal(0., 0., 0.);
      const bool valid = shape->Normal(point, normal);
      accumulator += valid ? static_cast<Precision>(1.) : static_cast<Precision>(0.);
      accumulator += SanitizeBenchmarkValue(normal.x());
      accumulator += SanitizeBenchmarkValue(normal.y());
      accumulator += SanitizeBenchmarkValue(normal.z());
      break;
    }
    case ShapeBenchmarkOperation::kDistanceToIn:
      accumulator += SanitizeBenchmarkValue(shape->DistanceToIn(point, direction));
      break;
    case ShapeBenchmarkOperation::kDistanceToOut:
      accumulator += SanitizeBenchmarkValue(shape->DistanceToOut(point, direction));
      break;
    case ShapeBenchmarkOperation::kSafetyToIn:
      accumulator += SanitizeBenchmarkValue(shape->SafetyToIn(point));
      break;
    case ShapeBenchmarkOperation::kSafetyToOut:
      accumulator += SanitizeBenchmarkValue(shape->SafetyToOut(point));
      break;
    }
  }
  return accumulator;
}

volatile Precision gShapeBenchmarkSink = 0.;

void ConsumeBenchmarkAccumulator(Precision value) { gShapeBenchmarkSink += value; }

ShapeBenchmarkSummary RunBenchmarkWorkload(vecgeom::VPlacedVolume const *shape,
                                           const vecgeom::test::ShapeContractSampleView &view,
                                           const ShapeBenchmarkWorkload &workload, const ShapeContractOptions &options)
{
  ShapeBenchmarkSummary summary;
  summary.label                = workload.label;
  summary.samples              = workload.count;
  summary.warmup_repetitions   = options.benchmark_warmup;
  summary.measured_repetitions = options.benchmark_repetitions;
  summary.loops_per_repeat     = std::max(1, (options.benchmark_target_calls + workload.count - 1) / workload.count);
  summary.calls_per_repeat =
      static_cast<std::uint64_t>(summary.samples) * static_cast<std::uint64_t>(summary.loops_per_repeat);

  auto run_once = [&]() {
    Precision local_accumulator = 0.;
    for (int loop = 0; loop < summary.loops_per_repeat; ++loop) {
      local_accumulator += ExecuteBenchmarkWorkload(shape, view, workload);
    }
    ConsumeBenchmarkAccumulator(local_accumulator);
  };

  for (int i = 0; i < options.benchmark_warmup; ++i) {
    run_once();
  }

  std::vector<Precision> timings;
  timings.reserve(options.benchmark_repetitions);
  for (int i = 0; i < options.benchmark_repetitions; ++i) {
    vecgeom::Stopwatch timer;
    timer.Start();
    run_once();
    timings.push_back(timer.Stop());
  }

  summary.min_seconds = timings.empty() ? 0. : *std::min_element(timings.begin(), timings.end());
  summary.max_seconds = timings.empty() ? 0. : *std::max_element(timings.begin(), timings.end());

  Precision sum = 0.;
  for (auto elapsed : timings)
    sum += elapsed;
  summary.mean_seconds = timings.empty() ? 0. : sum / static_cast<Precision>(timings.size());

  Precision variance = 0.;
  for (auto elapsed : timings) {
    const Precision centered = elapsed - summary.mean_seconds;
    variance += centered * centered;
  }
  if (timings.size() > 1) {
    variance /= static_cast<Precision>(timings.size() - 1);
  } else {
    variance = 0.;
  }
  summary.stddev_seconds = std::sqrt(variance);
  if (summary.calls_per_repeat > 0) {
    summary.ns_per_call =
        summary.mean_seconds * static_cast<Precision>(1.0e9) / static_cast<Precision>(summary.calls_per_repeat);
  }
  if (summary.mean_seconds > 0.) {
    summary.rel_sigma_percent = static_cast<Precision>(100.) * summary.stddev_seconds / summary.mean_seconds;
  }
  return summary;
}

void PrintBenchmarkSummaryHeader(const std::string &label, ShapeContractTier tier, ShapeContractTestFamily test_family,
                                 const ShapeContractOptions &options, std::uint64_t fingerprint)
{
  std::cout << "Benchmark " << label << " tier=" << TierLabel(tier) << " test_family=" << TestFamilyLabel(test_family)
            << " hot_cache=true warmup=" << options.benchmark_warmup << " repetitions=" << options.benchmark_repetitions
            << " target_calls=" << options.benchmark_target_calls << " fingerprint=" << fingerprint << std::endl;
}

void PrintBenchmarkSummaryLine(const ShapeBenchmarkSummary &summary)
{
  std::cout << "  workload=" << summary.label << " samples=" << summary.samples
            << " loops_per_repeat=" << summary.loops_per_repeat << " calls_per_repeat=" << summary.calls_per_repeat
            << " mean_s=" << summary.mean_seconds << " stddev_s=" << summary.stddev_seconds
            << " rel_sigma_pct=" << summary.rel_sigma_percent << " min_s=" << summary.min_seconds
            << " max_s=" << summary.max_seconds << " ns_per_call=" << summary.ns_per_call << std::endl;
}

void BenchmarkWorkloadsForView(vecgeom::VPlacedVolume const *shape, const vecgeom::test::ShapeContractSampleView &view,
                               const std::vector<ShapeBenchmarkWorkload> &workloads,
                               const ShapeContractOptions &options)
{
  VECGEOM_VALIDATE(!workloads.empty(), << "No benchmark workloads matched the selected helper family and sample set.");
  for (auto const &workload : workloads) {
    PrintBenchmarkSummaryLine(RunBenchmarkWorkload(shape, view, workload, options));
  }
}

void BenchmarkSelectedFamilies(const vecgeom::test::TestCaseSolid &solid_case, const ShapeContractOptions &options,
                               ShapeContractTier tier, ShapeContractTestFamily test_family)
{
  ShapeContractExecutionCache cache(solid_case, tier, options);
  auto view      = vecgeom::test::MakeShapeContractSampleView(cache.Samples());
  auto workloads = BuildBenchmarkWorkloadsForFamily(view, test_family);
  std::ostringstream label;
  label << "case='" << solid_case.name << "'";
  PrintBenchmarkSummaryHeader(label.str(), tier, test_family, options, cache.Fingerprint());
  BenchmarkWorkloadsForView(cache.Shape(), view, workloads, options);
}

void BenchmarkReplaySample(const vecgeom::test::TestCaseSolid &solid_case, const ShapeContractOptions &options,
                           ShapeContractTier tier, ShapeContractTestFamily test_family)
{
  ShapeContractExecutionCache cache(solid_case, tier, options);
  auto view = vecgeom::test::MakeShapeContractSampleView(cache.Samples());
  VECGEOM_VALIDATE(options.replay_index >= 0 && options.replay_index < view.TotalPoints(),
                   << "Replay index " << options.replay_index << " is outside [0, " << view.TotalPoints()
                   << ") for solid '" << cache.SolidCase().name << "'.");
  const auto context = vecgeom::test::MakeShapeCheckContext(view, options.replay_index);
  auto workloads     = BuildReplayBenchmarkWorkloads(test_family, options.replay_index);
  std::ostringstream label;
  label << "replay case='" << solid_case.name << "' sample_index=" << options.replay_index
        << " category=" << vecgeom::test::ShapeSampleCategoryLabel(context.sample_group);
  PrintBenchmarkSummaryHeader(label.str(), tier, test_family, options, cache.Fingerprint());
  BenchmarkWorkloadsForView(cache.Shape(), view, workloads, options);
}

bool ManualEdgeCaseNeedsTargetPoint(const vecgeom::test::ManualEdgeCase &manual_case)
{
  return manual_case.uses_target_point;
}

vecgeom::test::ShapeSampleSet MakeManualEdgeCaseSamples(const vecgeom::test::ManualEdgeCase &manual_case)
{
  vecgeom::test::ShapeSampleSet samples;
  const bool needs_target_point = ManualEdgeCaseNeedsTargetPoint(manual_case);

  // Manual cases bypass random sampling entirely. Build the minimal synthetic
  // sample set that matches the requested topology bucket and helper family.
  samples.max_points_inside =
      needs_target_point || manual_case.topology == vecgeom::test::ShapeSampleCategory::kInside ? 1 : 0;
  samples.max_points_surface = manual_case.topology == vecgeom::test::ShapeSampleCategory::kSurface ? 1 : 0;
  samples.max_points_edge    = manual_case.topology == vecgeom::test::ShapeSampleCategory::kEdge ? 1 : 0;
  samples.max_points_outside = manual_case.topology == vecgeom::test::ShapeSampleCategory::kOutside ? 1 : 0;

  samples.offset_inside  = 0;
  samples.offset_surface = samples.offset_inside + samples.max_points_inside;
  samples.offset_edge    = samples.offset_surface + samples.max_points_surface;
  samples.offset_outside = samples.offset_edge + samples.max_points_edge;

  const int total_points =
      samples.max_points_inside + samples.max_points_surface + samples.max_points_edge + samples.max_points_outside;
  samples.points.reserve(total_points);
  samples.directions.reserve(total_points);

  const Vec_t effective_direction = vecgeom::test::EffectiveManualEdgeCaseDirection(manual_case);
  if (needs_target_point || manual_case.topology == vecgeom::test::ShapeSampleCategory::kInside) {
    const Vec_t inside_point = needs_target_point ? manual_case.target_point : manual_case.point;
    samples.points.push_back(inside_point);
    samples.directions.push_back(manual_case.direction);
  }
  if (manual_case.topology == vecgeom::test::ShapeSampleCategory::kSurface) {
    samples.points.push_back(manual_case.point);
    samples.directions.push_back(effective_direction);
  }
  if (manual_case.topology == vecgeom::test::ShapeSampleCategory::kEdge) {
    samples.points.push_back(manual_case.point);
    samples.directions.push_back(effective_direction);
  }
  if (manual_case.topology == vecgeom::test::ShapeSampleCategory::kOutside) {
    samples.points.push_back(manual_case.point);
    samples.directions.push_back(effective_direction);
  }
  return samples;
}

int ManualEdgeCasePrimarySampleIndex(const vecgeom::test::ShapeSampleSet &samples,
                                     const vecgeom::test::ManualEdgeCase &manual_case)
{
  switch (manual_case.topology) {
  case vecgeom::test::ShapeSampleCategory::kInside:
    return samples.offset_inside;
  case vecgeom::test::ShapeSampleCategory::kSurface:
    return samples.offset_surface;
  case vecgeom::test::ShapeSampleCategory::kEdge:
    return samples.offset_edge;
  case vecgeom::test::ShapeSampleCategory::kOutside:
    return samples.offset_outside;
  default:
    return -1;
  }
}

Precision ResolveManualGrazingTolerance(const ShapeContractOptions &options,
                                        const vecgeom::test::ManualEdgeCase &manual_case)
{
  return options.grazing_tolerance >= static_cast<Precision>(0.) ? options.grazing_tolerance
                                                                 : manual_case.grazing_tolerance;
}

std::string MakeManualEdgeCaseDebugHint(const std::string &executable_path, ShapeContractTier tier,
                                        const vecgeom::test::ManualEdgeCase &manual_case, Precision grazing_tolerance)
{
  std::ostringstream out;
  out << "\nHow to debug:\n";
  out << "  rerun_command: " << executable_path << " -tier " << TierLabel(tier)
      << " -test_family manual_edge_cases -case_name " << manual_case.solid_case_name << " -manual_case_name "
      << manual_case.name;
  if (std::string(manual_case.target_family_name) == "surface" && grazing_tolerance > static_cast<Precision>(0.)) {
    out << " -grazing_tolerance " << grazing_tolerance;
  }
  out << "\n";
  return out.str();
}

template <typename ReplayT>
bool ManualReplayPassed(const ReplayT &replay)
{
  return replay.Passed();
}

void ValidateManualEdgeCase(const vecgeom::test::ManualEdgeCase &manual_case, const ShapeContractOptions &options,
                            const std::string &executable_path, ShapeContractTier tier, bool verbose_on_success)
{
  auto const *solid_case = vecgeom::test::FindTestCaseSolid(manual_case.solid_case_name);
  VECGEOM_VALIDATE(solid_case != nullptr, << "Manual edge case '" << manual_case.name << "' references unknown solid '"
                                          << manual_case.solid_case_name << "'.");

  auto shape                        = solid_case->make_shape();
  auto samples                      = MakeManualEdgeCaseSamples(manual_case);
  auto view                         = vecgeom::test::MakeShapeContractSampleView(samples);
  const int sample_index            = ManualEdgeCasePrimarySampleIndex(samples, manual_case);
  const auto target_family          = ParseTestFamilySelection(manual_case.target_family_name);
  const Precision solid_tolerance   = ResolveSolidTolerance(*solid_case);
  const Precision grazing_tolerance = ResolveManualGrazingTolerance(options, manual_case);

  VECGEOM_VALIDATE(sample_index >= 0 && sample_index < view.TotalPoints(),
                   << "Manual edge case '" << manual_case.name << "' did not build a valid sample index.");

  switch (target_family) {
  case ShapeContractTestFamily::kContracts: {
    auto replay = vecgeom::test::ReplayShapeConventionSample(shape.get(), view, sample_index, solid_tolerance,
                                                             MakeDistanceToOutCaller());
    if (!ManualReplayPassed(replay)) {
      VECGEOM_VALIDATE(false, << "Manual edge case '" << manual_case.name << "' for solid '"
                              << manual_case.solid_case_name << "' failed.\n"
                              << DescribeManualEdgeCase(manual_case) << "\n"
                              << vecgeom::test::DescribeShapeContractRayReplay(replay)
                              << MakeManualEdgeCaseDebugHint(executable_path, tier, manual_case, grazing_tolerance));
    }
    if (verbose_on_success)
      std::cout << DescribeManualEdgeCase(manual_case) << "\n"
                << vecgeom::test::DescribeShapeContractRayReplay(replay) << std::endl;
    break;
  }
  case ShapeContractTestFamily::kNormals: {
    auto replay = vecgeom::test::ReplayShapeNormalSample(shape.get(), view, sample_index, solid_tolerance,
                                                         MakeDistanceToOutCaller());
    if (!ManualReplayPassed(replay)) {
      VECGEOM_VALIDATE(false, << "Manual edge case '" << manual_case.name << "' for solid '"
                              << manual_case.solid_case_name << "' failed.\n"
                              << DescribeManualEdgeCase(manual_case) << "\n"
                              << vecgeom::test::DescribeShapeNormalRayReplay(replay)
                              << MakeManualEdgeCaseDebugHint(executable_path, tier, manual_case, grazing_tolerance));
    }
    if (verbose_on_success)
      std::cout << DescribeManualEdgeCase(manual_case) << "\n"
                << vecgeom::test::DescribeShapeNormalRayReplay(replay) << std::endl;
    break;
  }
  case ShapeContractTestFamily::kSurface: {
    auto replay = vecgeom::test::ReplayShapeSurfaceSample(shape.get(), view, sample_index, solid_tolerance,
                                                          grazing_tolerance, MakeDistanceToOutCaller());
    if (!ManualReplayPassed(replay)) {
      VECGEOM_VALIDATE(false, << "Manual edge case '" << manual_case.name << "' for solid '"
                              << manual_case.solid_case_name << "' failed.\n"
                              << DescribeManualEdgeCase(manual_case) << "\n"
                              << vecgeom::test::DescribeShapeSurfaceRayReplay(replay)
                              << MakeManualEdgeCaseDebugHint(executable_path, tier, manual_case, grazing_tolerance));
    }
    if (verbose_on_success)
      std::cout << DescribeManualEdgeCase(manual_case) << "\n"
                << vecgeom::test::DescribeShapeSurfaceRayReplay(replay) << std::endl;
    break;
  }
  case ShapeContractTestFamily::kDistanceToOut: {
    auto replay = vecgeom::test::ReplayShapeDistanceToOutSample(shape.get(), view, sample_index, solid_tolerance,
                                                                MakeDistanceToOutCaller());
    if (!ManualReplayPassed(replay)) {
      VECGEOM_VALIDATE(false, << "Manual edge case '" << manual_case.name << "' for solid '"
                              << manual_case.solid_case_name << "' failed.\n"
                              << DescribeManualEdgeCase(manual_case) << "\n"
                              << vecgeom::test::DescribeShapeDistanceToOutRayReplay(replay)
                              << MakeManualEdgeCaseDebugHint(executable_path, tier, manual_case, grazing_tolerance));
    }
    if (verbose_on_success)
      std::cout << DescribeManualEdgeCase(manual_case) << "\n"
                << vecgeom::test::DescribeShapeDistanceToOutRayReplay(replay) << std::endl;
    break;
  }
  case ShapeContractTestFamily::kDistanceToIn: {
    auto replay = vecgeom::test::ReplayShapeDistanceToInSample(shape.get(), view, sample_index, solid_tolerance);
    if (!ManualReplayPassed(replay)) {
      VECGEOM_VALIDATE(false, << "Manual edge case '" << manual_case.name << "' for solid '"
                              << manual_case.solid_case_name << "' failed.\n"
                              << DescribeManualEdgeCase(manual_case) << "\n"
                              << vecgeom::test::DescribeShapeDistanceToInRayReplay(replay)
                              << MakeManualEdgeCaseDebugHint(executable_path, tier, manual_case, grazing_tolerance));
    }
    if (verbose_on_success)
      std::cout << DescribeManualEdgeCase(manual_case) << "\n"
                << vecgeom::test::DescribeShapeDistanceToInRayReplay(replay) << std::endl;
    break;
  }
  case ShapeContractTestFamily::kSafeties: {
    auto replay = vecgeom::test::ReplayShapeSafetySample(shape.get(), view, sample_index, solid_tolerance,
                                                         MakeDistanceToOutCaller());
    if (!ManualReplayPassed(replay)) {
      VECGEOM_VALIDATE(false, << "Manual edge case '" << manual_case.name << "' for solid '"
                              << manual_case.solid_case_name << "' failed.\n"
                              << DescribeManualEdgeCase(manual_case) << "\n"
                              << vecgeom::test::DescribeShapeSafetyRayReplay(replay)
                              << MakeManualEdgeCaseDebugHint(executable_path, tier, manual_case, grazing_tolerance));
    }
    if (verbose_on_success)
      std::cout << DescribeManualEdgeCase(manual_case) << "\n"
                << vecgeom::test::DescribeShapeSafetyRayReplay(replay) << std::endl;
    break;
  }
  case ShapeContractTestFamily::kHitConsistency: {
    auto replay = vecgeom::test::ReplayShapeHitConsistencySample(shape.get(), view, sample_index, solid_tolerance,
                                                                 MakeDistanceToOutCaller());
    if (!ManualReplayPassed(replay)) {
      VECGEOM_VALIDATE(false, << "Manual edge case '" << manual_case.name << "' for solid '"
                              << manual_case.solid_case_name << "' failed.\n"
                              << DescribeManualEdgeCase(manual_case) << "\n"
                              << vecgeom::test::DescribeShapeHitConsistencyRayReplay(replay)
                              << MakeManualEdgeCaseDebugHint(executable_path, tier, manual_case, grazing_tolerance));
    }
    if (verbose_on_success)
      std::cout << DescribeManualEdgeCase(manual_case) << "\n"
                << vecgeom::test::DescribeShapeHitConsistencyRayReplay(replay) << std::endl;
    break;
  }
  case ShapeContractTestFamily::kManualEdgeCases:
  case ShapeContractTestFamily::kAll:
    VECGEOM_VALIDATE(false, << "Manual edge case '" << manual_case.name << "' has unsupported target family '"
                            << manual_case.target_family_name << "'.");
    break;
  }
}

void ValidateManualEdgeCases(const ShapeContractOptions &options, const std::string &executable_path,
                             ShapeContractTier tier)
{
  auto selected_cases = ResolveSelectedManualEdgeCases(options);
  VECGEOM_VALIDATE(!selected_cases.empty(),
                   << "No manual edge cases matched the requested filters. Use -list_manual_cases to inspect the "
                   << "configured manual cases.");
  const bool verbose_on_success = selected_cases.size() == 1;
  for (auto const *manual_case : selected_cases) {
    ValidateManualEdgeCase(*manual_case, options, executable_path, tier, verbose_on_success);
  }
}

void BenchmarkManualEdgeCase(const vecgeom::test::ManualEdgeCase &manual_case, const ShapeContractOptions &options,
                             ShapeContractTier tier)
{
  auto const *solid_case = vecgeom::test::FindTestCaseSolid(manual_case.solid_case_name);
  VECGEOM_VALIDATE(solid_case != nullptr, << "Manual edge case '" << manual_case.name << "' references unknown solid '"
                                          << manual_case.solid_case_name << "'.");

  auto shape        = solid_case->make_shape();
  auto samples      = MakeManualEdgeCaseSamples(manual_case);
  auto view         = vecgeom::test::MakeShapeContractSampleView(samples);
  const int index   = ManualEdgeCasePrimarySampleIndex(samples, manual_case);
  const auto family = ParseTestFamilySelection(manual_case.target_family_name);
  auto workloads    = BuildReplayBenchmarkWorkloads(family, index);

  std::ostringstream label;
  label << "manual_case='" << manual_case.name << "' solid='" << manual_case.solid_case_name
        << "' topology=" << vecgeom::test::ShapeSampleCategoryLabel(manual_case.topology);
  PrintBenchmarkSummaryHeader(label.str(), tier, family, options, MakeSampleFingerprint(samples));
  BenchmarkWorkloadsForView(shape.get(), view, workloads, options);
}

void BenchmarkManualEdgeCases(const ShapeContractOptions &options, ShapeContractTier tier)
{
  auto selected_cases = ResolveSelectedManualEdgeCases(options);
  VECGEOM_VALIDATE(!selected_cases.empty(),
                   << "No manual edge cases matched the requested filters. Use -list_manual_cases to inspect the "
                   << "configured manual cases.");
  for (auto const *manual_case : selected_cases) {
    BenchmarkManualEdgeCase(*manual_case, options, tier);
  }
}

void ReplayContractsSample(ShapeContractExecutionCache &cache, const ShapeContractOptions &options)
{
  auto view = vecgeom::test::MakeShapeContractSampleView(cache.Samples());

  VECGEOM_VALIDATE(options.replay_index >= 0 && options.replay_index < view.TotalPoints(),
                   << "Replay index " << options.replay_index << " is outside [0, " << view.TotalPoints()
                   << ") for solid '" << cache.SolidCase().name << "'.");

  auto replay = vecgeom::test::ReplayShapeConventionSample(cache.Shape(), view, options.replay_index,
                                                           cache.SolidTolerance(), MakeDistanceToOutCaller());
  std::cout << vecgeom::test::DescribeShapeContractRayReplay(replay) << std::endl;
}

void ReplayNormalsSample(ShapeContractExecutionCache &cache, const ShapeContractOptions &options)
{
  auto view = vecgeom::test::MakeShapeContractSampleView(cache.Samples());

  VECGEOM_VALIDATE(options.replay_index >= 0 && options.replay_index < view.TotalPoints(),
                   << "Replay index " << options.replay_index << " is outside [0, " << view.TotalPoints()
                   << ") for solid '" << cache.SolidCase().name << "'.");

  auto replay = vecgeom::test::ReplayShapeNormalSample(cache.Shape(), view, options.replay_index,
                                                       cache.SolidTolerance(),
                                                       MakeDistanceToOutCaller());
  std::cout << vecgeom::test::DescribeShapeNormalRayReplay(replay) << std::endl;
}

void ReplaySurfaceSample(ShapeContractExecutionCache &cache, const ShapeContractOptions &options)
{
  auto view = vecgeom::test::MakeShapeContractSampleView(cache.Samples());

  VECGEOM_VALIDATE(options.replay_index >= 0 && options.replay_index < view.TotalPoints(),
                   << "Replay index " << options.replay_index << " is outside [0, " << view.TotalPoints()
                   << ") for solid '" << cache.SolidCase().name << "'.");

  auto replay = vecgeom::test::ReplayShapeSurfaceSample(cache.Shape(), view, options.replay_index,
                                                        cache.SolidTolerance(), cache.GrazingTolerance(),
                                                        MakeDistanceToOutCaller());
  std::cout << vecgeom::test::DescribeShapeSurfaceRayReplay(replay) << std::endl;
}

void ReplayDistanceToOutSample(ShapeContractExecutionCache &cache, const ShapeContractOptions &options)
{
  auto view = vecgeom::test::MakeShapeContractSampleView(cache.Samples());

  VECGEOM_VALIDATE(options.replay_index >= 0 && options.replay_index < view.TotalPoints(),
                   << "Replay index " << options.replay_index << " is outside [0, " << view.TotalPoints()
                   << ") for solid '" << cache.SolidCase().name << "'.");

  auto replay = vecgeom::test::ReplayShapeDistanceToOutSample(cache.Shape(), view, options.replay_index,
                                                              cache.SolidTolerance(), MakeDistanceToOutCaller());
  std::cout << vecgeom::test::DescribeShapeDistanceToOutRayReplay(replay) << std::endl;
}

void ReplayDistanceToInSample(ShapeContractExecutionCache &cache, const ShapeContractOptions &options)
{
  auto view = vecgeom::test::MakeShapeContractSampleView(cache.Samples());

  VECGEOM_VALIDATE(options.replay_index >= 0 && options.replay_index < view.TotalPoints(),
                   << "Replay index " << options.replay_index << " is outside [0, " << view.TotalPoints()
                   << ") for solid '" << cache.SolidCase().name << "'.");

  auto replay = vecgeom::test::ReplayShapeDistanceToInSample(cache.Shape(), view, options.replay_index,
                                                             cache.SolidTolerance());
  std::cout << vecgeom::test::DescribeShapeDistanceToInRayReplay(replay) << std::endl;
}

void ReplaySafetySample(ShapeContractExecutionCache &cache, const ShapeContractOptions &options)
{
  auto view = vecgeom::test::MakeShapeContractSampleView(cache.Samples());

  VECGEOM_VALIDATE(options.replay_index >= 0 && options.replay_index < view.TotalPoints(),
                   << "Replay index " << options.replay_index << " is outside [0, " << view.TotalPoints()
                   << ") for solid '" << cache.SolidCase().name << "'.");

  auto replay = vecgeom::test::ReplayShapeSafetySample(cache.Shape(), view, options.replay_index,
                                                       cache.SolidTolerance(),
                                                       MakeDistanceToOutCaller());
  std::cout << vecgeom::test::DescribeShapeSafetyRayReplay(replay) << std::endl;
}

void ReplayHitConsistencySample(ShapeContractExecutionCache &cache, const ShapeContractOptions &options)
{
  auto view = vecgeom::test::MakeShapeContractSampleView(cache.Samples());

  VECGEOM_VALIDATE(options.replay_index >= 0 && options.replay_index < view.TotalPoints(),
                   << "Replay index " << options.replay_index << " is outside [0, " << view.TotalPoints()
                   << ") for solid '" << cache.SolidCase().name << "'.");

  auto replay = vecgeom::test::ReplayShapeHitConsistencySample(cache.Shape(), view, options.replay_index,
                                                               cache.SolidTolerance(), MakeDistanceToOutCaller());
  std::cout << vecgeom::test::DescribeShapeHitConsistencyRayReplay(replay) << std::endl;
}

void ValidateContractsFamily(ShapeContractExecutionCache &cache, const std::string &executable_path)
{
  auto const &outcome    = cache.ContractOutcome();
  auto const &solid_case = cache.SolidCase();
  auto const &config     = cache.Config();
  const auto tier        = cache.Tier();

  VECGEOM_VALIDATE(outcome.summary.Passed(),
                   << TestFamilyDescription(tier, ShapeContractTestFamily::kContracts) << " '" << solid_case.name
                   << "' failed." << " seed=" << config.seed << " stream_id=" << config.stream_id
                   << " npoints=" << config.max_points << " score=" << outcome.summary.score << " errors="
                   << outcome.result.CountErrors() << vecgeom::test::DescribeShapeCheckResult(outcome.result)
                   << MakeHowToDebugHint(executable_path, solid_case.name, outcome.result, config, solid_case, tier,
                                         ShapeContractTestFamily::kContracts, cache.GrazingTolerance()));

  VECGEOM_ASSERT(outcome.summary.score == 0);
  VECGEOM_ASSERT(outcome.result.CountErrors() == 0);

  if (tier != ShapeContractTier::kFast) {
    std::cout << TestFamilyDescription(tier, ShapeContractTestFamily::kContracts) << " '" << solid_case.name
              << "' passed" << " seed=" << config.seed << " stream_id=" << config.stream_id
              << " npoints=" << config.max_points << " fingerprint=" << cache.Fingerprint() << std::endl;
  }
}

void ValidateNormalsFamily(ShapeContractExecutionCache &cache, const std::string &executable_path)
{
  auto const &outcome    = cache.NormalOutcome();
  auto const &solid_case = cache.SolidCase();
  auto const &config     = cache.Config();
  const auto tier        = cache.Tier();

  VECGEOM_VALIDATE(outcome.summary.Passed(),
                   << TestFamilyDescription(tier, ShapeContractTestFamily::kNormals) << " '" << solid_case.name
                   << "' failed." << " seed=" << config.seed << " stream_id=" << config.stream_id
                   << " npoints=" << config.max_points << " score=" << outcome.summary.score << " errors="
                   << outcome.result.CountErrors() << vecgeom::test::DescribeShapeCheckResult(outcome.result)
                   << MakeHowToDebugHint(executable_path, solid_case.name, outcome.result, config, solid_case, tier,
                                         ShapeContractTestFamily::kNormals, cache.GrazingTolerance()));

  VECGEOM_ASSERT(outcome.summary.score == 0);
  VECGEOM_ASSERT(outcome.result.CountErrors() == 0);

  if (tier != ShapeContractTier::kFast) {
    std::cout << TestFamilyDescription(tier, ShapeContractTestFamily::kNormals) << " '" << solid_case.name << "' passed"
              << " seed=" << config.seed << " stream_id=" << config.stream_id << " npoints=" << config.max_points
              << " fingerprint=" << cache.Fingerprint() << std::endl;
  }
}

void ValidateSurfaceFamily(ShapeContractExecutionCache &cache, const std::string &executable_path)
{
  auto const &outcome    = cache.SurfaceOutcome();
  auto const &solid_case = cache.SolidCase();
  auto const &config     = cache.Config();
  const auto tier        = cache.Tier();

  VECGEOM_VALIDATE(outcome.summary.Passed(),
                   << TestFamilyDescription(tier, ShapeContractTestFamily::kSurface) << " '" << solid_case.name
                   << "' failed." << " seed=" << config.seed << " stream_id=" << config.stream_id
                   << " npoints=" << config.max_points << " score=" << outcome.summary.score << " errors="
                   << outcome.result.CountErrors() << vecgeom::test::DescribeShapeCheckResult(outcome.result)
                   << MakeHowToDebugHint(executable_path, solid_case.name, outcome.result, config, solid_case, tier,
                                         ShapeContractTestFamily::kSurface, cache.GrazingTolerance()));

  VECGEOM_ASSERT(outcome.summary.score == 0);
  VECGEOM_ASSERT(outcome.result.CountErrors() == 0);

  if (tier != ShapeContractTier::kFast) {
    std::cout << TestFamilyDescription(tier, ShapeContractTestFamily::kSurface) << " '" << solid_case.name << "' passed"
              << " seed=" << config.seed << " stream_id=" << config.stream_id << " npoints=" << config.max_points
              << " fingerprint=" << cache.Fingerprint() << std::endl;
  }
}

void ValidateDistanceToOutFamily(ShapeContractExecutionCache &cache, const std::string &executable_path)
{
  auto const &outcome    = cache.DistanceToOutOutcome();
  auto const &solid_case = cache.SolidCase();
  auto const &config     = cache.Config();
  const auto tier        = cache.Tier();

  VECGEOM_VALIDATE(outcome.summary.Passed(),
                   << TestFamilyDescription(tier, ShapeContractTestFamily::kDistanceToOut) << " '" << solid_case.name
                   << "' failed." << " seed=" << config.seed << " stream_id=" << config.stream_id
                   << " npoints=" << config.max_points << " score=" << outcome.summary.score << " errors="
                   << outcome.result.CountErrors() << vecgeom::test::DescribeShapeCheckResult(outcome.result)
                   << MakeHowToDebugHint(executable_path, solid_case.name, outcome.result, config, solid_case, tier,
                                         ShapeContractTestFamily::kDistanceToOut, cache.GrazingTolerance()));

  VECGEOM_ASSERT(outcome.summary.score == 0);
  VECGEOM_ASSERT(outcome.result.CountErrors() == 0);

  if (tier != ShapeContractTier::kFast) {
    std::cout << TestFamilyDescription(tier, ShapeContractTestFamily::kDistanceToOut) << " '" << solid_case.name
              << "' passed" << " seed=" << config.seed << " stream_id=" << config.stream_id
              << " npoints=" << config.max_points << " fingerprint=" << cache.Fingerprint() << std::endl;
  }
}

void ValidateDistanceToInFamily(ShapeContractExecutionCache &cache, const std::string &executable_path)
{
  auto const &outcome    = cache.DistanceToInOutcome();
  auto const &solid_case = cache.SolidCase();
  auto const &config     = cache.Config();
  const auto tier        = cache.Tier();

  VECGEOM_VALIDATE(outcome.summary.Passed(),
                   << TestFamilyDescription(tier, ShapeContractTestFamily::kDistanceToIn) << " '" << solid_case.name
                   << "' failed." << " seed=" << config.seed << " stream_id=" << config.stream_id << " npoints="
                   << config.max_points << " evaluated_outside_rays=" << outcome.summary.evaluated_outside_rays
                   << " score=" << outcome.summary.score << " errors=" << outcome.result.CountErrors()
                   << vecgeom::test::DescribeShapeCheckResult(outcome.result)
                   << MakeHowToDebugHint(executable_path, solid_case.name, outcome.result, config, solid_case, tier,
                                         ShapeContractTestFamily::kDistanceToIn, cache.GrazingTolerance()));

  VECGEOM_ASSERT(outcome.summary.score == 0);
  VECGEOM_ASSERT(outcome.result.CountErrors() == 0);

  if (tier != ShapeContractTier::kFast) {
    std::cout << TestFamilyDescription(tier, ShapeContractTestFamily::kDistanceToIn) << " '" << solid_case.name
              << "' passed" << " seed=" << config.seed << " stream_id=" << config.stream_id
              << " npoints=" << config.max_points << " fingerprint=" << cache.Fingerprint() << std::endl;
  }
}

void ValidateSafetyFamily(ShapeContractExecutionCache &cache, const std::string &executable_path)
{
  auto const &outcome    = cache.SafetyOutcome();
  auto const &solid_case = cache.SolidCase();
  auto const &config     = cache.Config();
  const auto tier        = cache.Tier();

  VECGEOM_VALIDATE(outcome.summary.Passed(),
                   << TestFamilyDescription(tier, ShapeContractTestFamily::kSafeties) << " '" << solid_case.name
                   << "' failed." << " seed=" << config.seed << " stream_id=" << config.stream_id << " npoints="
                   << config.max_points << " evaluated_inside_points=" << outcome.summary.evaluated_inside_points
                   << " evaluated_outside_points=" << outcome.summary.evaluated_outside_points
                   << " score=" << outcome.summary.score << " errors=" << outcome.result.CountErrors()
                   << vecgeom::test::DescribeShapeCheckResult(outcome.result)
                   << MakeHowToDebugHint(executable_path, solid_case.name, outcome.result, config, solid_case, tier,
                                         ShapeContractTestFamily::kSafeties, cache.GrazingTolerance()));

  VECGEOM_ASSERT(outcome.summary.score == 0);
  VECGEOM_ASSERT(outcome.result.CountErrors() == 0);

  if (tier != ShapeContractTier::kFast) {
    std::cout << TestFamilyDescription(tier, ShapeContractTestFamily::kSafeties) << " '" << solid_case.name
              << "' passed" << " seed=" << config.seed << " stream_id=" << config.stream_id
              << " npoints=" << config.max_points << " fingerprint=" << cache.Fingerprint() << std::endl;
  }
}

void ValidateHitConsistencyFamily(ShapeContractExecutionCache &cache, const std::string &executable_path)
{
  auto const &outcome    = cache.HitConsistencyOutcome();
  auto const &solid_case = cache.SolidCase();
  auto const &config     = cache.Config();
  const auto tier        = cache.Tier();

  VECGEOM_VALIDATE(outcome.summary.Passed(),
                   << TestFamilyDescription(tier, ShapeContractTestFamily::kHitConsistency) << " '" << solid_case.name
                   << "' failed." << " seed=" << config.seed << " stream_id=" << config.stream_id << " npoints="
                   << config.max_points << " evaluated_inside_points=" << outcome.summary.evaluated_inside_points
                   << " evaluated_outside_points=" << outcome.summary.evaluated_outside_points
                   << " score=" << outcome.summary.score << " errors=" << outcome.result.CountErrors()
                   << vecgeom::test::DescribeShapeCheckResult(outcome.result)
                   << MakeHowToDebugHint(executable_path, solid_case.name, outcome.result, config, solid_case, tier,
                                         ShapeContractTestFamily::kHitConsistency, cache.GrazingTolerance()));

  VECGEOM_ASSERT(outcome.summary.score == 0);
  VECGEOM_ASSERT(outcome.result.CountErrors() == 0);

  if (tier != ShapeContractTier::kFast) {
    std::cout << TestFamilyDescription(tier, ShapeContractTestFamily::kHitConsistency) << " '" << solid_case.name
              << "' passed" << " seed=" << config.seed << " stream_id=" << config.stream_id
              << " npoints=" << config.max_points << " fingerprint=" << cache.Fingerprint() << std::endl;
  }
}

// Dispatch one sampled solid through the requested helper family, reusing a
// single execution cache when `-test_family all` groups several families
// together for the same solid/tier invocation.
void ValidateSelectedFamilies(const vecgeom::test::TestCaseSolid &solid_case, const ShapeContractOptions &options,
                              const std::string &executable_path, ShapeContractTier tier,
                              ShapeContractTestFamily test_family)
{
  ShapeContractExecutionCache cache(solid_case, tier, options);
  switch (test_family) {
  case ShapeContractTestFamily::kContracts:
    ValidateContractsFamily(cache, executable_path);
    break;
  case ShapeContractTestFamily::kNormals:
    ValidateNormalsFamily(cache, executable_path);
    break;
  case ShapeContractTestFamily::kSurface:
    ValidateSurfaceFamily(cache, executable_path);
    break;
  case ShapeContractTestFamily::kDistanceToOut:
    ValidateDistanceToOutFamily(cache, executable_path);
    break;
  case ShapeContractTestFamily::kDistanceToIn:
    ValidateDistanceToInFamily(cache, executable_path);
    break;
  case ShapeContractTestFamily::kSafeties:
    ValidateSafetyFamily(cache, executable_path);
    break;
  case ShapeContractTestFamily::kHitConsistency:
    ValidateHitConsistencyFamily(cache, executable_path);
    break;
  case ShapeContractTestFamily::kManualEdgeCases:
    VECGEOM_VALIDATE(false, << "manual_edge_cases should be handled before sampled family dispatch.");
    break;
  case ShapeContractTestFamily::kAll:
    ValidateContractsFamily(cache, executable_path);
    ValidateNormalsFamily(cache, executable_path);
    ValidateSurfaceFamily(cache, executable_path);
    ValidateDistanceToOutFamily(cache, executable_path);
    ValidateDistanceToInFamily(cache, executable_path);
    ValidateSafetyFamily(cache, executable_path);
    ValidateHitConsistencyFamily(cache, executable_path);
    break;
  }
}

// Replay one sampled ray from the cached sample set for the selected family.
void ReplaySelectedFamily(const vecgeom::test::TestCaseSolid &solid_case, const ShapeContractOptions &options,
                          ShapeContractTier tier, ShapeContractTestFamily test_family)
{
  ShapeContractExecutionCache cache(solid_case, tier, options);
  switch (test_family) {
  case ShapeContractTestFamily::kContracts:
    ReplayContractsSample(cache, options);
    break;
  case ShapeContractTestFamily::kNormals:
    ReplayNormalsSample(cache, options);
    break;
  case ShapeContractTestFamily::kSurface:
    ReplaySurfaceSample(cache, options);
    break;
  case ShapeContractTestFamily::kDistanceToOut:
    ReplayDistanceToOutSample(cache, options);
    break;
  case ShapeContractTestFamily::kDistanceToIn:
    ReplayDistanceToInSample(cache, options);
    break;
  case ShapeContractTestFamily::kSafeties:
    ReplaySafetySample(cache, options);
    break;
  case ShapeContractTestFamily::kHitConsistency:
    ReplayHitConsistencySample(cache, options);
    break;
  case ShapeContractTestFamily::kManualEdgeCases:
    VECGEOM_VALIDATE(false, << "manual_edge_cases do not use -replay_index. Rerun the selected manual case instead.");
    break;
  case ShapeContractTestFamily::kAll:
    VECGEOM_VALIDATE(false, << "Replay requires a single helper test family, not -test_family all.");
    break;
  }
}

// Entry point shared by the default "run everything", per-family, per-case,
// per-family-group, and replay CLI shapes.
void RunRequestedSolidCases(const ShapeContractOptions &options, const std::string &executable_path)
{
  const auto tier        = ParseTierSelection(options.tier_name);
  const auto test_family = ParseTestFamilySelection(options.test_family_name);
  VECGEOM_VALIDATE(options.case_name == "all" || options.family_name == "all",
                   << "Use either -case_name or -family, not both.");
  if (test_family == ShapeContractTestFamily::kManualEdgeCases) {
    if (options.benchmark_mode) {
      BenchmarkManualEdgeCases(options, tier);
    } else {
      ValidateManualEdgeCases(options, executable_path, tier);
    }
    return;
  }
  VECGEOM_VALIDATE(!(options.replay_index >= 0 && options.case_name == "all"),
                   << "Replay requires a single case name, not 'all' or a family selection.");
  VECGEOM_VALIDATE(!(options.replay_index >= 0 && test_family == ShapeContractTestFamily::kAll),
                   << "Replay requires a single helper test family, not -test_family all.");

  if (options.case_name == "all") {
    if (options.family_name != "all") {
      for (auto const *solid_case : vecgeom::test::FindTestCaseSolidsByFamily(options.family_name)) {
        if (options.benchmark_mode) {
          BenchmarkSelectedFamilies(*solid_case, options, tier, test_family);
        } else {
          ValidateSelectedFamilies(*solid_case, options, executable_path, tier, test_family);
        }
      }
      return;
    }
    for (auto const &solid_case : vecgeom::test::GetTestCaseSolids()) {
      if (options.benchmark_mode) {
        BenchmarkSelectedFamilies(solid_case, options, tier, test_family);
      } else {
        ValidateSelectedFamilies(solid_case, options, executable_path, tier, test_family);
      }
    }
    return;
  }

  auto const *solid_case = vecgeom::test::FindTestCaseSolid(options.case_name);
  VECGEOM_VALIDATE(solid_case != nullptr, << "Unknown ShapeContractTest case '" << options.case_name << "'.");
  if (options.replay_index >= 0) {
    if (options.benchmark_mode) {
      BenchmarkReplaySample(*solid_case, options, tier, test_family);
    } else {
      ReplaySelectedFamily(*solid_case, options, tier, test_family);
    }
    return;
  }
  if (options.benchmark_mode) {
    BenchmarkSelectedFamilies(*solid_case, options, tier, test_family);
  } else {
    ValidateSelectedFamilies(*solid_case, options, executable_path, tier, test_family);
  }
}

} // namespace

int main(int argc, char *argv[])
{
  const auto options = ParseOptions(argc, argv);
  if (options.show_help) {
    PrintUsage(argv[0]);
    return 0;
  }
  ParseTierSelection(options.tier_name);
  const auto test_family = ParseTestFamilySelection(options.test_family_name);
  ValidateManualOptionUsage(options, test_family);
  ValidateBenchmarkOptions(options);
  if (options.list_families) {
    std::cout << JoinConfiguredFamilyNames() << std::endl;
    return 0;
  }
  if (options.list_test_families) {
    std::cout << JoinConfiguredTestFamilyNames() << std::endl;
    return 0;
  }
  ValidateFamilySelection(options.family_name);
  if (options.list_cases) {
    std::cout << JoinConfiguredCaseNames(options.family_name) << std::endl;
    return 0;
  }
  if (options.list_manual_cases) {
    PrintManualEdgeCases(options);
    return 0;
  }
  PrintDefaultRunSummary(options);

  VECGEOM_ASSERT(vecgeom::test::ShapeConventionMessages().size() ==
                 static_cast<size_t>(vecgeom::test::kShapeConventionBitCount));
  RunRequestedSolidCases(options, argv[0]);
  return 0;
}
