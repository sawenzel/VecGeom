# Navigation {#navigation}

This page collects implementation notes for VecGeom navigation. It is intended
as the entry point for the full navigation system documentation. The sections
below are intentionally organized by concept so that the navigation-state table
description can grow together with navigator, locator, safety, surface, GPU,
and validation details.

## Navigation State Tables {#navigation_state_tables}

VecGeom navigators need a compact state that identifies the current touchable:
the concrete path from the world volume to the currently located placed volume.
On GPU this state must be small, trivially copyable, and independent of host
pointers. The navigation table is the precomputed data structure that makes
this possible. It translates between a navigation state and the placed-volume
path metadata needed during transport: parent links, placed-volume ids,
logical-volume ids, child ids, daughter links, scene ids, and cached
transformations.

The navigation table is derived from the placed-volume hierarchy during geometry
closure. It is not a second geometry description. It stores enough indexing
information to recover the current path and to move up or down the hierarchy
without carrying a CPU-only path object in each track. The same common
`NavigationState` API is implemented by both table-backed representations:
`NavStateIndex` and `NavStateTuple`.

### NavStateIndex

`NavStateIndex` is the fully expanded representation. A navigation state is one
`NavIndex_t` pointing to a table record for the full touchable. Each record
stores its parent, the current placed-volume id, logical-volume id, touchable
id, child id, daughter records, level, and optional cached transformation.

The main advantage is runtime simplicity. Since every touchable has its own
record, the state is a single index and operations can follow direct parent and
daughter links in one expanded table. When the table fits comfortably in the
accepted memory budget, this representation is usually the preferred
performance-oriented choice.

The cost is table size and initialization time. Repeated detector structures can
create a very large number of distinct touchables even when the underlying
placed and logical volumes are heavily reused. In such geometries, the expanded
index table can become too large to transfer or initialize efficiently. It also
has an encoded daughter-count limit, so geometries with a logical volume having
too many daughters cannot use `NavStateIndex`.

### NavStateTuple

`NavStateTuple` is the scene-compressed representation and is the default
navigation-state table choice. Instead of expanding every repeated subtree into
every parent context, the table is split into scene-local touchable records and
shared logical-volume records. A track state stores a fixed-size tuple of
scene-local table indices, one component per active scene.

This greatly reduces table size for geometries with repeated or reusable
subtrees, and keeps the default geometry closure safer for large detector
setups. The cost is that the per-track state is larger than a single index and
some operations must reconstruct full-path information by combining tuple
components and scene-local parent links.

The tuple depth is the maximum number of active scene components that can be
stored in a track state. The default depth is 4. Smaller depths are preferable
when they fit the geometry because they reduce per-track state size and GPU
state traffic. Larger depths may be needed when a geometry requires deeper scene
nesting to stay within the configured table memory budget.

### Choosing a representation

The preferred representation is the fastest one that fits the user's accepted
memory and initialization budget. In practice this means:

- use `NavStateIndex` when the expanded table fits comfortably and the geometry
  satisfies the encoding limits;
- use `NavStateTuple` when the expanded index table is too large, too slow to
  initialize or transfer, or incompatible with the geometry;
- keep tuple depth as small as possible while still satisfying the memory
  target.

`VECGEOM_NAV` selects the representation (`tuple` or `index`). The default is
`tuple`, with `VECGEOM_NAVTUPLE_MAXDEPTH=4`.

`VECGEOM_NAVTABLE_WARN_MEMORY_MB` sets the table memory budget used for
guidance messages. The default is 512 MB. Set it to 0 to disable size guidance.

`VECGEOM_NAVTABLE_RECOMMEND` enables the more expensive recommendation mode. It
can run additional count-only table passes to estimate alternative tuple depths
and representation choices. This is opt-in because users generally need this
scan only when choosing settings for a geometry, not on every production run.

During `CloseGeometry`, VecGeom reports cheap guidance when the current setting
looks suboptimal for the configured memory budget, for example when
`NavStateIndex` appears to fit but `NavStateTuple` was selected, or when the
index representation is impossible for the geometry. Recommendation mode can
then be used once to choose a better tuple depth or representation.

## Navigators {#navigation_navigators}

To be expanded.

## Locators {#navigation_locators}

To be expanded.

## Safety Estimators {#navigation_safety_estimators}

To be expanded.

## Surface Navigation {#navigation_surface}

To be expanded.

## GPU Navigation {#navigation_gpu}

To be expanded.

## Validation and Diagnostics {#navigation_validation}

To be expanded.
