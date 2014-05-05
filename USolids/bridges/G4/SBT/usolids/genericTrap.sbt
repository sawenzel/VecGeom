#
# GEANT4 SBT Script to test G4GenericTrap
# derived from G4ExtrudedSolid done by I.Hrivnakova
# 19/11/2009 T.Nikitina

#
/test/maxPoints 1000
#
# --- genericTrap.a1.log
# Generic Trap with 8 vertices no twist(box like)
#

/performance/maxPoints 1000


# --- genericTrap.c1.log
# Generic Trap with 5 dif. vertices no twist(Tet like)
#

/solid/G4GenericTrap 1 (-3, -3, 3, 3, -3, -3, 3, 3) (-2, -2, -2, -2, -2, -2, -2, -2)
/performance/errorFileName log/genericTrap3-1000-values/trapp.a1.log
/performance/repeat 1

/control/execute usolids/values.sbt

exit

# Generic Trap with 8 vertices no twist(Trd like)

/solid/G4GenericTrap 1 (-3, -3, 3, 3, -3, -3, 3, 3) (-3, 3, 3, -3, -3, 3, 3, -3)
/performance/errorFileName log/genericTrap-1000-values/trapp.a1.log
/performance/repeat 1

/control/execute usolids/values.sbt


/solid/G4GenericTrap 1 (-3, -3, 3, 3, -3, -3, 3, 3) (-2, 2, 2, -2, -2, 2, 2, -2)
/performance/errorFileName log/genericTrap2-1000-values/trapp.a1.log
/performance/repeat 1

/control/execute usolids/values.sbt


#
#
# --- genericTrap.d1.log
# Generic Trap with 8 vertices with twist
#

/solid/G4GenericTrap 1 (-3, -3, 3, 3, -0.5, -2, 2, 2) (-3, 3, 3, -3, -2, 2, 2, -2)
/performance/errorFileName log/genericTrap4-1000-values/trapp.a1.log
/performance/repeat 1

/control/execute usolids/values.sbt

exit

