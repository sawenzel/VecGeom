#
# GEANT4 SBT Script to test G4Cons
# DCW 19/3/99 First try
#
# Increment the number below when errors become a bit more rare
#
#
# --- trap.a{1,2}.log
# Here is a trap,  used and getting problems in CMS    May 28, 1999
#

/performance/maxPoints 10000

/solid/G4Trap   60	20	5	40	30	40	10	16	10	14	10

/performance/errorFileName log/trap2-1m-values-test/trapp.a1.log
/performance/repeat 1

/control/execute usolids/values.sbt

exit

/performance/maxPoints 10

/solid/G4Trap   1268.  0 0 295.  1712.2 1870.29 0 295.  1712.2 1870.29 0

/performance/errorFileName log/trap-1m-values-check/trapp.a1.log
/performance/repeat 1

/performance/maxInsidePercent 10 # john: 10
/performance/maxOutsidePercent 60 # john: 60
/performance/method DistanceToIn
/performance/run 

/control/execute usolids/values.sbt

exit

/performance/maxPoints 1000000






#4Trap(const G4String& pName,
#             G4double  pDz, G4double  pTheta,
#             G4double  pPhi, 
#			 G4double  pDy1,
#             G4double  pDx1, G4double  pDx2,
#             G4double  pAlp1, G4double  pDy2,
#             G4double  pDx3, G4double  pDx4,
#             G4double  pAlp2) 

# In the picture: pDx1 = 30, pDx2 = 40, pDy1 = 40
# pDx3 = 10, pDx4 = 14, pDy2 = 16
# pDz = 60, pTheta = 20*Degree
# pDphi = 5*Degree, pAlph1 = pAlph2 = 10*Degree




exit







/performance/maxPoints 10000000

/solid/G4Trap   1268.  0 0 295.  1712.2 1870.29 0 295.  1712.2 1870.29 0

/performance/errorFileName log/trap-10m/trapp.a1.log
/performance/repeat 1000

/control/execute usolids/performance.sbt

exit





/performance/differenceTolerance 0.02

/performance/maxInsidePercent 25
/performance/maxOutsidePercent 50
/performance/method Inside
/performance/run

exit
