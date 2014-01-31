import itertools
allpos=set([0,1,2,3,4,5,6,7,8])
conversiontable={0:"x", 1:"y", 2: "z"}

def printoneline(x, zeroentries):
    code="local"+conversiontable[x]+"="
    first=True
    for j in range(0,3):
	pos = x+3*j;
        if not pos in zeroentries:
            if first:
                code=code + "mt" + conversiontable[j] + "*rot[" + str(pos) + "]";
                first=False
            else:
                code=code + "+mt" + conversiontable[j] + "*rot[" + str(pos) + "]";
    code=code+";"
    return code       

def printspecializedrotation(footprint, zeroentries):
    # x position
    print "if(rid=="+str(footprint)+"){"
    print "     " + printoneline(0,zeroentries)
    print "     " + printoneline(1,zeroentries)
    print "     " + printoneline(2,zeroentries)
    print "     return;"
    print "}"


def printgeneralrotation():
    # x position
    zeroentries=set()
    print printoneline(0,zeroentries)
    print printoneline(1,zeroentries)
    print printoneline(2,zeroentries)

def printinstantiationcode(tid,rid,classname):
    print "if(tid == " + str(tid) + " && rid == " + str(rid) + " ) return new " + str(classname) + "<" + str(tid) + "," + str(rid) + ">(bp,tmp)"

def emitheader():
    print "template<RotationIdType rid> void emitrotationcode(Vector3D const & mt, Vector3D & local) const {"

def emitheaderT():
    print "template<RotationIdType rid, typename T> void emitrotationcode(T const & mx, T const & mx, T const & mx, T & localx, T & localy, T & localz) const {"

def emitspecializedrotationcode():
    emitheaderT()
    #do the cases with four zeros
    for i in range(0,3):
        for j in range(0,3):
            getfourzeros(i,j)

        #do the cases with six zeros
    getsixzeros()

# print general case
    printgeneralrotation()
    print "}"

def getfourzeros(i,j):
    """
    i is column, j is row of the +- one entry
    """
    id=0
    CS=set([0,1,2])
    RS=set([0,1,2])
    A=set();A.add(i)
    B=set();B.add(j)
    C=set([])
    for x in CS.difference(A):
        y=j
        id=id+(x+y*3)*(x+y*3)*(x+y*3)
        #print str(x)+" "+str(y)+" "+ str(x+y*3)
        C.add(x+y*3)
    x=i
    for y in RS.difference(B):
        id=id+(x+y*3)*(x+y*3)*(x+y*3)
        #print str(x)+" "+str(y)+" "+ str(x+y*3)
        C.add(x+y*3)

            #print C
            #print "## id of this pattern (" + str(i) + ";" + str(j) + ") = " + str(id)
    printspecializedrotation(id,C)
  # printinstantiationcode(0,id,"PlacedBox")
  # printinstantiationcode(1,id,"PlacedBox")

def getsixzeros():
    id=0

    for i in range(0,3):
        for j in range(0,3):
            id=id+(i+3*j)*(i+3*j)*(i+3*j)
    P=itertools.permutations([0,1,2])

    for x in P:
        zeros=allpos
        s=0
        for i in range(0,3):
           s=s+(i+3*x[i])*(i+3*x[i])*(i+3*x[i])
           zeros=zeros.difference(set([i+3*x[i]]))
#print id-s
        printspecializedrotation(id-s,zeros)
   #    printinstantiationcode(0,id-s,"PlacedBox")
   #    printinstantiationcode(1,id-s,"PlacedBox")

def main():
    #for i in range(0,3):
    #    for j in range(0,3):
    #        getzeroentries(i,j)

#getsixzeros()
    emitspecializedrotationcode()

if __name__ == "__main__":
    main()
