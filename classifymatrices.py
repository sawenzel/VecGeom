import itertools
allpos=set([0,1,2,3,4,5,6,7,8])

def printoneline(x, zeroentries):
    if(x==0):
        code="local.x = "
    if(x==1):
        code="local.y = "
    if(x==2):
        code="local.z = "
    first=True
    for j in range(0,3):
        pos = x+3*j;
        if not pos in zeroentries:
            if first:
                code=code + "mt" + str(j) + "*rot[" + str(pos) + "]";
                first=False
            else:
                code=code + "+mt" + str(j) + "*rot[" + str(pos) + "]";
    code=code+";"
    return code       

def printspecializedrotation(zeroentries):
    # x position
    print printoneline(0,zeroentries)
    print printoneline(1,zeroentries)    
    print printoneline(2,zeroentries)

def getzeroentries(i,j):
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
        print str(x)+" "+str(y)+" "+ str(x+y*3)
        C.add(x+y*3)
    x=i
    for y in RS.difference(B):
        id=id+(x+y*3)*(x+y*3)*(x+y*3)
        print str(x)+" "+str(y)+" "+ str(x+y*3)
        C.add(x+y*3)

    print C
    print "## id of this pattern (" + str(i) + ";" + str(j) + ") = " + str(id)
    printspecializedrotation(C)

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
        print id-s
        printspecializedrotation(zeros)           

def main():
    for i in range(0,3):
        for j in range(0,3):
            getzeroentries(i,j)

    getsixzeros()

if __name__ == "__main__":
    main()
