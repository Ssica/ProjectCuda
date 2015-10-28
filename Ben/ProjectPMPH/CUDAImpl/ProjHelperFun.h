#ifndef PROJ_HELPER_FUNS
#define PROJ_HELPER_FUNS

#include <vector>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Constants.h"

using namespace std;


struct PrivGlobs {

    //	grid
    REAL*               myX;        // [numX]
    REAL*               myY;        // [numY]
    REAL*               myTimeline; // [numT]
    unsigned            myXindex;  
    unsigned            myYindex;

    //	variable
    REAL* myResult; // [numX][numY]

    //	coeffs
    REAL*   myVarX; // [numX][numY]
    REAL*   myVarY; // [numX][numY]

    //	operators
    REAL*   myDxx;  // [numX][4]
    REAL*   myDyy;  // [numY][4]

    PrivGlobs( ) {
        printf("Invalid Contructor: need to provide the array sizes! EXITING...!\n");
        exit(0);
    }

    PrivGlobs(  const unsigned int& numX,
                const unsigned int& numY,
                const unsigned int& numT ) {
        this-> myX = (REAL*)malloc(numX*sizeof(REAL));
        this-> myDxx = (REAL*)malloc(numX*4*sizeof(REAL));

        this-> myY = (REAL*)malloc(numY*sizeof(REAL));
        this-> myDyy = (REAL*)malloc(numY*4*sizeof(REAL));

        this-> myTimeline = (REAL*)malloc(numT*sizeof(REAL));

        this-> myVarX = (REAL*)malloc(numX*numY*sizeof(REAL));
        this-> myVarY = (REAL*)malloc(numX*numY*sizeof(REAL));
        this-> myResult = (REAL*)malloc(numX*numY*sizeof(REAL));
        
    }
} __attribute__ ((aligned (128)));


void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const unsigned numX, const unsigned numY, const unsigned numT, PrivGlobs& globs   
            );

void initOperator(  const vector<REAL>& x, 
                    vector<vector<REAL> >& Dxx
                 );

void updateParams(const unsigned g, const REAL alpha, const REAL beta, const REAL nu, PrivGlobs& globs);

void setPayoff(const REAL strike, PrivGlobs& globs );

void tridag(
    const vector<REAL>&   a,   // size [n]
    const vector<REAL>&   b,   // size [n]
    const vector<REAL>&   c,   // size [n]
    const vector<REAL>&   r,   // size [n]
    const int             n,
          vector<REAL>&   u,   // size [n]
          vector<REAL>&   uu   // size [n] temporary
);

void rollback( const unsigned g, PrivGlobs& globs );

REAL   value(   PrivGlobs    globs,
                const REAL s0,
                const REAL strike, 
                const REAL t, 
                const REAL alpha, 
                const REAL nu, 
                const REAL beta,
                const unsigned int numX,
                const unsigned int numY,
                const unsigned int numT
            );

void run_OrigCPU(  
                const unsigned int&   outer,
                const unsigned int&   numX,
                const unsigned int&   numY,
                const unsigned int&   numT,
                const REAL&           s0,
                const REAL&           t, 
                const REAL&           alpha, 
                const REAL&           nu, 
                const REAL&           beta,
                      REAL*           res   // [outer] RESULT
            );

#endif // PROJ_HELPER_FUNS
