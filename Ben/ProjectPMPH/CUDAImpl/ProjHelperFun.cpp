#include "ProjHelperFun.h"

/**************************/
/**** HELPER FUNCTIONS ****/
/**************************/

/**
 * Fills in:
 *   globs.myTimeline  of size [0..numT-1]
 *   globs.myX         of size [0..numX-1]
 *   globs.myY         of size [0..numY-1]
 * and also sets
 *   globs.myXindex and globs.myYindex (both scalars)
 */
void initGrid2(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const unsigned numX, const unsigned numY, const unsigned numT, 
                REAL* myTimeline, unsigned myXindex, REAL* myX, unsigned myYindex, REAL* myY
) {
    for(unsigned i=0;i<numT;++i)
        myTimeline[i] = t*i/(numT-1);

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    myXindex = static_cast<unsigned>(s0/dx) % numX;
    printf("xindex: %u\n", myXindex);

    for(unsigned i=0;i<numX;++i)
        myX[i] = i*dx - myXindex*dx + s0;

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    myYindex = static_cast<unsigned>(numY/2.0);
    printf("yindex: %u\n", myYindex);
    
    for(unsigned i=0;i<numY;++i)
        myY[i] = i*dy - myYindex*dy + logAlpha;
}

/**
 * Fills in:
 *    Dx  [0..n-1][0..3] and 
 *    Dxx [0..n-1][0..3] 
 * Based on the values of x, 
 * Where x's size is n.
 */
void initOperator(  REAL* x, 
                    REAL* Dxx, unsigned numX
) {
	//const unsigned n = x.size();

	REAL dxl, dxu;

	//	lower boundary
	dxl		 =  0.0;
	dxu		 =  x[1] - x[0];
	
	Dxx[0] =  0.0;
	Dxx[0 + 1] =  0.0;
	Dxx[0 + 2] =  0.0;
    Dxx[0 + 3] =  0.0;
	
	//	standard case
	for(unsigned i=1;i<numX-1;i++)
	{
		dxl      = x[i]   - x[i-1];
		dxu      = x[i+1] - x[i];

		Dxx[i * 4 + 0] =  2.0/dxl/(dxl+dxu);
		Dxx[i * 4 + 1] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
		Dxx[i * 4 + 2] =  2.0/dxu/(dxl+dxu);
        Dxx[i * 4 + 3] =  0.0; 
	}

	//	upper boundary
	dxl		   =  x[numX-1] - x[numX-2];
	dxu		   =  0.0;

	Dxx[(numX-1) * 4 + 0] = 0.0;
	Dxx[(numX-1) * 4 + 1] = 0.0;
	Dxx[(numX-1) * 4 + 2] = 0.0;
    Dxx[(numX-1) * 4 + 3] = 0.0;
}
