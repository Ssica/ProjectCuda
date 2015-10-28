#include "Constants.h"
#include "TridagKernel.cu.h"

__global__ void updateParams_kernel(const unsigned g, const REAL alpha, const REAL beta, const REAL nu,
                             REAL *myX, REAL *myY, REAL *myVarX, REAL *myVarY, REAL *myTimeline, int numX, int numY){
    //already parallelized, no loop dependencies
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= numX && j >= numY) { return; }
        
    myVarX[i * numY + j] = exp(2.0 * (beta * log(myX[i]) + myY[j] - 0.5*nu*nu*myTimeline[g]));
    myVarY[i * numY + j] = exp(2.0 * (alpha * log(myX[i]) + myY[j] - 0.5*nu*nu*myTimeline[g]));
  }

__global__ void setPayoff_kernel(const REAL strike, REAL *myX, REAL *myY, REAL *myResult) {
    //already parallelized
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= numX && j >= numY) { return; }
    
    REAL payoff = MAX(myX[i] - strike, (REAL)0.0);
    myResult[i * numY + j] = payoff;
        
}

__global__ void rollback_explicit_x(REAL *u, 
                                    REAL* myResult, 
                                    REAL* myVarX, 
                                    REAL* myDxx,
                                    REAL dtInv,
                                    int numX, 
                                    int numY) {
    
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;                                
                                    
    if (i >= numX && j >= numY) return;
    
    u[j * numX + i] = dtInv * myResult[i * numY + j];
    
    if(i > 0) {
        u[j * numX + i] += 0.5*( 0.5*myVarX[i * numY + j]*myDxx[i * numY + 0] ) 
                            * myResult[i * numY - 1 + j];
    }
    
    u[j * numX + i] += 0.5*( 0.5*myVarX[i * numY + j]*myDxx[i * numY + 1] ) 
                            * myResult[i * numY + j];
                            
    íf (i < numX - 1) {
        u[j * numX + i] += 0.5*( 0.5*myVarX[i * numY + j]*myDxx[i * numY + 2] ) 
                            * myResult[i * numY + 1 + j];
    }
    
}

__global__ void rollback_explicit_y(REAL *v,
                                    REAL *u,
                                    REAL *myResult, 
                                    REAL *myVarY, 
                                    REAL *myDyy,
                                    REAL dtInv,
                                    int numX, 
                                    int numY) {
    
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;                                
                                    
    if (i >= numX && j >= numY) return;
    
    v[i * numY + j] = 0.0;
    
    if(j > 0) {
        v[i * numY + j] += 0.5*( 0.5*myVarY[i * numY + j]*myDyy[j * numX + 0] ) 
                            * myResult[i * numY + j - 1];
    }
    
    v[i * numY + j] += 0.5*( 0.5*myVarY[i * numY + j]*myDyy[j * numX + 1] ) 
                        * myResult[i * numY + j];                            
    íf (i < numX - 1) {
        v[i * numY + j] += 0.5*( 0.5*myVarY[i * numY + j]*myDyy[j * numX + 2] ) 
                            * myResult[i * numY + j + 1];
    }
    
    u[j * numX + i] += v[i * numY + j];
    
}

__global__ void rollback_implicit_x(REAL *a, 
                                    REAL *b, 
                                    REAL *c, 
                                    REAL *myVarX, 
                                    REAL* myDxx, 
                                    REAL* dtInv,
                                    int numY, 
                                    int numX){

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;                                

    if (i >= numX && j >= numY) return; 

    a[j * numY + i] = -0.5*(0.5*myVarX[i * numY + j]*myDxx[i * numY + 0]);
    b[j * numY + i] = dtInv - 0.5*(0.5*myVarX[i * numY + j]*myDxx[i * numY + 1]);
    c[j * numY + i] = -0.5*(0.5*myVarX[i * numY + j]*myDxx[i * numY + 2]);
    
}

__global__ void rollback_implicit_y(REAL *a, 
                                    REAL *b, 
                                    REAL *c, 
                                    REAL *y, 
                                    REAL *u, 
                                    REAL *v, 
                                    REAL *myVarY, 
                                    REAL* myDyy, 
                                    REAL* dtInv,
                                    int numY, 
                                    int numX){

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;                                

    if (i >= numX && j >= numY) return;

    a[i * numX + j] = -0.5*(0.5*myVarY[i * numY + j]*myDxx[j * numY + 0]);
    b[i * numX + j] = dtInv - 0.5*(0.5*myVarX[i * numY + j]*myDxx[i * numY + 1]);
    c[i * numX + j] = -0.5*(0.5*myVarX[i * numY + j]*myDxx[i * numY + 2]);
    y[i * numX + j] = dtInv*u[j * numX + i] - 0.5*v[i * numY + j];
}
/*
REAL   value(   REAL* myX,
                REAL* myY,
                REAL* myTimeline,
                REAL* myXindex,
                REAL* myYindex,
                REAL* myResult,
                REAL* myVarX,
                REAL* myVarY,
                REAL* myDxx;
                REAL* myDyy,
                const REAL s0,
                const REAL strike, 
                const REAL t, 
                const REAL alpha, 
                const REAL nu, 
                const REAL beta,
                const unsigned int numX,
                const unsigned int numY,
                const unsigned int numT
) {	
    initGrid(s0,alpha,nu,t, numX, numY, numT,
                REAL* myX,
                REAL* myY,
                REAL* myTimeline,
                REAL* myXindex,
                REAL* myYindex,
                REAL* myResult,
                REAL* myVarX,
                REAL* myVarY,
                REAL* myDxx,
                REAL* myDyy);
    initOperator(myX,myDxx);
    initOperator(myY,myDyy);
    
    
    
    setPayoff(strike, globs);
    for(int i = globs.myTimeline.size()-2;i>=0;--i)
    {
        updateParams(i,alpha,beta,nu,globs);
        rollback(i, globs);
    }

    return globs.myResult[globs.myXindex][globs.myYindex];
}
*/
/*
void initGrid(  const REAL s0, const REAL alpha, const REAL nu,const REAL t, 
                const unsigned numX, const unsigned numY, const unsigned numT,    
                REAL* myX,
                REAL* myY,
                REAL* myTimeline,
                REAL* myXindex,
                REAL* myYindex,
                REAL* myResult,
                REAL* myVarX,
                REAL* myVarY,
                REAL* myDxx;
                REAL* myDyy,
) {
    for(unsigned i=0;i<numT;++i)
        myTimeline[i] = t*i/(numT-1);

    const REAL stdX = 20.0*alpha*s0*sqrt(t);
    const REAL dx = stdX/numX;
    myXindex = static_cast<unsigned>(s0/dx) % numX;

    for(unsigned i=0;i<numX;++i)
        myX[i] = i*dx - myXindex*dx + s0;

    const REAL stdY = 10.0*nu*sqrt(t);
    const REAL dy = stdY/numY;
    const REAL logAlpha = log(alpha);
    myYindex = static_cast<unsigned>(numY/2.0);

    for(unsigned i=0;i<numY;++i)
        myY[i] = i*dy - myYindex*dy + logAlpha;
}

__global__ void run_CUDA(  
                const unsigned int&   outer,
                const unsigned int&   numX,
                const unsigned int&   numY,
                const unsigned int&   numT,
                const REAL&           s0,
                const REAL&           t, 
                const REAL&           alpha, 
                const REAL&           nu, 
                const REAL&           beta,
                      REAL*           res,
                      REAL*           strike) {   // [outer] RESULT

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
                      
    if (i >= outer) return;
    
    strike[i] = 0.001*i;
    
    res[i] = value()
    
}
*/





