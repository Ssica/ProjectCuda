#include "ProjHelperFun.h"
#include "Constants.h"
#include "TridagKernel.cu.h"

#include <cuda_runtime.h>

const unsigned int SGM_SIZE = 8;

const unsigned int T = 32;

__global__ void updateParams_kernel(const unsigned g, const REAL alpha, const REAL beta, const REAL nu,
                             REAL *myX, REAL *myY, REAL *myVarX, REAL *myVarY, REAL *myTimeline, int numX, int numY){
    //already parallelized, no loop dependencies
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i >= numX && j >= numY) { return; }
        
    myVarX[i * numY + j] = exp(2.0 * (beta * log(myX[i]) + myY[j] - 0.5*nu*nu*myTimeline[g]));
    myVarY[i * numY + j] = exp(2.0 * (alpha * log(myX[i]) + myY[j] - 0.5*nu*nu*myTimeline[g]));
  }

__global__ void setPayoff_kernel(const REAL strike, REAL *myX, REAL *myY, REAL *myResult, int numX, int numY) {
    //already parallelized
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= numX && j >= numY) { return; }
    
    REAL payoff = max(myX[i] - strike, (REAL)0.0);
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
                            
    if (i < numX - 1) {
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
    if (i < numX - 1) {
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
                                    REAL dtInv,
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

    a[i * numX + j] = -0.5*(0.5*myVarY[i * numY + j]*myDyy[j * numY + 0]);
    b[i * numX + j] = dtInv - 0.5*(0.5*myVarY[i * numY + j]*myDyy[i * numY + 1]);
    c[i * numX + j] = -0.5*(0.5*myVarY[i * numY + j]*myDyy[i * numY + 2]);
    y[i * numX + j] = dtInv*u[j * numX + i] - 0.5*v[i * numY + j];
}

__global__ void initOperator_kernel(REAL *x, REAL *Dxx, const unsigned numX) {
    unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;

    if (i >= numX)
        return;

    if (i == 0 || i == numX-1) {
        Dxx[i * numX + 0] =  0.0;
        Dxx[i * numX + 1] =  0.0;
        Dxx[i * numX + 2] =  0.0;
        Dxx[i * numX + 3] =  0.0;
    } else {
        REAL dxl = x[i] - x[i-1];
        REAL dxu = x[i+1] - x[i];

        Dxx[i * numX + 0] =  2.0/dxl/(dxl+dxu);
        Dxx[i * numX + 1] = -2.0*(1.0/dxl + 1.0/dxu)/(dxl+dxu);
        Dxx[i * numX + 2] =  2.0/dxu/(dxl+dxu);
        Dxx[i * numX + 3] =  0.0;
    }
}




REAL  value(     REAL* myX,
                 REAL* myY,
                 REAL* myTimeline,
                 unsigned myXindex,
                 unsigned myYindex,
                 REAL* myResult,
                 REAL* myVarX,
                 REAL* myVarY,
                 REAL* myDxx,
                 REAL* myDyy,
                 const REAL s0,
                 const REAL strike,
                 const REAL t,
                 const REAL alpha,
                 const REAL nu,
                 const REAL beta,
                 const unsigned int numX,
                 const unsigned int numY,
                 const unsigned int numT){

    initGrid(s0, alpha, nu, t, numX, numY, numT, myTimeline, myXindex, myX, myYindex, myY);
    initOperator(myX, myDxx, numX);
    initOperator(myY, myDyy, numY);
    
    dim3 grid(numY, numX,1);
    dim3 block(T,T,1); //husk at sætte T
    
    unsigned numZ = max(numX,numY);
    
    REAL* u = (REAL*) malloc(numX*numY*numT*sizeof(REAL));
    REAL* v = (REAL*) malloc(numX*numY*numT*sizeof(REAL));
    REAL* a = (REAL*) malloc(numZ*numZ*sizeof(REAL));
    REAL* b = (REAL*) malloc(numZ*numZ*sizeof(REAL));
    REAL* c = (REAL*) malloc(numZ*numZ*sizeof(REAL));
    REAL* yy = (REAL*) malloc(numZ*numZ*sizeof(REAL)); 
    REAL* y = (REAL*) malloc(numY*numX*sizeof(REAL));
    
    REAL* d_myX;
    REAL* d_myY;
    REAL* d_myResult;
    REAL* d_myVarX;
    REAL* d_myVarY;
    REAL* d_myTimeline;
    REAL* d_u;
    REAL* d_v;
    REAL* d_myDxx;
    REAL* d_myDyy;
    REAL* d_a;
    REAL* d_b;
    REAL* d_c;
    REAL* d_yy;
    REAL* d_y;
    
    cudaMalloc((void**)&d_myX,numX*sizeof(REAL));
    cudaMalloc((void**)&d_myY,numY*sizeof(REAL));
    cudaMalloc((void**)&d_myResult,numX*numY*sizeof(REAL));
    cudaMalloc((void**)&d_myVarX,numX*numT*sizeof(REAL)); //array expand by numT
    cudaMalloc((void**)&d_myVarY,numY*numT*sizeof(REAL)); //array expand by numT
    cudaMalloc((void**)&d_myTimeline,numT*sizeof(REAL));
    cudaMalloc((void**)&d_u,numX*numY*numT*sizeof(REAL)); //array expanded with numT
    cudaMalloc((void**)&d_v,numX*numY*numT*sizeof(REAL)); //array expanded with numT
    cudaMalloc((void**)&d_myDxx,numX*4*sizeof(REAL));
    cudaMalloc((void**)&d_myDyy,numY*4*sizeof(REAL));
    
    cudaMemcpy(d_myX, myX, numX*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myY, myY, numY*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myResult, myResult, numX*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myVarX, myVarX, numX*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myVarY, myVarY, numY*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myTimeline, myTimeline, numT*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, numX*numY*numT*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, numX*numY*numT*sizeof(REAL), cudaMemcpyHostToDevice);
 
    setPayoff_kernel<<< grid, block >>>(strike, d_myX, d_myY, d_myResult);
    
    cudaMemcpy(myResult, d_myResult, numX*numY*sizeof(REAL), cudaMemcpyDeviceToHost);
 
    for(int i = numT-2; i>=0;--i){
        REAL dtInv = 1.0/(myTimeline[i+1]-myTimeline[i]);
        updateParams_kernel <<< grid, block >>> (i, alpha, beta, nu, d_myX, d_myY, d_myVarX[i], d_myVarY[i], d_myTimeline, numX, numY);
        rollback_explicit_x <<< grid, block >>> (d_u[i], d_myResult, myVarX, myDxx, dtInv, numX, numY); 
        rollback_explicit_y <<< grid, block >>> (d_v[i], d_u[i], d_myResult, d_myVarY[i], d_myDyy, dtInv, numX, numY);
        
        cudaMalloc((void**)&d_a, numZ*numZ*sizeof(REAL));
        cudaMalloc((void**)&d_b, numZ*numZ*sizeof(REAL));
        cudaMalloc((void**)&d_c, numZ*numZ*sizeof(REAL));
        cudaMalloc((void**)&d_yy, numZ*numZ*sizeof(REAL));
        cudaMalloc((void**)&d_y, numY*numX*sizeof(REAL)); //array expanded with numX
        
        cudaMemcpy(d_a, a, numZ*numZ*sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, numZ*numZ*sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, c, numZ*numZ*sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_yy, yy, numZ*numZ*sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, numY*numX*sizeof(REAL), cudaMemcpyHostToDevice);
        
        rollback_implicit_x<<<grid, block>>>(d_a, d_b, d_c, d_myVarX, d_myDxx, dtInv, numY, numX);
        
        TRIDAG_SOLVER<<<grid, block>>>(d_a, d_b, d_c, d_u[i], numX, SGM_SIZE, d_u[i], d_yy);

        rollback_implicit_y<<<grid, block>>>(d_a, d_b, d_c, d_y, d_u[i], d_v[i], d_myVarY[i], d_myDyy, dtInv, numY, numX);
        
        TRIDAG_SOLVER<<<grid, block>>>(d_a, d_b, d_c, d_y, numY, SGM_SIZE, d_myResult, d_yy);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFree(d_yy);
        cudaFree(d_y);
    }
 
    cudaFree(d_myX);
    cudaFree(d_myY);
    cudaFree(d_myResult);
    cudaFree(d_myVarX);
    cudaFree(d_myVarY);
    cudaFree(d_myTimeline);
    cudaFree(d_u);
    cudaFree(d_myDxx);
    cudaFree(d_myDyy);                  
    
    return myResult[myXindex][myYindex];
//call InitGrid, std funktion

//call InitOperator, std funktion
//call InitOperator, std funktion

//call kernel SetPayoff(strike globs)

//for(int i = globs.myTimeline.size()-2;i>=0;--i)
    //call kernel updateParams
    //call kernel rollback_explicit_x;
    //call kernel rollback_explicit_y;
    //call kernel rollback_implicit_x;
    //call kernel TRIDAG_SOLVER
    //call kernel rollback_implicit_y;
    //call kernel TRIDAG_SOLVER
    
    
}

void   run_OrigCPU(  
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
) {
    
    REAL strike;
    
    for( unsigned i = 0; i < outer; ++ i ) {
        
        PrivGlobs    globs(numX, numY, numT);
        
        
        
        strike = 0.001*i;
        res[i] = value( globs.myX, globs.myY, globs.myTimeline, 
                        globs.myXindex, globs.myYindex, globs.myResult,
                        globs.myVarX, globs.myVarY, globs.myDxx, 
                        globs.myDyy, s0, strike, t,
                        alpha, nu,    beta,
                        numX,  numY,  numT );
    }
}






