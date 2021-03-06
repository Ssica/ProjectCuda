#include "ProjHelperFun.h"
#include "Constants.h"
#include "TridagKernel.cu.h"

#include <cuda_runtime.h>

const unsigned BLOCK_SIZE = 32;

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
    
    u[j * numY + i] = dtInv * myResult[i * numY + j];
    
    if(i > 0) {
        u[j * numY + i] += 0.5*( 0.5*myVarX[i * numY + j]*myDxx[i * 4 + 0] ) 
                            * myResult[i * numY - 1 + j];
    }
    
    u[j * numY + i] += 0.5*( 0.5*myVarX[i * numY + j]*myDxx[i * 4 + 1] ) 
                            * myResult[i * numY + j];
                            
    if (i < numX - 1) {
        u[j * numY + i] += 0.5*( 0.5*myVarX[i * numY + j]*myDxx[i * 4 + 2] ) 
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
        v[i * numY + j] += 0.5*( 0.5*myVarY[i * numY + j]*myDyy[j * 4 + 0] ) 
                            * myResult[i * numY + j - 1];
    }
    
    v[i * numY + j] += 0.5*( 0.5*myVarY[i * numY + j]*myDyy[j * 4 + 1] ) 
                        * myResult[i * numY + j];                            
    if (i < numX - 1) {
        v[i * numY + j] += 0.5*( 0.5*myVarY[i * numY + j]*myDyy[j * 4 + 2] ) 
                            * myResult[i * numY + j + 1];
    }
    
    u[j * numY + i] += v[i * numY + j];
    
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

    a[j * numY + i] = -0.5*(0.5*myVarX[i * numY + j]*myDxx[i * 4 + 0]);
    b[j * numY + i] = dtInv - 0.5*(0.5*myVarX[i * numY + j]*myDxx[i * 4 + 1]);
    c[j * numY + i] = -0.5*(0.5*myVarX[i * numY + j]*myDxx[i * 4 + 2]);
    
}

__global__ void rollback_implicit_y(REAL *a, 
                                    REAL *b, 
                                    REAL *c, 
                                    REAL *y, 
                                    REAL *u, 
                                    REAL *v, 
                                    REAL *myVarY, 
                                    REAL* myDyy, 
                                    REAL dtInv,
                                    int numY, 
                                    int numX){

    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;                                

    if (i >= numX && j >= numY) return;
/*
    a[j * numY + i] = -0.5*(0.5*myVarY[i * numY + j]*myDyy[i * numY + 0]);
    b[j * numY + i] = dtInv - 0.5*(0.5*myVarY[i * numY + j]*myDyy[i * numY + 1]);
    c[j * numY + i] = -0.5*(0.5*myVarY[i * numY + j]*myDyy[i * numY + 2]);
    */
    a[i * numY + j] = -0.5*(0.5*myVarY[i * numY + j]*myDyy[j * 4 + 0]);
    b[i * numY + j] = dtInv - 0.5*(0.5*myVarY[i * numY + j]*myDyy[i * 4 + 1]);
    c[i * numY + j] = -0.5*(0.5*myVarY[i * numY + j]*myDyy[i * 4 + 2]);
    y[i * numY + j] = dtInv*u[j * numY + i] - 0.5*v[i * numY + j];
}

REAL  values(    REAL* myX,
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
                 const unsigned numX,
                 const unsigned numY,
                 const unsigned numT,
                 const unsigned outer){
                     
    initGrid2(s0, alpha, nu, t, numX, numY, numT, myTimeline, myXindex, myX, myYindex, myY);
    initOperator(myX, myDxx, numX);
    initOperator(myY, myDyy, numY);
    
    dim3 grid(BLOCK_SIZE, BLOCK_SIZE);
    dim3 block(numX / BLOCK_SIZE, numY / BLOCK_SIZE, outer);
    dim3 block_switch(numX / BLOCK_SIZE, numY / BLOCK_SIZE, outer);
    
    unsigned numZ = max(numX,numY);
    
    REAL* u = (REAL*) malloc(numX*numY*sizeof(REAL)); //done
    REAL* v = (REAL*) malloc(numX*numY*sizeof(REAL)); //done
    REAL* a = (REAL*) malloc(numX*numY*sizeof(REAL)); //array expand because of tridag
    REAL* b = (REAL*) malloc(numX*numY*sizeof(REAL)); //array expand because of tridag
    REAL* c = (REAL*) malloc(numX*numY*sizeof(REAL)); //array expand because of tridag
    REAL* yy = (REAL*) malloc(numZ*sizeof(REAL)); //done
    REAL* y = (REAL*) malloc(numY*numX*sizeof(REAL)); //array expand because of tridag
    
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
    cudaMalloc((void**)&d_myVarX,numX*numY*sizeof(REAL));
    cudaMalloc((void**)&d_myVarY,numY*numX*sizeof(REAL));
    cudaMalloc((void**)&d_myTimeline,numT*sizeof(REAL));
    cudaMalloc((void**)&d_u,numX*numY*sizeof(REAL));
    cudaMalloc((void**)&d_v,numX*numY*sizeof(REAL));
    cudaMalloc((void**)&d_myDxx,numX*4*sizeof(REAL));
    cudaMalloc((void**)&d_myDyy,numY*4*sizeof(REAL));
    
    cudaMemcpy(d_myX, myX, numX*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myY, myY, numY*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myResult, myResult, numX*numY*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myVarX, myVarX, numX*numY*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myVarY, myVarY, numY*numX*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myTimeline, myTimeline, numT*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, u, numX*numY*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_v, v, numX*numY*sizeof(REAL), cudaMemcpyHostToDevice);
    
    setPayoff_kernel<<<block,grid>>>(strike, d_myX, d_myY, d_myResult, numX, numY);
    
    for(int i = numT-2; i>=0;--i){
        REAL dtInv = 1.0/(myTimeline[i+1]-myTimeline[i]);
        
        updateParams_kernel<<<block,grid>>> (i, alpha, beta, nu, d_myX, d_myY, d_myVarX, d_myVarY, d_myTimeline, numX, numY);
        
        rollback_explicit_x <<<block,grid>>>(d_u, d_myResult, myVarX, myDxx, dtInv, numX, numY); 
        rollback_implicit_x<<<block, grid>>>(d_a, d_b, d_c, d_myVarX, d_myDxx, dtInv, numY, numX);
        cudaThreadSynchronize();
        
        //change block
        rollback_explicit_y <<<block_switch,grid>>>(d_v, d_u, d_myResult, d_myVarY, d_myDyy, dtInv, numX, numY);
        cudaThreadSynchronize();
        
        cudaMalloc((void**)&d_a, numX*numY*sizeof(REAL));
        cudaMalloc((void**)&d_b, numX*numY*sizeof(REAL));
        cudaMalloc((void**)&d_c, numX*numY*sizeof(REAL));
        cudaMalloc((void**)&d_yy, numZ*sizeof(REAL));
        cudaMalloc((void**)&d_y, numY*numX*sizeof(REAL)); //array expanded with numX
        
        cudaMemcpy(d_a, a, numX*numY*sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, numX*numY*sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_c, c, numX*numY*sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_yy, yy, numZ*sizeof(REAL), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y, y, numY*numX*sizeof(REAL), cudaMemcpyHostToDevice);
        
        //change block
        
        tridagCUDAWrapper(numY,d_a,d_b,d_c,d_y,numX * numY * outer, numY, d_u, d_yy);
        //TRIDAG_SOLVER<<<block, grid>>>(d_a, d_b, d_c, d_u, numX, SGM_SIZE, d_u, d_yy);
        
        //change block
        rollback_implicit_y<<<block_switch, grid>>>(d_a, d_b, d_c, d_y, d_u, d_v, d_myVarY, d_myDyy, dtInv, numY, numX);
        cudaThreadSynchronize();
        
        tridagCUDAWrapper(numY,d_a,d_b,d_c,d_y,numX * numY * outer, numY, d_myResult, d_yy);
        //TRIDAG_SOLVER<<<block, grid>>>(d_a, d_b, d_c, d_y, numY, SGM_SIZE, d_myResult, d_yy);
        
        cudaMemcpy(myResult, d_myResult, numX*numY*sizeof(REAL), cudaMemcpyDeviceToHost);
        
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
        cudaFree(d_yy);
        cudaFree(d_y);
        
    }
    printf("numY: %d\n",numY);
    printf("numM: %d\n",numY*numX*sizeof(REAL));
    cudaFree(d_myX);
    cudaFree(d_myY);
    cudaFree(d_myResult);
    cudaFree(d_myVarX);
    cudaFree(d_myVarY);
    cudaFree(d_myTimeline);
    cudaFree(d_u);
    cudaFree(d_myDxx);
    cudaFree(d_myDyy);                  
    
    return myResult[5000000];
    
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
    PrivGlobs2   globs(numX, numY, numT);
    
    for( unsigned i = 0; i < outer; ++ i ) {
                
        strike = 0.001*i;
        res[i] = values(globs.myX, globs.myY, globs.myTimeline, 
                        globs.myXindex, globs.myYindex, globs.myResult,
                        globs.myVarX, globs.myVarY, globs.myDxx, 
                        globs.myDyy, s0, strike, t,
                        alpha, nu,    beta,
                        numX,  numY,  numT,outer);
    }
}






