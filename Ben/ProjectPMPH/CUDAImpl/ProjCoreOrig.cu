__global__ void updateParams_kernel(const unsigned g, const REAL alpha, const REAL beta, const REAL nu,
                             REAL *myX, REAL *myY, REAL *myVarX, REAL *myVarY, REAL *myTimeline, int numX, int numY){
    //already parallelized, no loop dependencies
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    
        
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



