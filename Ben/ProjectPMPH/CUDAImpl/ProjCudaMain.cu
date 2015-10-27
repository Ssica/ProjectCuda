void   mod_value(REAL* myX,
                 REAL* myY,
                 REAL* myTimeline
                 unsigned myXindex,
                 unsigned myYindex,
                 REAL* myResult,
                 REAL* myVarX,
                 REAL* myVArY,
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
    initOperator(myX, myDxx);
    initOperator(myY, myDyy);
    
    dim3 grid(numY, numX,1);
    dim3 block(T,T,1) //husk at sætte T
    
    REAL* d_myX;
    REAL* d_myY;
    REAL* d_myResult;
    
    cudaMalloc((void**)&d_myX,numX*sizeof(REAL));
    cudaMalloc((void**)&d_myY,numY*sizeof(REAL));
    cudaMalloc((void**)&d_myResult,numX*numY*sizeof(REAL));
    
    cudaMemcpy(d_myX, myX, numX*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myY, myY, numY*sizeof(REAL), cudaMemcpyHostToDevice);
    cudaMemcpy(d_myResult, myResult, numX*sizeof(REAL), cudaMemcpyHostToDevice);
 
    setPayoff_kernel <<< grid, block >>> (strike, d_myX, d_myY, d_myResult);
    
    cudaMemcpy(myResult, d_myResult, numX*numY*sizeof(REAL), cudaMemcpyDeviceToHost);
 
    cudaFree(d_myX);
    cudaFree(d_myY);
    cudaFree(d_myResult);
                      
                      
}
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
    