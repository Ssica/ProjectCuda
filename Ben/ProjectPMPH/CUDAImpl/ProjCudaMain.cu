REAL   mod_value(REAL* myX,
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
 
    setPayoff_kernel <<< grid, block >>> (strike, d_myX, d_myY, d_myResult);
    
    cudaMemcpy(myResult, d_myResult, numX*numY*sizeof(REAL), cudaMemcpyDeviceToHost);
 
    for(int i = numT-2; i>=0;--i){
        dtInv = 1.0/(myTimeline[i+1]-myTimeline[i]);
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
        
        rollback_implicit_x <<< grid, block >>> (d_a, d_b, d_c, d_myVarX, d_myDxx, dtInv, numY, numX);
        
        TRIDAG_SOLVER <<< grid, block >>> (d_a, d_b, d_c, d_u[i], numX, sgm_sz, d_u[i], d_yy);

        rollback_implicit_y <<< grid, block >>> (d_a, d_b, d_c, d_y, d_u[i], d_v[i], d_myVarY[i], d_myDyy, dtInv, numY, numX);
        
        TRIDAG_SOLVER <<< grid, block >>> (d_a, d_b, d_c, d_y, numY, sgm_sz, d_myResult, d_yy);
        
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
    cudaFree(d_my_Dxx);
    cudaFree(d_my_Dyy);                  
    
    return myResult[myXIndex][myYindex];
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




