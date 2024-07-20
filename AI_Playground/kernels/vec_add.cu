__global__ void vec_add(float *A, float *B, float *C)
{
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
	for(int i = 0; i < 1024; i++)
	{
	    C[idx] += (A[idx] + B[idx]);
	}
}