__device__ __forceinline__ float atomicMin(float *address, float val);    

__device__ __forceinline__ float atomicMax(float *address, float val);
__device__ __forceinline__ double atomicMin(double *address, double val);
__device__ __forceinline__ double atomicMax(double *address, double val);


//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ __forceinline__ double atomicAdd(double *address, double val);
//#endif

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 350)
__device__ __forceinline__ long long atomicMin(long long *address, long long val);
__device__ __forceinline__ long long atomicMax(long long *address, long long val);
#endif


__device__ __forceinline__ float atomicMin(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val < __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__device__ __forceinline__ float atomicMax(float *address, float val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
            break;
    }
    return __int_as_float(ret);
}

__device__ __forceinline__ double atomicMin(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val < __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

// double atomicMax
__device__ __forceinline__ double atomicMax(double *address, double val)
{
    unsigned long long ret = __double_as_longlong(*address);
    while(val > __longlong_as_double(ret))
    {
        unsigned long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, old, __double_as_longlong(val))) == old)
            break;
    }
    return __longlong_as_double(ret);
}

/*
// double atomicMax,xp modify
__device__ __forceinline__ double atomicMax(double *address, double val)
{
	unsigned long long int*address_as_null = (unsigned long long int*)address;
	unsigned long long int assumed;

	assumed = atomicMax(address_as_null, __double_as_longlong(val));
	
	return __longlong_as_double(assumed);
}
*/
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 350)
__device__ __forceinline__ long long atomicMin(long long *address, long long val)
{
    long long ret = *address;
    while(val < ret)
    {
        long long old = ret;
        if((ret = atomicCAS((unsigned long long *)address, (unsigned long long)old, (unsigned long long)val)) == old)
            break;
    }
    return ret;
}

__device__ __forceinline__ long long atomicMax(long long *address, long long val)
{
    long long ret = *address;
    while(val > ret)
    {
        long long old = ret;
        if((ret = (long long)atomicCAS((unsigned long long *)address, (unsigned long long)old, (unsigned long long)val)) == old)
            break;
    }
    return ret;
}
#endif

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 600)
__device__ __forceinline__ double atomicAdd(double *address, double val)
{
    // Doing it all as longlongs cuts one __longlong_as_double from the inner loop
    unsigned long long *ptr = (unsigned long long *)address;
    unsigned long long old, newdbl, ret = *ptr;
    do {
        old = ret;
        newdbl = __double_as_longlong(__longlong_as_double(old)+val);
    } while((ret = atomicCAS(ptr, old, newdbl)) != old);
    
    return __longlong_as_double(ret);
}
//#endif


