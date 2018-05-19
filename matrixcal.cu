#include "matrixcal.h"

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"
//#include "device_functions.h"
//#include "algri.h"
//#include <stdio.h>
//#include "atomic.h"

cudaError_t matrixMul(Mat256x256i8& sourceMatrix, const Mat256x256i8* tmpMatrix, const AlgriMatList* matList_int8, uint8_t *sequence, int8_t* threadID, uint8_t *tmpMat);
cudaError_t matrixMul_CuBlas(Mat256x256i8& sourceMatrix, const Mat256x256i8* tmpMatrix, AlgriMatList* matList_int8, uint8_t *sequence);
//__global__ void mulKernel(Mat256x256i8& sourceMatrix, Mat256x256i8* tmpMatrix, Mat256x256i8* seqMatrix);

#define LOOP_COUNT		2
#define SEQUENCE_COUNT	32

#define BLOCK_SIZE		256
#define THREAD_SIZE		256

// tmp * seq[index] -> source
//////////////////////////////////single kernel loop////////////////////////////////
//__global__ void mulKernel(int8_t* sourceMatrix, int8_t* tmpMatrix, int8_t* seqMatrix, uint8_t index)
//{
//	//int tid = threadIdx.x + (int)blockIdx.x * blockDim.x;
//	int curRow = threadIdx.x;
//	int curCol = blockIdx.x;
//
//	//if (tid < 65536){
//		int tmp = 0;
//		
//		for (int i = 0; i < BLOCK_SIZE; i++){
//			//tmp += tmpMatrix->d[curRow][i] * seqMatrix->d[i][curCol];
//			tmp += tmpMatrix[curRow * BLOCK_SIZE + i] * seqMatrix[index * 65536 + i * BLOCK_SIZE + curCol];
//		}
//
//		//sourceMatrix[curRow * BLOCK_SIZE + curCol] = tmp;
//		//multi finish
//
//		//extra calculate
//		sourceMatrix[curRow * BLOCK_SIZE + curCol] = ((tmp & 0xFF) + ((tmp >> 8) & 0xFF)) & 0xFF;
//	//}
//}
///////////////////////////////////////////////////////////////////////////////////////////


//__global__ void mulKernel(int8_t* sourceMatrix, int8_t* tmpMatrix, int8_t* seqMatrix, double* atomic)
//{
//	int tid = threadIdx.x + (int)blockIdx.x * blockDim.x;
//	int curRow = threadIdx.x;
//	int curCol = blockIdx.x;
//
//	if (tid < 65536){
//
//		for (int i = 0; i < LOOP_COUNT; i++)
//		{
//			for (int j = 0; j < SEQUENCE_COUNT; j++)
//			{
//				atomic[0] = 0;
//				atomic[1] = 0;
//				int tmp = 0;
//				for (int k = 0; k < 256; k++){
//					tmp += tmpMatrix[curRow * BLOCK_SIZE + k] * seqMatrix[j * 65536 + k * BLOCK_SIZE + curCol];
//				}
//				//extra calculate
//				sourceMatrix[curRow * BLOCK_SIZE + curCol] = ((tmp & 0xFF) + ((tmp >> 8) & 0xFF)) & 0xFF;
//				__syncthreads();
//				if (curRow == 0)
//				{
//					atomicAdd(&atomic[0], 1);
//				}
//				while (atomic[0] < 256) {}
//
//				//source -> tmp
//				tmpMatrix[curRow * BLOCK_SIZE + curCol] = sourceMatrix[curRow * BLOCK_SIZE + curCol];
//				__syncthreads();
//				if (curRow == 0)
//				{
//					atomicAdd(&atomic[1], 1);
//				}
//				while (atomic[1] < 256) {}
//
//				//atomicAdd(&atomic[1], 1);
//				//__iAtomicAdd(&atomic[1], 1);
//				//while (atomic[1] < 65535) {}
//			}
//		}
//	}
//
//}


#define TILE_WIDTH 16
__global__ void matrixMultiplyShared(int8_t *A, int8_t *B_all, int8_t *C, uint8_t index)
{
	//@@ Insert code to implement matrix multiplication here
	//@@ You have to use shared memory for this MP

	__shared__ int8_t sharedM[TILE_WIDTH][TILE_WIDTH];
	__shared__ int8_t sharedN[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = by*TILE_WIDTH + ty;
	int col = bx*TILE_WIDTH + tx;
	int v = 0;

	int8_t *B = B_all + index * BLOCK_SIZE * BLOCK_SIZE;

	for (int i = 0; i < (int)(ceil((float)BLOCK_SIZE / TILE_WIDTH)); i++)
	{
		if (i*TILE_WIDTH + tx < BLOCK_SIZE && row < BLOCK_SIZE)
			sharedM[ty][tx] = A[row*BLOCK_SIZE + i*TILE_WIDTH + tx];
		else
			sharedM[ty][tx] = 0.0;

		if (i*TILE_WIDTH + ty < BLOCK_SIZE && col < BLOCK_SIZE)
			sharedN[ty][tx] = B[(i*TILE_WIDTH + ty)*BLOCK_SIZE + col];
		else
			sharedN[ty][tx] = 0.0;
		__syncthreads();

		for (int j = 0; j < TILE_WIDTH; j++)
			v += sharedM[ty][j] * sharedN[j][tx];
		__syncthreads();
	}

	if (row < BLOCK_SIZE && col < BLOCK_SIZE)
	{
		//extra calculate
		v = ((v & 0xFF) + ((v >> 8) & 0xFF)) & 0xFF;

		C[row*BLOCK_SIZE + col] = v;
	}
}

__global__ void Matrix_Mul(int8_t *md, int8_t *nd, int8_t *pd, uint8_t index)
{
	int bx, by, tx, ty;
	bx = blockIdx.x;
	by = blockIdx.y;
	tx = threadIdx.x;
	ty = threadIdx.y;
	int mulResult = 0;
	int8_t *B = nd + index * 1 * BLOCK_SIZE * BLOCK_SIZE;		//sizeof(uint8_t) = 1
	for (int i = 0; i < gridDim.x; ++i)
	{
		__shared__ int8_t d_m[TILE_WIDTH][TILE_WIDTH];
		__shared__ int8_t d_n[TILE_WIDTH][TILE_WIDTH];
		d_m[ty][tx] = *(md + (by * blockDim.y + ty) * BLOCK_SIZE + i * blockDim.x + tx);
		d_n[ty][tx] = *(B + (i * blockDim.y + ty) * BLOCK_SIZE + bx * blockDim.x + tx);
		__syncthreads();
		for (int j = 0; j < blockDim.x; ++j)
		{
			mulResult += d_m[ty][j] * d_n[j][tx];
		}
		__syncthreads();
	}
	pd[(by*blockDim.y + ty)*BLOCK_SIZE + bx*blockDim.x + tx] = ((mulResult & 0xFF) + ((mulResult >> 8) & 0xFF)) & 0xFF;
}

__global__ void matrixExtraCal(int *sourceMatrix, int8_t *tmpMatrix)
{
	//int tid = threadIdx.x + (int)blockIdx.x * blockDim.x;
	//int curRow = tid / BLOCK_SIZE;
	//int curCol = tid % BLOCK_SIZE;

	//if (tid < BLOCK_SIZE * THREAD_SIZE)
	//{
	//	int tmp = sourceMatrix[curRow * BLOCK_SIZE + curCol];
	//	//int8_t tmp2 = tmpMatrix[curRow * BLOCK_SIZE + curCol];
	//	tmpMatrix[curCol * BLOCK_SIZE + curRow] = ((tmp & 0xFF) + ((tmp >> 8) & 0xFF)) & 0xFF;
	//	//sourceMatrix[curRow * BLOCK_SIZE + curCol] = tmp2;
	//}
}


cudaError_t matrixMul(Mat256x256i8& sourceMatrix, const Mat256x256i8* tmpMatrix, int8_t* matList, 
	uint8_t *sequence, uint32_t threadID, int8_t *tmpMulMat)
{
	double start, end, start_t, end_t;
	cudaError_t cudaStatus;
	start = GetMillsec();
	cudaStatus = cudaSetDevice(threadID);

	int alpha = 1;
	int beta = 0;

	int matrixSize = sizeof(int8_t) * 256 * 256;
	//int8_t *source;
	int *source;
	int8_t *tmp, *tmpSource;
	//source = (int8_t *)memory_pool->CMalloc(threadID, matrixSize);
	source = (int *)memory_pool->CMalloc(threadID, sizeof(int) * 256 * 256);
	tmp = (int8_t *)memory_pool->CMalloc(threadID, matrixSize);

	cudaStatus = cudaMemcpy(tmp, tmpMatrix->d, matrixSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);

	end = GetMillsec();
	printf("\n\t kernel copy time: %lfms\n", (end - start));

	//////////////////////////////////single kernel loop////////////////////////////////
	start = GetMillsec();
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		for (int j = 0; j < SEQUENCE_COUNT; j++)
		{
			start_t = GetMillsec();
			//mulKernel << <BLOCK_SIZE, THREAD_SIZE >> >(source, tmp, matList, sequence[j]);

			//dim3 DimGrid(ceil(BLOCK_SIZE / TILE_WIDTH), ceil(BLOCK_SIZE / TILE_WIDTH), 1);
			//dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
			//matrixMultiplyShared << < DimGrid, DimBlock >> >(tmp, matList, source, sequence[j]);
			//cudaDeviceSynchronize();

			//dim3 grid(256 / TILE_WIDTH, 256 / TILE_WIDTH);
			//dim3 blocks(TILE_WIDTH, TILE_WIDTH);
			//Matrix_Mul << <grid, blocks >> >(tmp, matList, source, sequence[j]);
			//cudaDeviceSynchronize();

			cublasStatus_t cublasSatus = cublasGemmEx(g_handle[threadID], CUBLAS_OP_T, CUBLAS_OP_T, 256, 256, 256,
				(void *)&alpha, (void *)tmp, CUDA_R_8I, 256,
				(void *)(matList + sequence[j] * matrixSize), CUDA_R_8I, 256,
				(void *)&beta, (void *)source, CUDA_R_32I, 256,
				CUDA_R_32I, CUBLAS_GEMM_DFALT);
			if (cublasSatus != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasGemmEx error!, j: %d cublasError: %d\n", j, cublasSatus);
			}
			end_t = GetMillsec();
			if (i == 0 && j == 0)
			{
				printf("\t first kernel time1: %lfms\n", (end_t - start_t));
			}

			matrixExtraCal << <256, 256 >> >(source, tmp);
			cudaDeviceSynchronize();

			if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
			{
				printf("[%s:%d]|Error|Cuda kernel error: %s|%d\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus), cudaStatus);
				return cudaStatus;
			}

			/*tmpSource = tmp;
			tmp = source;
			source = tmpSource;*/
			
			end_t = GetMillsec();
			if (i ==0 && j == 0)
			{
				printf("\t first kernel time2: %lfms\n", (end_t - start_t));
			}
		}
	}
	
	////////////////////////////////////////////////////////////////////////
	end = GetMillsec();
	printf("\t kernel time: %lfms\n", (end - start));
	/*start = clock();
	mulKernel << <BLOCK_SIZE, THREAD_SIZE >> >(source, tmp, matList, atomicGPU);
	cudaDeviceSynchronize();

	if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
	{
		printf("[%s:%d]|Error|Cuda kernel error: %s|%d\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus), cudaStatus);
		return cudaStatus;
	}*/
	start = GetMillsec();
	cudaStatus = cudaMemcpy(sourceMatrix.d, tmp, matrixSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);

	//

	memory_pool->CFree(threadID, tmp);
	memory_pool->CFree(threadID, source);

	end = GetMillsec();
	printf("\t kernel tail time: %lfms\n", (end - start));

	return cudaStatus;
}



//
__global__ void ReadKernel(int8_t *tmp, int8_t *matList, float *f_tmp, float *f_matList)
{
	int curRow = threadIdx.x;
	int curCol = blockIdx.x;

	f_tmp[curRow * BLOCK_SIZE + curCol] = (float)tmp[curRow * BLOCK_SIZE + curCol];
	for (int i = 0; i < SEQUENCE_COUNT; i++)
	{
		f_matList[i * BLOCK_SIZE * BLOCK_SIZE + curRow * BLOCK_SIZE + curCol] = (float)f_matList[i * BLOCK_SIZE * BLOCK_SIZE + curRow * BLOCK_SIZE + curCol];
	}
}

__global__ void matrixMultiplyShared_float(float *A, float *B_all, float *C, uint8_t index)
{
	//@@ Insert code to implement matrix multiplication here
	//@@ You have to use shared memory for this MP

	__shared__ float sharedM[TILE_WIDTH][TILE_WIDTH];
	__shared__ float sharedN[TILE_WIDTH][TILE_WIDTH];
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	int row = by*TILE_WIDTH + ty;
	int col = bx*TILE_WIDTH + tx;
	float v = 0;

	float *B = B_all + index * BLOCK_SIZE * BLOCK_SIZE;

	for (int i = 0; i < (int)(ceil((float)BLOCK_SIZE / TILE_WIDTH)); i++)
	{
		if (i*TILE_WIDTH + tx < BLOCK_SIZE && row < BLOCK_SIZE)
			sharedM[ty][tx] = A[row*BLOCK_SIZE + i*TILE_WIDTH + tx];
		else
			sharedM[ty][tx] = 0.0;

		if (i*TILE_WIDTH + ty < BLOCK_SIZE && col < BLOCK_SIZE)
			sharedN[ty][tx] = B[(i*TILE_WIDTH + ty)*BLOCK_SIZE + col];
		else
			sharedN[ty][tx] = 0.0;
		__syncthreads();

		for (int j = 0; j < TILE_WIDTH; j++)
			v += sharedM[ty][j] * sharedN[j][tx];
		__syncthreads();
	}

	int v_tmp = (int)v;
	if (row < BLOCK_SIZE && col < BLOCK_SIZE)
	{
		//extra calculate
		v_tmp = ((v_tmp & 0xFF) + ((v_tmp >> 8) & 0xFF)) & 0xFF;

		C[row*BLOCK_SIZE + col] = (float)v_tmp;
	}
}

cudaError_t matrixMul_CuBlas(Mat256x256i8& sourceMatrix, const Mat256x256i8* tmpMatrix, AlgriMatList* matList_int8, uint8_t *sequence)
{
	double start, end;
	cudaError_t cudaStatus;
	start = GetMillsec();
	cudaStatus = cudaSetDevice(0);

	float alpha = 1;
	float beta = 0;

	int matrixSize_int8 = sizeof(int8_t) * 256 * 256;
	int8_t *tmp, *matList;
	cudaStatus = cudaMalloc((void **)&tmp, matrixSize_int8);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
	cudaStatus = cudaMalloc((void **)&matList, matrixSize_int8 * SEQUENCE_COUNT);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);

	cudaStatus = cudaMemcpy(tmp, tmpMatrix->d, matrixSize_int8, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
	int offset = 0;
	for (int i = 0; i < SEQUENCE_COUNT; i++)
	{
		cudaStatus = cudaMemcpy(matList + offset, matList_int8->at(sequence[i]).d, matrixSize_int8, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
			printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
		offset += matrixSize_int8;
	}

	int matrixSize = sizeof(float) * 256 * 256;
	float *f_source, *f_tmp, *f_matList, *tmpSource;
	cudaStatus = cudaMalloc((void **)&f_source, matrixSize);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
	cudaStatus = cudaMemset(f_source, 0, matrixSize);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
	cudaStatus = cudaMalloc((void **)&f_tmp, matrixSize);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
	cudaStatus = cudaMalloc((void **)&f_matList, matrixSize * SEQUENCE_COUNT);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);

	ReadKernel << <BLOCK_SIZE, THREAD_SIZE >> >(tmp, matList, f_tmp, f_matList);
	cudaDeviceSynchronize();
	if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
	{
		printf("[%s:%d]|Error|Cuda kernel error: %s|%d\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus), cudaStatus);
		return cudaStatus;
	}

	end = GetMillsec();
	printf("\t kernel copy time: %lfs\n", (end - start));

	//////////////////////////////////single kernel loop////////////////////////////////
	start = GetMillsec();
	cublasStatus_t cublasError;
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		for (int j = 0; j < SEQUENCE_COUNT; j++)
		{
			/*cublasHandle_t handle;
			cublasCreate(&handle);
			cublasError = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 256, 256, 256,
				&alpha, f_tmp, 256, f_matList + j * matrixSize, 256, &beta, f_source, 256);
			if (cublasError != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasSgemm_v2 error!, j: %d cublasError: %d\n", j, cublasError);
			}*/
			//cublasGemmEx();
			dim3 DimGrid(ceil(BLOCK_SIZE / TILE_WIDTH), ceil(BLOCK_SIZE / TILE_WIDTH), 1);
			dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);
			matrixMultiplyShared_float << < DimGrid, DimBlock >> >(f_tmp, f_matList, f_source, j);
			cudaDeviceSynchronize();
			if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
			{
				printf("[%s:%d]|Error|Cuda kernel error: %s|%d\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus), cudaStatus);
				return cudaStatus;
			}

			tmpSource = f_tmp;
			f_tmp = f_source;
			f_source = tmpSource;

			/*if (i == 0 && j == 10)
			{
				end = clock();
				printf("\t 10 single kernel cal time: %lf s\n", (double)(end - start) / CLOCKS_PER_SEC);
			}*/
		}
	}
	////////////////////////////////////////////////////////////////////////
	float *hostSouce = (float *)malloc(matrixSize);
	memset(hostSouce, 0, matrixSize);
	cudaStatus = cudaMemcpy(hostSouce, f_tmp, matrixSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);

	for (int i = 0; i < 256; i++)
	{
		for (int j = 0; j < 256; j++)
		{
			sourceMatrix.d[i][j] = (int8_t)hostSouce[j * 256 + i];
		}
	}

	end = GetMillsec();
	printf("\t kernel time: %lfs\n", (end - start));

	cudaFree(tmp);
	cudaFree(matList);
	cudaFree(f_source);
	cudaFree(f_tmp);
	cudaFree(f_matList);

	free(hostSouce);

	return cudaStatus;
}



//typedef struct st_matrixMulThreadArg{
//	
//
//}stMatrixMulThreadArg, *pstMatrixMulThreadArg;

//void* matrixMul_Thread(void *arg)
//{
//	uint8_t sequence[128];
//	rhash_sha3_256_init(ctx);
//	rhash_sha3_update(ctx, msg + (len*k / 4), len / 4);
//	rhash_sha3_final(ctx, sequence);
//	Mat256x256i8 *tmp = new Mat256x256i8;
//	tmp->toIdentityMatrix();
//
//	//GPU process
//	cudaError_t cudaStatus = matrixMul(*mat, tmp, matList, sequence, threadID, tmpMulMat);
//	if (cudaStatus != cudaSuccess){
//		printf("ERROR: cuda error during GPU process.\n");
//	}
//
//	res[k].copyFrom(*mat);
//	delete tmp;
//}

void iter(
	const uint8_t *msg,
	uint32_t len,
	uint8_t result[32],
	uint32_t threadID) {
	Mat256x256i8 *res = new Mat256x256i8[4];
	Mat256x256i8 *mat = new Mat256x256i8;
	sha3_ctx *ctx = (sha3_ctx*)calloc(1, sizeof(*ctx));
	
	double start, end;
	start = GetMillsec();

	cudaError_t cudaStatus;
	int8_t* matList = (int8_t *)memory_pool->CMalloc(threadID, sizeof(int8_t) * 256 * 256 * 256);
	int8_t* tmpMulMat = (int8_t *)memory_pool->CMalloc(threadID, sizeof(int8_t) * 256 * 256 * 256);
	cudaStatus = cudaMemcpy(matList, matList_int8->matVec, sizeof(int8_t) * 256 * 256 * 256, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);

	for (int k = 0; k < 4; k++) {
		uint8_t sequence[128];
		rhash_sha3_256_init(ctx);
		rhash_sha3_update(ctx, msg + (len*k / 4), len / 4);
		rhash_sha3_final(ctx, sequence);
		Mat256x256i8 *tmp = new Mat256x256i8;
		tmp->toIdentityMatrix();

		//GPU process
		cudaStatus = matrixMul(*mat, tmp, matList, sequence, threadID, tmpMulMat);
		//cudaStatus = matrixMul_CuBlas(*mat, tmp, matList_int8, sequence);
		if (cudaStatus != cudaSuccess){
			printf("ERROR: cuda error during GPU process.\n");
		}

		res[k].copyFrom(*mat);
		delete tmp;
	}

	/////////////////////////////////
	/*pthread_t matrixMulThread[4];
	pstMatrixMulThreadArg matrixMulThreadArg = new stMatrixMulThreadArg[4]();
	for (int i = 0; i < 4; i++)
	{
		if (pthread_create(&matrixMulThread[i], NULL, matrixMul_Thread, (void *)&matrixMulThreadArg[i]) != 0)
		{
			printf("ERROR: calculateThread create failed.\n");
			return;
		}
	}

	for (int i = 0; i < 4; i++)
	{
		if (pthread_join(matrixMulThread[i], NULL) != 0)
		{
			printf("ERROR: calculateThread join failed.\n");
			return;
		}
	}*/

	memory_pool->CFree(threadID, matList);
	memory_pool->CFree(threadID, tmpMulMat);
	end = GetMillsec();
	std::cout << "\t\tTime for getting MulMatix: "
		<< (end - start) << "ms"
		<< std::endl;

	mat->add(res[0], res[1]);
	mat->add(*mat, res[2]);
	mat->add(*mat, res[3]);

	Arr256x64i32 arr(*mat);
	arr.reduceFNV();
	rhash_sha3_256_init(ctx);
	rhash_sha3_update(ctx, arr.d0RawPtr(), 256);
	rhash_sha3_final(ctx, result);
	delete mat;
	delete[] res;
	free(ctx);
}



double GetMillsec()
{
	double t1;
	struct timeval starttime;
	gettimeofday(&starttime, NULL);
	//sprintf(temp," send msg time:%f ms\n",(starttime.tv_sec*1000+starttime.tv_usec*0.001));
	t1 = starttime.tv_sec * 1000 + starttime.tv_usec*0.001;
	return t1;
}
