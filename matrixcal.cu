#include "matrixcal.h"

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

//#define TILE_WIDTH 16
//__global__ void matrixMultiplyShared(int8_t *A, int8_t *B_all, int8_t *C, uint8_t index)
//{
//	//@@ Insert code to implement matrix multiplication here
//	//@@ You have to use shared memory for this MP
//
//	__shared__ int8_t sharedM[TILE_WIDTH][TILE_WIDTH];
//	__shared__ int8_t sharedN[TILE_WIDTH][TILE_WIDTH];
//	int bx = blockIdx.x;
//	int by = blockIdx.y;
//	int tx = threadIdx.x;
//	int ty = threadIdx.y;
//	int row = by*TILE_WIDTH + ty;
//	int col = bx*TILE_WIDTH + tx;
//	int v = 0;
//
//	int8_t *B = B_all + index * BLOCK_SIZE * BLOCK_SIZE;
//
//	for (int i = 0; i < (int)(ceil((float)BLOCK_SIZE / TILE_WIDTH)); i++)
//	{
//		if (i*TILE_WIDTH + tx < BLOCK_SIZE && row < BLOCK_SIZE)
//			sharedM[ty][tx] = A[row*BLOCK_SIZE + i*TILE_WIDTH + tx];
//		else
//			sharedM[ty][tx] = 0.0;
//
//		if (i*TILE_WIDTH + ty < BLOCK_SIZE && col < BLOCK_SIZE)
//			sharedN[ty][tx] = B[(i*TILE_WIDTH + ty)*BLOCK_SIZE + col];
//		else
//			sharedN[ty][tx] = 0.0;
//		__syncthreads();
//
//		for (int j = 0; j < TILE_WIDTH; j++)
//			v += sharedM[ty][j] * sharedN[j][tx];
//		__syncthreads();
//	}
//
//	if (row < BLOCK_SIZE && col < BLOCK_SIZE)
//	{
//		//extra calculate
//		v = ((v & 0xFF) + ((v >> 8) & 0xFF)) & 0xFF;
//
//		C[row*BLOCK_SIZE + col] = v;
//	}
//}
//
//__global__ void Matrix_Mul(int8_t *md, int8_t *nd, int8_t *pd, uint8_t index)
//{
//	int bx, by, tx, ty;
//	bx = blockIdx.x;
//	by = blockIdx.y;
//	tx = threadIdx.x;
//	ty = threadIdx.y;
//	int mulResult = 0;
//	int8_t *B = nd + index * 1 * BLOCK_SIZE * BLOCK_SIZE;		//sizeof(uint8_t) = 1
//	for (int i = 0; i < gridDim.x; ++i)
//	{
//		__shared__ int8_t d_m[TILE_WIDTH][TILE_WIDTH];
//		__shared__ int8_t d_n[TILE_WIDTH][TILE_WIDTH];
//		d_m[ty][tx] = *(md + (by * blockDim.y + ty) * BLOCK_SIZE + i * blockDim.x + tx);
//		d_n[ty][tx] = *(B + (i * blockDim.y + ty) * BLOCK_SIZE + bx * blockDim.x + tx);
//		__syncthreads();
//		for (int j = 0; j < blockDim.x; ++j)
//		{
//			mulResult += d_m[ty][j] * d_n[j][tx];
//		}
//		__syncthreads();
//	}
//	pd[(by*blockDim.y + ty)*BLOCK_SIZE + bx*blockDim.x + tx] = ((mulResult & 0xFF) + ((mulResult >> 8) & 0xFF)) & 0xFF;
//}
//



__global__ void matrixExtraCal(int *sourceMatrix, int8_t *tmpMatrix)
{
	int tid = threadIdx.x + (int)blockIdx.x * blockDim.x;
	int curRow = tid / BLOCK_SIZE;
	int curCol = tid % BLOCK_SIZE;

	if (tid < BLOCK_SIZE * THREAD_SIZE)
	{
		int tmp = sourceMatrix[curRow * BLOCK_SIZE + curCol];
		//int8_t tmp2 = tmpMatrix[curRow * BLOCK_SIZE + curCol];
		tmpMatrix[curCol * BLOCK_SIZE + curRow] = ((tmp & 0xFF) + ((tmp >> 8) & 0xFF)) & 0xFF;
		//sourceMatrix[curRow * BLOCK_SIZE + curCol] = tmp2;
	}
}


cudaError_t matrixMul(Mat256x256i8& sourceMatrix, const Mat256x256i8* tmpMatrix, int8_t* matList, 
	uint8_t *sequence, uint32_t threadID)
{
	cudaSetDevice(threadID);
	cudaError_t cudaStatus;

	int alpha = 1;
	int beta = 0;

	int matrixSize = sizeof(int8_t) * 256 * 256;
	int *source;
	int8_t *tmp, *tmpSource;
	source = (int *)memory_pool->CMalloc(threadID, sizeof(int) * 256 * 256);
	tmp = (int8_t *)memory_pool->CMalloc(threadID, matrixSize);

	cudaStatus = cudaMemcpy(tmp, tmpMatrix->d, matrixSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);

	cublasStatus_t cublasSatus;
	//////////////////////////////////single kernel loop////////////////////////////////
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		for (int j = 0; j < SEQUENCE_COUNT; j++)
		{

			cublasSatus = cublasGemmEx(g_handle[threadID], CUBLAS_OP_T, CUBLAS_OP_T, 256, 256, 256,
				(void *)&alpha, (void *)tmp, CUDA_R_8I, 256,
				(void *)(matList + sequence[j] * matrixSize), CUDA_R_8I, 256,
				(void *)&beta, (void *)source, CUDA_R_32I, 256,
				CUDA_R_32I, CUBLAS_GEMM_DFALT);
			
			if (cublasSatus != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasGemmEx error!, j: %d cublasError: %d\n", j, cublasSatus);
			}

			matrixExtraCal << <256, 256 >> >(source, tmp);
			usleep(1);
			cudaDeviceSynchronize();

			if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
			{
				printf("[%s:%d]|Error|Cuda kernel error: %s|%d\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus), cudaStatus);
				return cudaStatus;
			}
		}
	}

	cudaStatus = cudaMemcpy(sourceMatrix.d, tmp, matrixSize, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);

	memory_pool->CFree(threadID, tmp);
	memory_pool->CFree(threadID, source);

	//t2 = GetMillsec();
	//printf("\t kernel total time: %lfms\n", (t2 - t1));
	return cudaStatus;
}

__global__ void arrProcess(int8_t *res, uint32_t *d_arr, int8_t *mat)
{
	int curRow = blockIdx.x;
	int curCol = threadIdx.x;

	int res_tmp = (int)*(res + BLOCK_SIZE * THREAD_SIZE + curRow * BLOCK_SIZE + curCol) + (int)*(res + curRow * BLOCK_SIZE + curCol); //res[0] + res[1]
	int8_t mat_tmp = (res_tmp & 0xFF);
	res_tmp = (int)mat_tmp + (int)*(res + BLOCK_SIZE * THREAD_SIZE * 2 + curRow * BLOCK_SIZE + curCol);	//mat + res[2]
	mat_tmp = (res_tmp & 0xFF);
	res_tmp = (int)mat_tmp + (int)*(res + BLOCK_SIZE * THREAD_SIZE * 3 + curRow * BLOCK_SIZE + curCol);	//mat + res[3]
	mat_tmp = (res_tmp & 0xFF);
	*(mat + curRow * BLOCK_SIZE + curCol) = mat_tmp;
	__syncthreads();

	if (curCol < 64)
	{
		uint32_t d;
		d = ((uint32_t(uint8_t(*(mat + curRow * BLOCK_SIZE + curCol + 192)))) << 24) |
			((uint32_t(uint8_t(*(mat + curRow * BLOCK_SIZE + curCol + 128)))) << 16) |
			((uint32_t(uint8_t(*(mat + curRow * BLOCK_SIZE + curCol + 64)))) << 8) |
			((uint32_t(uint8_t(*(mat + curRow * BLOCK_SIZE + curCol)))) << 0);
		d_arr[curRow * BLOCK_SIZE + curCol] = d;
	}
}

typedef struct st_matrixMulThreadArg{
	bool updateFlag;
	int threadID;
	int k;
	uint8_t *msg;
	uint32_t len;
	//sha3_ctx *ctx;
	//Mat256x256i8 *res;
	int8_t* res;
	int8_t* device_matList;
	uint8_t *sequence;
}stMatrixMulThreadArg, *pstMatrixMulThreadArg;

void* matrixMul_Thread(void *arg)
{
	pstMatrixMulThreadArg matrixMulThreadArg = (pstMatrixMulThreadArg)arg;
	//Mat256x256i8 *res = matrixMulThreadArg->res;
	int8_t *res = matrixMulThreadArg->res;
	int8_t* device_matList = matrixMulThreadArg->device_matList;
	int threadID = matrixMulThreadArg->threadID; 
	//uint8_t *sequence = matrixMulThreadArg->sequence;
	int k = matrixMulThreadArg->k;

	if (matrixMulThreadArg->updateFlag == false)
	{
		printf("updateFlag is false.\n");
		sha3_ctx *ctx = (sha3_ctx *)cpu_memory_pool->mem_malloc(sizeof(*ctx));
		memset(ctx, 0, sizeof(*ctx));
		//uint8_t *sequence = (uint8_t *)cpu_memory_pool->mem_malloc(sizeof(uint8_t) * 128);
		//memset(sequence, 0, sizeof(uint8_t) * 128);
		rhash_sha3_256_init(ctx);
		rhash_sha3_update(ctx, matrixMulThreadArg->msg + (matrixMulThreadArg->len * k / 4), matrixMulThreadArg->len / 4);
		//rhash_sha3_final(ctx, sequence);
		rhash_sha3_final(ctx, g_sequence[k]);
		cpu_memory_pool->mem_free(ctx);
	}

	Mat256x256i8 *tmp = (Mat256x256i8 *)cpu_memory_pool->mem_malloc(sizeof(Mat256x256i8));
	tmp->toIdentityMatrix();

	int alpha = 1;
	int beta = 0;

	int matrixSize = sizeof(int8_t) * 256 * 256;
	int *source;
	int8_t *tmpMatrix;
	source = (int *)memory_pool->CMalloc(threadID, sizeof(int) * 256 * 256);
	tmpMatrix = (int8_t *)memory_pool->CMalloc(threadID, matrixSize);

	cudaError_t cudaStatus = cudaSetDevice(threadID);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
	cudaStatus = cudaMemcpy(tmpMatrix, tmp->d, matrixSize, cudaMemcpyHostToDevice);
	//cudaStatus = cudaMemcpy(res, tmp->d, matrixSize, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
	for (int i = 0; i < LOOP_COUNT; i++)
	{
		for (int j = 0; j < SEQUENCE_COUNT; j++)
		{
			cublasStatus_t cublasSatus = cublasGemmEx(g_handle[threadID], CUBLAS_OP_T, CUBLAS_OP_T, 256, 256, 256,
				(void *)&alpha, (void *)tmpMatrix, CUDA_R_8I, 256,
				//(void *)&alpha, (void *)res, CUDA_R_8I, 256,
				(void *)(device_matList + g_sequence[k][j] * matrixSize), CUDA_R_8I, 256,
				(void *)&beta, (void *)source, CUDA_R_32I, 256,
				CUDA_R_32I, CUBLAS_GEMM_DFALT);

			if (cublasSatus != CUBLAS_STATUS_SUCCESS)
			{
				printf("cublasGemmEx error!, j: %d cublasError: %d\n", j, cublasSatus);
			}

			matrixExtraCal << <256, 256 >> >(source, tmpMatrix);
			//matrixExtraCal << <256, 256 >> >(source, res);
			cudaDeviceSynchronize();

			if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
			{
				printf("[%s:%d]|Error|Cuda kernel error: %s|%d\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus), cudaStatus);
			}
		}
	}

	//cudaStatus = cudaMemcpy(res, tmpMatrix, matrixSize, cudaMemcpyDeviceToHost);
	cudaStatus = cudaMemcpy(res, tmpMatrix, matrixSize, cudaMemcpyDeviceToDevice);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);

	memory_pool->CFree(threadID, tmpMatrix);
	memory_pool->CFree(threadID, source);

	//delete tmp;
	cpu_memory_pool->mem_free(tmp);
	//cpu_memory_pool->mem_free(sequence);
}

void iter(
	uint8_t *msg,
	uint8_t *seed,
	uint32_t len,
	uint8_t result[32],
	uint32_t threadID){
	
	memory_pool->inital(DEVICENUM, DEVICEMEMORY);
	cpu_memory_pool->run();

	cudaError_t cudaStatus;
	cudaSetDevice(threadID);
	int8_t* device_matList;
	
	double start_t, end_t;
	//start_t = GetMillsec();
	//add mutex?
	if (memcmp(seed, g_seed, 32) == 0)
	{
		//printf("seed alread exist.\n");
		g_seed_update = true;
	}
	else
	{
		printf("seed changed.\n");
		Words32 extSeed = extSeedCreate(seed);
		memset(matList_int8->matVec, 0, sizeof(Mat256x256i8) * 256);
		for (int i = 0; i<256; i++) {
			matList_int8->matVec[i].toIdentityMatrix();
		}
		matList_int8->init(extSeed);
		memcpy(g_seed, seed, 32);
		cudaStatus = cudaMemcpy(g_device_matList[threadID], matList_int8->matVec, sizeof(int8_t) * 256 * 256 * 256, cudaMemcpyHostToDevice);
		if (cudaStatus != cudaSuccess)
			printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
		g_seed_update = false;
	}
	device_matList = g_device_matList[threadID];

	//////////////////////////////////////////////////////////////////////////////////////////
	//single thread process

	//Mat256x256i8 *res = new Mat256x256i8[4];
	//Mat256x256i8 *mat = new Mat256x256i8;
	//sha3_ctx *ctx = (sha3_ctx*)calloc(1, sizeof(*ctx));
	////memset(ctx, 0, sizeof(*ctx));
	//end_t = GetMillsec();
	//printf("iter prepare time: %lf\n", end_t - start_t);

	//for (int k = 0; k < 4; k++) {
	//	uint8_t sequence[128];
	//	rhash_sha3_256_init(ctx);
	//	rhash_sha3_update(ctx, msg + (len*k / 4), len / 4);
	//	rhash_sha3_final(ctx, sequence);
	//	Mat256x256i8 *tmp = new Mat256x256i8;
	//	tmp->toIdentityMatrix();

	//	//GPU process
	//	/*cudaStatus = matrixMul(*mat, tmp, matList, sequence, threadID);
	//	if (cudaStatus != cudaSuccess){
	//		printf("ERROR: cuda error during GPU process.\n");
	//	}*/

	//	//cudaSetDevice(threadID);

	//	int alpha = 1;
	//	int beta = 0;

	//	int matrixSize = sizeof(int8_t) * 256 * 256;
	//	int *source;
	//	int8_t *tmpMatrix;
	//	source = (int *)memory_pool->CMalloc(threadID, sizeof(int) * 256 * 256);
	//	tmpMatrix = (int8_t *)memory_pool->CMalloc(threadID, matrixSize);

	//	cudaStatus = cudaMemcpy(tmpMatrix, tmp->d, matrixSize, cudaMemcpyHostToDevice);
	//	if (cudaStatus != cudaSuccess)
	//		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
	//	for (int i = 0; i < LOOP_COUNT; i++)
	//	{
	//		for (int j = 0; j < SEQUENCE_COUNT; j++)
	//		{
	//			cublasStatus_t cublasSatus = cublasGemmEx(g_handle[threadID], CUBLAS_OP_T, CUBLAS_OP_T, 256, 256, 256,
	//				(void *)&alpha, (void *)tmpMatrix, CUDA_R_8I, 256,
	//				(void *)(device_matList + sequence[j] * matrixSize), CUDA_R_8I, 256,
	//				(void *)&beta, (void *)source, CUDA_R_32I, 256,
	//				CUDA_R_32I, CUBLAS_GEMM_DFALT);

	//			if (cublasSatus != CUBLAS_STATUS_SUCCESS)
	//			{
	//				printf("cublasGemmEx error!, j: %d cublasError: %d\n", j, cublasSatus);
	//			}

	//			matrixExtraCal << <256, 256 >> >(source, tmpMatrix);
	//			cudaDeviceSynchronize();

	//			if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
	//			{
	//				printf("[%s:%d]|Error|Cuda kernel error: %s|%d\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus), cudaStatus);
	//			}
	//		}
	//	}

	//	cudaStatus = cudaMemcpy(mat->d, tmpMatrix, matrixSize, cudaMemcpyDeviceToHost);
	//	if (cudaStatus != cudaSuccess)
	//		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);

	//	memory_pool->CFree(threadID, tmpMatrix);
	//	memory_pool->CFree(threadID, source);

	//	res[k].copyFrom(*mat);
	//	delete tmp;
	//}
	//memory_pool->CFree(threadID, device_matList);

	//mat->add(res[0], res[1]);
	//mat->add(*mat, res[2]);
	//mat->add(*mat, res[3]);;
	//Arr256x64i32 arr(*mat);
	//arr.reduceFNV();
	//rhash_sha3_256_init(ctx);
	//rhash_sha3_update(ctx, arr.d0RawPtr(), 256);
	//rhash_sha3_final(ctx, result);
	//delete mat;
	//delete[] res;
	//free(ctx);
	///////////////////////////////////////////////////////////////////////////////////

	////////////////////////////////////////////////////////////////////////////////////
	//multeThread process 
	start_t = GetMillsec();
	//Mat256x256i8 *res = new Mat256x256i8[4];
	//Mat256x256i8 *mat = new Mat256x256i8;
	//sha3_ctx *ctx = (sha3_ctx*)calloc(1, sizeof(*ctx));
	//uint8_t **sequence = (uint8_t **)malloc(sizeof(uint8_t *) * 4);
	//uint8_t **sequence = (uint8_t **)cpu_memory_pool->mem_malloc(sizeof(uint8_t *) * 4);

	//Mat256x256i8 *res = (Mat256x256i8 *)cpu_memory_pool->mem_malloc(sizeof(Mat256x256i8) * 4);
	int8_t *res = (int8_t *)memory_pool->CMalloc(threadID, sizeof(int8_t) * 256 * 256 * 4);

	sha3_ctx *ctx = (sha3_ctx *)cpu_memory_pool->mem_malloc(sizeof(*ctx));
	memset(ctx, 0, sizeof(*ctx));

	pthread_t matrixMulThread[4];
	//pstMatrixMulThreadArg matrixMulThreadArg = new stMatrixMulThreadArg[4]();
	pstMatrixMulThreadArg matrixMulThreadArg = (pstMatrixMulThreadArg)cpu_memory_pool->mem_malloc(sizeof(stMatrixMulThreadArg) * 4);
	for (int i = 0; i < 4; i++)
	{
		//sequence[i] = (uint8_t *)malloc(sizeof(uint8_t) * 128);
		//sequence[i] = (uint8_t *)cpu_memory_pool->mem_malloc(sizeof(uint8_t) * 128);
		//memset(sequence[i], 0, sizeof(uint8_t) * 128);
		//rhash_sha3_256_init(ctx);
		//rhash_sha3_update(ctx, msg + (len * i / 4), len / 4);
		//rhash_sha3_final(ctx, sequence[i]);
		matrixMulThreadArg[i].updateFlag = g_seed_update;
		matrixMulThreadArg[i].threadID = threadID;
		matrixMulThreadArg[i].k = i;
		matrixMulThreadArg[i].msg = msg;
		matrixMulThreadArg[i].len = len;
		matrixMulThreadArg[i].res = res + i * sizeof(int8_t) * 256 * 256;
		matrixMulThreadArg[i].device_matList = device_matList;
		//matrixMulThreadArg[i].sequence = sequence[i];

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
		//free(sequence[i]);
		//cpu_memory_pool->mem_free(sequence[i]);
	}
	//free(sequence);
	//cpu_memory_pool->mem_free(sequence);
	memory_pool->CFree(threadID, device_matList);

	end_t = GetMillsec();
	printf("iter multi porcess time: %lf\n", end_t - start_t);

	/*Mat256x256i8 *mat = (Mat256x256i8 *)cpu_memory_pool->mem_malloc(sizeof(Mat256x256i8));
	mat->add(res[0], res[1]);
	mat->add(*mat, res[2]);
	mat->add(*mat, res[3]);
	Arr256x64i32 arr(*mat);
	arr.reduceFNV();*/

	//GPU arr process
	uint32_t *d_arr = (uint32_t *)memory_pool->CMalloc(threadID, sizeof(uint32_t) * 256 * 64);
	int8_t *mat = (int8_t *)memory_pool->CMalloc(threadID, sizeof(int8_t) * 256 * 256);
	uint32_t arr[256][64];
	arrProcess << <256, 256 >> >(res, d_arr, mat);
	cudaDeviceSynchronize();
	if ((cudaStatus = cudaGetLastError()) != cudaSuccess)
	{
		printf("[%s:%d]|Error|Cuda kernel error: %s|%d\n", __FILE__, __LINE__, cudaGetErrorString(cudaStatus), cudaStatus);
	}
	cudaStatus = cudaMemcpy(arr, d_arr, sizeof(uint32_t) * 256 * 64, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess)
		printf("[%s:%d]Cuda failed, error code:%d.\n", __FILE__, __LINE__, cudaStatus);
	//arr.reduceFNV();
	for (int k = 256; k > 1; k = k / 2) {
		for (int j = 0; j < k / 2; j++) {
			for (int i = 0; i < 64; i++) {
				arr[j][i] = FNV(arr[j][i], arr[j + k / 2][i]);
			}
		}
	}

	rhash_sha3_256_init(ctx);
	//rhash_sha3_update(ctx, arr.d0RawPtr(), 256);
	rhash_sha3_update(ctx, (uint8_t*)(arr[0]), 256);
	rhash_sha3_final(ctx, result);
	//delete mat;
	//delete[] res;
	//free(ctx);
	//cpu_memory_pool->mem_free(mat);
	//cpu_memory_pool->mem_free(res);
	memory_pool->CFree(threadID, mat);
	memory_pool->CFree(threadID, res);
	cpu_memory_pool->mem_free(ctx);

	/////////////////////////////////////////////////////////////////////////////////////////////
}

static void init_seed(Words32 &seed, uint32_t _seed[32])
{
	for (int i = 0; i < 16; i++)
		seed.lo.w[i] = _seed[i];
	for (int i = 0; i < 16; i++)
		seed.hi.w[i] = _seed[16 + i];
}


//add by wyh, create extseed from seed
Words32 extSeedCreate(uint8_t *seed)
{
	uint32_t exted[32];
	extend(exted, seed);
	Words32 extSeed;
	init_seed(extSeed, exted);

	return extSeed;
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