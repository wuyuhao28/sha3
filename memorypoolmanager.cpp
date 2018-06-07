/**************************************************
description:内存管理器
author:zhoubin
time:2017-6-28
filename:MemoryManager.cpp
**************************************************/
#include "memorypoolmanager.h"
//#include "common.h"

CMemoryManagerPool * CMemoryManagerPool::instance()
{
	static CMemoryManagerPool *default_hdl = new CMemoryManagerPool();	
	return default_hdl;
}

CMemoryManagerPool::CMemoryManagerPool()
{
	_mapmemorypool.clear();
    _memory_size = 0;
	initFlag = false;
	initOverFlag = false;
	pthread_mutex_init(&memoryMutex, NULL);
}

CMemoryManagerPool::~CMemoryManagerPool()
{
    //_ReleaseMemoryPool(_memorypool);
}


//初始化内存
int CMemoryManagerPool::inital(int num, unsigned long long poolSize, int memoryType)
{
	if (initOverFlag == true)
		return 0;

	pthread_mutex_lock(&memoryMutex);
	if (initFlag == false)
	{
		initFlag = true;
		_mapmemorypool.clear();
		pthread_rwlock_init(&_rwlock_memorypool, NULL);   //初始化读写锁
		_memory_size = poolSize;
		for (int i = 0; i < num; i++)
		{
			printf("[%s %d|Info] i:%d,poolsize:%ld\n", __FILE__, __LINE__, i, poolSize);
			PMEMORYPOOL _memorypool = ::_CreateMemoryPool(i, poolSize, memoryType);
			if (NULL == _memorypool)
			{
				printf("CMemoryManagerPool::inital create memory pool failed. size [%d].\n", poolSize);
				continue;
			}
			_mapmemorypool[i] = _memorypool;
		}

		printf("inital is over \n");
		initOverFlag = true;
	}
	else
	{
		while(initOverFlag == false)
		{
			usleep(10);
		}
		printf("inital already done \n");
	}

	pthread_mutex_unlock(&memoryMutex);
    return 0;
}

int CMemoryManagerPool::clear()
{
	std::map<int, PMEMORYPOOL>::iterator it = _mapmemorypool.begin();
	int num = 0;
	for (; it != _mapmemorypool.end(); it++)
	{
		PMEMORYPOOL _memorypool = (PMEMORYPOOL)it->second;
		cleanMemoryPool(num, _memorypool);
	}
}

//申请内存，type :TYPE_COMMON、TYPE_PERMANENT
void * CMemoryManagerPool::CMalloc(int index, size_t sMemorySize, TYPE_MEMORY type)
{
	if (sMemorySize <= 0)
		return NULL;

    if (sMemorySize > _memory_size)
    {
        //printf("CMemoryManagerPool::CMalloc. can not malloc sMemorySize[%d], max pool size [%d]\n",sMemorySize, _memory_size);
		printf("CMemoryManagerPool::CMalloc. can not malloc sMemorySize[%ld], max pool size [%ld]\n", sMemorySize, _memory_size);	//modify by wyh
        return NULL;
    }

    if (type <= TYPE_IDLE || type > TYPE_PERMANENT)
    {
        printf("CMemoryManagerPool::CMalloc. undefine type [%d]\n",sMemorySize, type);
        return NULL;
    }


	std::map<int, PMEMORYPOOL>::iterator it = _mapmemorypool.find(index);
	if (it == _mapmemorypool.end())
	{
		printf("CMemoryManagerPool::CMalloc. can not find memory pool point.\n");
        return NULL;
	}
	PMEMORYPOOL _memorypool = it->second;
	if (NULL == _memorypool)
    {
        printf("CMemoryManagerPool::CMalloc. can not find memory pool point.\n");
        return NULL;
    }
	
    pthread_rwlock_wrlock(&_rwlock_memorypool);
    void * ret = ::_GetMemory(sMemorySize,_memorypool,type);
    pthread_rwlock_unlock(&_rwlock_memorypool);

	if(ret == NULL)
			printf ("ret is null\n");
	
    return ret;
}

//释放内存
int CMemoryManagerPool::CFree(int index, void * ptrMemoryBlock)
{
    if (NULL == ptrMemoryBlock)
    {
        printf("CMemoryManagerPool::CFree. memory point is null.\n");
        return -1;
    } 

    std::map<int, PMEMORYPOOL>::iterator it = _mapmemorypool.find(index);
	if (it == _mapmemorypool.end())
	{
		printf("CMemoryManagerPool::CMalloc. can not find memory pool point.\n");
        return -1;
	}
	
	PMEMORYPOOL _memorypool = it->second;
	if (NULL == _memorypool)
    {
        printf("CMemoryManagerPool::CMalloc. can not find memory pool point.\n");
        return -1;
    }

    pthread_rwlock_wrlock(&_rwlock_memorypool);
    int ret = ::_FreeMemory(ptrMemoryBlock,_memorypool);
    pthread_rwlock_unlock(&_rwlock_memorypool);

    return ret;
}

//刷新内存
int CMemoryManagerPool::CRefresh(int index, std::list<RefreshPoint> &refreshList)
{
    refreshList.clear();
    std::map<int, PMEMORYPOOL>::iterator it1 = _mapmemorypool.find(index);
	if (it1 == _mapmemorypool.end())
	{
		printf("CMemoryManagerPool::CMalloc. can not find memory pool point.\n");
        return -1;
	}
	
	PMEMORYPOOL _memorypool = it1->second;
	if (NULL == _memorypool)
    {
        printf("CMemoryManagerPool::CMalloc. can not find memory pool point.\n");
        return -1;
    }

    pthread_rwlock_wrlock(&_rwlock_memorypool);
    memory_block* iterator = NULL;
    memory_block* start = _memorypool->pmem_map;
    POOL_WALK(_memorypool, iterator)
    {
        if (iterator->type == TYPE_PERMANENT)
        {
            int memsize = iterator->count * MINUNITSIZE;
            void * ret = ::_GetMemory(memsize,_memorypool,TYPE_PERMANENT);
            if (NULL == ret)
                continue;

            void * before_point = index2addr(_memorypool, iterator-start);
            if ((char *)before_point - (char *)ret >= 0)
            {
                ::_FreeMemory(ret,_memorypool);
                continue;
            }
            else
            {
                CRefreshPoint refreshPoint;
                refreshPoint.refresh_after_point = ret;
                refreshPoint.refresh_before_point = before_point;
                
                memcpy(ret,before_point,memsize);
                //推迟回收
                //::_FreeMemory(before_point,_memorypool);

                refreshList.push_back(refreshPoint);
            }
        }
    }

    
    std::list<CRefreshPoint>::iterator it;
    for (it=refreshList.begin(); it != refreshList.end(); it++)
    {
        CRefreshPoint  point = (CRefreshPoint)*it;
        ::_FreeMemory(point.refresh_before_point,_memorypool);
    }
    pthread_rwlock_unlock(&_rwlock_memorypool);
    return refreshList.size();
}

void CMemoryManagerPool::Info()
{
  /*  if (NULL == _memorypool)
    //{
      //  printf("CMemoryManagerPool::Info. can not find memory pool point.\n");
        //return;
    //}

     pthread_rwlock_rdlock(&_rwlock_memorypool);
     printf("***************************************************************\n");
     printf("pool size[%lld], block size[%d], block count[%d]\n",_memorypool->size,MINUNITSIZE, _memorypool->mem_block_count);
     printf("pool use size[%lld], free block count[%d], chunk count[%d]\n",_memorypool->mem_used_size,_memorypool->mem_map_pool_count, _memorypool->mem_chunk_count);
     printf("***************************************************************\n");
     pthread_rwlock_unlock(&_rwlock_memorypool);
	 */
}


