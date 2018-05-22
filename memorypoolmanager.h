/**************************************************
description:ÄÚ´æ¹ÜÀíÆ÷
author:zhoubin
time:2017-6-28
filename:MemoryManager.h
**************************************************/
#ifndef _MEMORY_MANAGER_H_
#define _MEMORY_MANAGER_H_

#include <stdio.h>
#include <string.h>
#include <string>
#include "memorypool.h"
#include <pthread.h>
#include <list>
#include <map>
#include <unistd.h>

#ifdef __cplusplus
extern "C" {
#endif 

#define  DEFAULT_MEMORY_POOL_SIZE 4*1024*1024


typedef struct RefreshPoint
{
    void * refresh_before_point;
    void * refresh_after_point;

    RefreshPoint& operator=(RefreshPoint& value)
    {
        refresh_before_point = value.refresh_before_point;
        refresh_after_point = value.refresh_after_point;
        return *this;
    }
}CRefreshPoint;

/**
 * use Memory Manager pool assigning, make Algorithm complexity o(1).
 * @remark:
 * use message pool here, reduce memory alloc/release, fragment.
 *
 */
class CMemoryManagerPool {

    public:
        static CMemoryManagerPool *instance();
		bool initFlag;
		bool initOverFlag;
		pthread_mutex_t memoryMutex;

    public:
        CMemoryManagerPool();
        ~CMemoryManagerPool();

        int inital(int num, unsigned long long poolSize = DEFAULT_MEMORY_POOL_SIZE, int memoryType = TYPE_GPU);

		int clear();
        
		void * CMalloc(int index, size_t sMemorySize, TYPE_MEMORY type = TYPE_COMMON);	//modify by wyh
        //void * CMalloc(int index, int sMemorySize, TYPE_MEMORY type = TYPE_COMMON);

        int CFree(int index, void * ptrMemoryBlock);

        int CRefresh(int index, std::list<CRefreshPoint> &refreshList);

        void Info();

    private:
        std::map<int, PMEMORYPOOL> _mapmemorypool;
        pthread_rwlock_t _rwlock_memorypool;
        unsigned long long _memory_size;

};

#define memory_pool CMemoryManagerPool::instance()

#ifdef __cplusplus
}
#endif 

#endif


