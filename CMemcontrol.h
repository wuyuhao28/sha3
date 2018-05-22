#ifndef _CMEMCONTROL_H_  
#define _CMEMCONTROL_H_  

#include <stdio.h>
#include <string.h>
#include <string>
#include <vector>
#include <pthread.h>
#include "memorypool.h"
#include "memorypoolmanager.h"

class CMemControl 
{
public:
	static CMemControl *instance();
	bool initFlag;
	bool initOverFlag;
	pthread_mutex_t memoryMutex;

public:
	CMemControl();
	~CMemControl();
	void *mem_malloc(size_t t_size);
	void mem_free(void *ptr);
	int run();
private:
	int Create_mem_pool();
private:
	//内存map变量的index
	int m_pool_index;
	CMemoryManagerPool *m_mem_pool;
};

#define cpu_memory_pool CMemControl::instance()

#endif //_CMEMCONTROL_H_ 
