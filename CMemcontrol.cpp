#include "CMemcontrol.h"

#define CPU_MEMORY		268435456LL		//256MB
#define CPU_POOL_INDEX	1

CMemControl * CMemControl::instance()
{
	static CMemControl *default_hdl = new CMemControl();
	return default_hdl;
}

CMemControl::CMemControl()
{
	m_pool_index = CPU_POOL_INDEX;
	m_mem_pool = NULL;
}

CMemControl::~CMemControl()
{
	delete m_mem_pool;
}

int CMemControl::Create_mem_pool()
{
	m_mem_pool = new CMemoryManagerPool();

	m_mem_pool->inital(m_pool_index, CPU_MEMORY, TYPE_CPU);

	printf("[%s %d|Info] inital is over \n",__FILE__,__LINE__);

	return 0;
}

void *CMemControl::mem_malloc(size_t t_size)
{
	return m_mem_pool->CMalloc(m_pool_index-1, t_size);
}

void CMemControl::mem_free(void *ptr)
{
	m_mem_pool->CFree(m_pool_index-1, ptr);
}

int CMemControl::run()
{
	Create_mem_pool();
}

