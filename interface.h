#ifndef INTERFACE_H_
#define INTERFACE_H_

#ifdef __cplusplus
extern "C" {
#endif
	#include <stdint.h>
	
	int get_hash_init();
	int get_hash(uint8_t msg[32], uint8_t seed[32],uint32_t *deviceids,uint8_t target[32],char* jobid,uint8_t* len,uint64_t* nonce);
	void get_rets();
	int devicecount();
	int get_ret(uint8_t *msg,uint8_t* result,uint8_t* target,char* jobid,uint8_t* len,uint64_t* nonce);
	void clearOldJob(char* newjobid);
#ifdef __cplusplus
}
#endif

#endif
