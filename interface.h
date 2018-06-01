#ifndef INTERFACE_H_
#define INTERFACE_H_

#ifdef __cplusplus
extern "C" {
#endif
	#include <stdint.h>
	
	int get_hash_init();
	int get_hash(uint8_t msg[32], uint8_t seed[32],uint8_t ret[32],uint32_t *deviceids);
#ifdef __cplusplus
}
#endif

#endif
