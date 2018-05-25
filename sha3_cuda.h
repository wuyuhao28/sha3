#ifndef SHA3_CUDA_H
#define SHA3_CUDA_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_functions.h"

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>

void runBenchmarks(char *h_messages, uint8_t *sequence);

#endif