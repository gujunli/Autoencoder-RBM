#ifndef __katdoth__
#define __katdoth__

#include "../include/Random123/philox.h"
#include "../include/Random123/threefry.h"
#include "../include/Random123/ars.h"
#include "../include/Random123/aes.h"

typedef unsigned int uint32_t;

struct array4x32 
{	
	uint32_t v[4]; 
};

enum method_e
{
	threefry4x32_e,

    unused // silences warning about dangling comma
};

#endif