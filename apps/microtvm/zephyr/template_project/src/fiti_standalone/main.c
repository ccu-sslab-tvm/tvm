#include <stdio.h>
#include <zephyr.h>

#include "tvmgen_default.h"
#include "tvm/runtime/crt/page_allocator.h"
#include "tvm/runtime/crt/error_codes.h"
#include "dlpack/dlpack.h"

void main(void) {
    int count = 0;
    while (true) {
        printf("Hello_world %d\r\n", ++count);
    }
}