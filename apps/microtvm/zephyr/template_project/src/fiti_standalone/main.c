#include <stdio.h>
#include <zephyr.h>
#include <sys/printk.h>

#include "tvmgen_default.h"
#include "tvm/runtime/crt/page_allocator.h"
#include "tvm/runtime/crt/error_codes.h"
#include "dlpack/dlpack.h"
#include "process.h"

__attribute__((section("SDRAM2"))) MemoryManagerInterface* memory_manager;

__attribute__((section("SDRAM2"))) extern int8_t raw_data[192 * 192];
__attribute__((section("SDRAM2"))) int8_t result[1 * 540 * 10];
__attribute__((section("SDRAM2"))) struct tvmgen_default_inputs inputs;
__attribute__((section("SDRAM2"))) struct tvmgen_default_outputs outputs;
__attribute__((section("SDRAM2"))) uint8_t memory[5 * 1024 * 1024];

void TVMPlatformAbort(tvm_crt_error_t error_code) {
	printf("[%s] tvm_crt_error_t: %d\r\n", __func__, error_code);
}

void main(void) {
    uint32_t t1, t2, t3, t4, t5, t6;
    uint32_t clock_freq = sys_clock_hw_cycles_per_sec();
    TVMLogf("Now hardware frequency is %uMHz\r\n", clock_freq/1000000);

    t1 = k_cycle_get_32();
    inputs.input_1_int8 = raw_data; //inputs.[input_layer_name]
    outputs.Identity_int8 = result; //outputs.[output_layer_name]

    PageMemoryManagerCreate(&memory_manager, memory, sizeof(memory), 8 /* page_size_log2 */);
    t2 = k_cycle_get_32();

	TVMLogf("[%s] starting in %u ms...\r\n", __func__, (uint32_t)((uint64_t)(t2-t1)*1000/clock_freq));
    while(true){
        t3 = k_cycle_get_32();
        tvmgen_default_run(&inputs, &outputs);
        t4 = k_cycle_get_32();

        t5 = k_cycle_get_32();
        post_process(outputs.Identity_int8);
        t6 = k_cycle_get_32();
        TVMLogf("model cost %u ms, post_process cost %u ms\r\n", (uint32_t)((uint64_t)(t4-t3)*1000/clock_freq), (uint32_t)((uint64_t)(t6-t5)*1000/clock_freq));
    }
}