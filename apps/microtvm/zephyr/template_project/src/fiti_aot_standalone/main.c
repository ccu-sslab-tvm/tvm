#include <stdio.h>
#include <zephyr/kernel.h>
#include <zephyr/sys/printk.h>

#include "tvmgen_default.h"

extern int8_t raw_data[];
int8_t result[1 * 540 * 10];
struct tvmgen_default_inputs inputs;
struct tvmgen_default_outputs outputs;

void main(void) {
    uint32_t t1, t2, t3, t4;
    uint32_t clock_freq = sys_clock_hw_cycles_per_sec();
    TVMLogf("Now hardware frequency is %uMHz\r\n", clock_freq/1000000);

	TVMPlatformInitialize();
    inputs.input_1_int8 = raw_data; //inputs.[input_layer_name]
    outputs.Identity_int8 = result; //outputs.[output_layer_name]

	TVMLogf("[%s] starting...\r\n", __func__);
    while(true){
        t1 = k_cycle_get_32();
        tvmgen_default_run(&inputs, &outputs);
        t2 = k_cycle_get_32();

        t3 = k_cycle_get_32();
        post_process(outputs.Identity_int8);
        t4 = k_cycle_get_32();
        TVMLogf("model cost %u ms, post_process cost %u ms\r\n", (uint32_t)((uint64_t)(t2-t1)*1000/clock_freq), (uint32_t)((uint64_t)(t4-t3)*1000/clock_freq));
    }
}