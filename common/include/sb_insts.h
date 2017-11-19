#ifndef SB_INSTS_H
#define SB_INSTS_H

//Mask for accessing shared scratchpad
#define SHARED_SP 0x10000
#define SHARED_SP_INDEX 16


// This sets the context -- ie. which cores the following commands apply to
#define SB_CONTEXT(bitmask) \
  __asm__ __volatile__("sb_ctx %0, t0, 0" : : "r"(bitmask));

//This is the same as SB_CONTEXT butwith
#define SB_SET_ACCEL(core_id) \
  SB_CONTEXT(1<<core_id)


#define SB_CONTEXT_I(bitmask) \
  __asm__ __volatile__("sb_ctx t0, t0, %0" : : "i"(bitmask));

//Stream in the Config
#define SB_CONFIG(mem_addr, size) \
  __asm__ __volatile__("sb_cfg  %0, %1" : : "r"(mem_addr), "i"(size));

//Fill the scratchpad from DMA (from memory or cache)
//Note that scratch_addr will be written linearly
#define SB_DMA_SCRATCH_LOAD_GENERAL(mem_addr, stride, acc_size, stretch, n_strides, scr_addr, shr) \
  __asm__ __volatile__("sb_stride    %0, %1, %2" : : "r"(stride), "r"(acc_size), "i"(stretch)); \
  __asm__ __volatile__("sb_dma_addr  %0, %1" : : "r"(mem_addr), "r"(mem_addr)); \
  __asm__ __volatile__("sb_dma_scr   %0, %1, %2" : : "r"(n_strides), "r"(scr_addr), "i"(shr));

#define SB_DMA_SCRATCH_LOAD_STRETCH(mem_addr, stride, acc_size, stretch, n_strides, scr_addr) \
 SB_DMA_SCRATCH_LOAD_GENERAL(mem_addr, stride, acc_size, stretch, n_strides, scr_addr, 0);


#define SB_SCRATCH_LOAD_REMOTE(remote_scr_addr, stride, acc_size, stretch, n_strides, scr_addr) \
 SB_DMA_SCRATCH_LOAD_GENERAL(remote_scr_addr, stride, acc_size, stretch, n_strides, scr_addr, 1);

//Maintain compatibility with old thing (stretch=0)
#define SB_DMA_SCRATCH_LOAD(mem_addr, stride, acc_size, n_strides, scr_addr) \
  SB_DMA_SCRATCH_LOAD_STRETCH(mem_addr,stride, acc_size, 0, n_strides, scr_addr)

//Fill the scratchpad from DMA (from memory or cache)
//Note that mem_addr will be written linearly
#define SB_SCRATCH_DMA_STORE_GENERAL(scr_addr, stride, access_size, num_strides, mem_addr, shr) \
  __asm__ __volatile__("sb_stride    %0, %1, 0" : : "r"(stride), "r"(access_size)); \
  __asm__ __volatile__("sb_dma_addr  %0, %1" : : "r"(mem_addr), "r"(mem_addr)); \
  __asm__ __volatile__("sb_scr_dma   %0, %1, %2" : : "r"(num_strides), "r"(scr_addr), "i"(shr));

#define SB_SCRATCH_DMA_STORE(scr_addr, stride, access_size, num_strides, mem_addr) \
  SB_SCRATCH_DMA_STORE_GENERAL(scr_addr, stride, access_size, num_strides, mem_addr, 0)

#define SB_SCRATCH_STORE_REMOTE(scr_addr, stride, access_size, num_strides, mem_addr) \
  SB_SCRATCH_DMA_STORE_GENERAL(scr_addr, stride, access_size, num_strides, mem_addr, 1)

//Read from scratch into a cgra port
#define SB_SCR_PORT_STREAM_STRETCH(scr_addr,stride,acc_size,stretch,n_strides, port) \
  __asm__ __volatile__("sb_stride %0, %1, %2" : : "r"(stride), "r"(acc_size), "i"(stretch)); \
  __asm__ __volatile__("sb_scr_rd   %0, %1, %2 " : : "r"(scr_addr), "r"(n_strides), "i"(port)); 

#define SB_SCR_PORT_STREAM(scr_addr,stride,acc_size,n_strides, port) \
   SB_SCR_PORT_STREAM_STRETCH(scr_addr,stride,acc_size,0,n_strides, port) 

//A convienience command for linear access
#define SB_SCRATCH_READ(scr_addr, n_bytes, port) \
  SB_SCR_PORT_STREAM_STRETCH(scr_addr,8,8,0,n_bytes/8, port) 

//Read from DMA into a port
#define SB_DMA_READ_STRETCH(mem_addr, stride, acc_size, stretch, n_strides, port ) \
  __asm__ __volatile__("sb_stride %0, %1, %2" : : "r"(stride), "r"(acc_size), "i"(stretch)); \
  __asm__ __volatile__("sb_dma_rd %0, %1, %2" : : "r"(mem_addr), "r"(n_strides), "i"(port)); 

#define SB_DMA_READ(mem_addr, stride, acc_size, n_strides, port ) \
  SB_DMA_READ_STRETCH(mem_addr, stride, acc_size, 0, n_strides, port )

#define SB_DMA_READ_SIMP(mem_addr, num_strides, port ) \
  __asm__ __volatile__("sb_dma_rd    %0, %1, %2" : : "r"(mem_addr), "r"(num_strides), "i"(port)); 

//Throw away some outputs.  We will add a proper instruction for this at some point, rather then writing to memory
#define SB_GARBAGE(output_port, num_elem) \
  __asm__ __volatile__("sb_stride   %0, %1, 0" : : "r"(8), "r"(8)); \
  __asm__ __volatile__("sb_wr_dma   %0, %1, %2"   : : "r"(0), "r"(num_elem), "i"(output_port|0x100)); 

//Throw away some outputs.  We will add a proper instruction for this at some point, rather then writing to memory
#define SB_GARBAGE_SIMP(output_port, num_elem) \
  __asm__ __volatile__("sb_wr_dma   %0, %1, %2"   : : "r"(0), "r"(num_elem), "i"(output_port|0x100)); 


// Memory Oriented Instructions

//Set this back to zero if you need different kinds of writes later in the same code!!!
#define SB_GARBAGE_BEFORE_STRIDE(num_garb) \
  __asm__ __volatile__("sb_garb   %0, %1, 0" : : "r"(num_garb), "r"(num_garb)); \

// Plain Write to Memory
#define SB_DMA_WRITE(output_port, stride, access_size, num_strides, mem_addr) \
  __asm__ __volatile__("sb_stride   %0, %1, 0" : : "r"(stride), "r"(access_size)); \
  __asm__ __volatile__("sb_wr_dma   %0, %1, %2"   : : "r"(mem_addr), "r"(num_strides), "i"(output_port)); 

// This is for optimizations when core is a bottleneck, break into two commands:
#define SB_STRIDE_STRETCH(stride, acc_size, stretch) \
 __asm__ __volatile__("sb_stride %0, %1, %2" : : "r"(stride), "r"(acc_size), "i"(stretch));

#define SB_STRIDE(stride, access_size) \
   SB_STRIDE_STRETCH(stride, access_size, 0)


#define SB_DMA_WRITE_SIMP(output_port, num_strides, mem_addr) \
  __asm__ __volatile__("sb_wr_dma   %0, %1, %2"   : : "r"(mem_addr), "r"(num_strides), "i"(output_port)); 


//Write to DMA, but throw away all but the last 16-bits from each word
#define SB_DMA_WRITE_SHF16(output_port, stride, access_size, num_strides, mem_addr) \
  __asm__ __volatile__("sb_stride   %0, %1, 0" : : "r"(stride), "r"(access_size)); \
  __asm__ __volatile__("sb_wr_dma   %0, %1, %2"   : : "r"(mem_addr),  "r"(num_strides), "i"(output_port|0x40)); 

//Write to DMA, but throw away all but the last 32-bits from each word  (implemented, not tested yet)
#define SB_DMA_WRITE_SHF32(output_port, stride, access_size, num_strides, mem_addr) \
  __asm__ __volatile__("sb_stride   %0, %1, 0" : : "r"(stride), "r"(access_size)); \
  __asm__ __volatile__("sb_wr_dma   %0, %1, %2"   : : "r"(mem_addr),  "r"(num_strides), "i"(output_port|0x80)); 


// Scratch Oriented Instructions
// Plain Write to Scratch  (TODO -- NEW -- TEST)
#define SB_SCR_WRITE(output_port, num_bytes, scr_addr) \
  __asm__ __volatile__("sb_wr_scr   %0, %1, %2"   : : "r"(scr_addr), "r"(num_bytes), "i"(output_port)); 




// Unused Instructions
//  __asm__ __volatile__("sb_wr %0 "          : : "i"(output_port)); 
//  __asm__ __volatile__("sb_dma_addr_p %0, %1, " #output_port : : "r"(mem_addr), "r"(stride_size)); 
//  __asm__ __volatile__("sb_dma_wr   %0, " : : "r"(num_strides)); 

//Send a constant value, repetated num_elements times to a port
#define SB_CONST(port, val, num_elements) \
  __asm__ __volatile__("sb_const %0, %1, %2 " : : "r"(val), "r"(num_elements), "i"(port)); 

//Put a softbrain generated output value to a riscv core variable
#define SB_RECV(out_port, val) \
  __asm__ __volatile__("sb_recv %0, a0, %1 " : "=r"(val) : "i"(out_port)); 

//Send a constant value, repetated num_elements times to a port
// Plain Write to Scratch
#define SB_2D_CONST(port, val1, v1_repeat, val2, v2_repeat, iters) \
  __asm__ __volatile__("sb_set_iter %0 " : : "r"(iters)); \
  __asm__ __volatile__("sb_const %0, %1, %2 " : : "r"(val1), "r"(v1_repeat), "i"(port|(1<<7))); \
  __asm__ __volatile__("sb_const %0, %1, %2 " : : "r"(val2), "r"(v2_repeat), "i"(port|(1<<6))); 


// This tells the port to repeat a certain number of times before consuming
#define SB_CONFIG_PORT(repeat_times, stretch) \
  __asm__ __volatile__("sb_cfg_port %0, t0, %1" : : "r"(repeat_times), "i"(stretch));

#define SB_REPEAT_PORT(times) \
  SB_CONFIG_PORT(times,0);

//Write to Scratch from a CGRA output port.  Note that only linear writes are currently allowed
//#define SB_SCRATCH_WRITE(output_port, num_bytes, scratch_addr) 
//%  __asm__ __volatile__("sb_scr_wr   %0, %1, %2" : : "r"(scratch_addr), "r"(num_bytes), "i"(output_port)); 

//Write from output to input port
#define SB_RECURRENCE(output_port, input_port, num_strides) \
  __asm__ __volatile__("sb_wr_rd %0, %1" : : "r"(num_strides), "i"((input_port<<5) | (output_port)));

//Write from output to remote input port
//pos: local=0, left=1, right=2, undef=3
//(might be replaced later by some other RISCV instructions)
#define SB_XFER_LEFT(output_port, input_port, num_strides) \
  __asm__ __volatile__("sb_wr_rd %0, %1" : : "r"(num_strides), "i"(1<<10 | (input_port<<5) | (output_port)));
#define SB_XFER_RIGHT(output_port, input_port, num_strides) \
  __asm__ __volatile__("sb_wr_rd %0, %1" : : "r"(num_strides), "i"( -2048 + (0<<10 |  (input_port<<5) | (output_port))) );


//Write from output to input port  (type -- 3:8-bit,2:16-bit,1:32-bit,0:64-bit)
#define SB_INDIRECT(ind_port, addr_offset, type, num_elem, input_port) \
  __asm__ __volatile__("sb_ind %0, %1, %2" : : "r"(addr_offset), "r"(num_elem),\
                                               "i"((type<<10)|(input_port<<5) | (ind_port)));
//64-bit indicies, 64-bit values
#define SB_INDIRECT64(ind_port, addr_offset, num_elem, input_port) \
  SB_INDIRECT(ind_port, addr_offset, 0, num_elem, input_port)

// THIS DOES NOT WORK DO NOT USE THIS : )
#define SB_INDIRECT32(ind_port, addr_offset, input_port) \
  SB_INDIRECT(ind_port, addr_offset, 12231, input_port)


#define SB_INDIRECT_WR(ind_port, addr_offset, type, num_elem, output_port) \
  __asm__ __volatile__("sb_ind_wr %0, %1, %2" : : "r"(addr_offset), "r"(num_elem),\
                                       "i"((type<<10)|(output_port<<5) | (ind_port)));
//64-bit indicies, 64-bit values
#define SB_INDIRECT64_WR(ind_port, addr_offset, num_elem, output_port) \
  SB_INDIRECT_WR(ind_port, addr_offset, 0, num_elem, output_port)



//Wait with custom bit vector -- probably don't need to use
#define SB_WAIT(bit_vec) \
  __asm__ __volatile__("sb_wait t0, t0, " #bit_vec); \

//Wait for all softbrain commands and computations to be visible to memory from control core 
#define SB_WAIT_ALL() \
  __asm__ __volatile__("sb_wait t0, t0, 0" : : : "memory"); \

//Wait for all prior scratch writes to be complete.
#define SB_WAIT_SCR_WR() \
  __asm__ __volatile__("sb_wait t0, t0, 1"); \

//wait for everything except outputs to be complete. (useful for debugging)
#define SB_WAIT_COMPUTE() \
  __asm__ __volatile__("sb_wait t0, t0, 2"); \

//wait for all prior scratch reads to be complete
#define SB_WAIT_SCR_RD() \
  __asm__ __volatile__("sb_wait t0, t0, 4"); \

//wait for all prior scratch reads to be complete (NOT IMPLEMENTED IN SIMULTOR YET)
#define SB_WAIT_SCR_RD_QUEUED() \
  __asm__ __volatile__("sb_wait t0, t0, 8"); \


//Indirect Ports
#define P_IND_1 (31)
#define P_IND_2 (30)
#define P_IND_3 (29)
#define P_IND_4 (28)

#define P_IND_TRIP0 (25)
#define P_IND_TRIP1 (26)
#define P_IND_TRIP2 (27)

#define P_IND_DOUB0 (26)
#define P_IND_DOUB1 (27)


#endif
