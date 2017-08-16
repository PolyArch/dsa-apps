#ifndef SB_INSTS_H
#define SB_INSTS_H

//TODO Seperate out the sb_wait commands
//and other ones which are dependent on arguments

int garbage[1024];

//Stream in the Config
#define SB_CONFIG(mem_addr, size) \
  __asm__ __volatile__("sb_cfg  %0, %1" : : "r"(mem_addr), "i"(size));

//Fill the scratchpad from DMA (from memory or cache)
//Note that scratch_addr will be written linearly
#define SB_DMA_SCRATCH_LOAD(mem_addr, stride, access_size, num_strides, scratch_addr) \
  __asm__ __volatile__("sb_stride    %0, %1" : : "r"(stride), "r"(access_size)); \
  __asm__ __volatile__("sb_dma_addr  %0, %1" : : "r"(mem_addr), "r"(mem_addr)); \
  __asm__ __volatile__("sb_dma_scr   %0, %1, 0" : : "r"(num_strides), "r"(scratch_addr));

//Read from scratch into a cgra port
#define SB_SCR_PORT_STREAM(scr_addr, stride, access_size, num_strides, port ) \
  __asm__ __volatile__("sb_stride   %0, %1" : : "r"(stride), "r"(access_size)); \
  __asm__ __volatile__("sb_scr_rd   %0, %1, %2 " : : "r"(scr_addr), "r"(num_strides), "i"(port)); 

//A convienience CMD if you want to read linearly
#define SB_SCRATCH_READ(scr_addr, num_bytes, port) \
  __asm__ __volatile__("sb_stride   %0, %1" : : "r"(8), "r"(8)); \
  __asm__ __volatile__("sb_scr_rd   %0, %1, %2 " : : "r"(scr_addr), "r"(num_bytes/8), "i"(port)); 

//Read from DMA into a port
#define SB_DMA_READ(mem_addr, stride, access_size, num_strides, port ) \
  __asm__ __volatile__("sb_stride    %0, %1" : : "r"(stride), "r"(access_size)); \
  __asm__ __volatile__("sb_dma_rd    %0, %1, %2" : : "r"(mem_addr), "r"(num_strides), "i"(port)); 

#define SB_DMA_READ_SIMP(mem_addr, num_strides, port ) \
  __asm__ __volatile__("sb_dma_rd    %0, %1, %2" : : "r"(mem_addr), "r"(num_strides), "i"(port)); 

//Throw away some outputs.  We will add a proper instruction for this at some point, rather then writing to memory
#define SB_GARBAGE(output_port, num_elem) \
  __asm__ __volatile__("sb_stride   %0, %1" : : "r"(8), "r"(8)); \
  __asm__ __volatile__("sb_wr_dma   %0, %1, %2"   : : "r"(&garbage), "r"(num_elem), "i"(output_port|0x100)); 

//Throw away some outputs.  We will add a proper instruction for this at some point, rather then writing to memory
#define SB_GARBAGE_SIMP(output_port, num_elem) \
  __asm__ __volatile__("sb_wr_dma   %0, %1, %2"   : : "r"(&garbage), "r"(num_elem), "i"(output_port|0x100)); 


//Write to DMA.
#define SB_DMA_WRITE(output_port, stride, access_size, num_strides, mem_addr) \
  __asm__ __volatile__("sb_stride   %0, %1" : : "r"(stride), "r"(access_size)); \
  __asm__ __volatile__("sb_wr_dma   %0, %1, %2"   : : "r"(mem_addr), "r"(num_strides), "i"(output_port)); 

// This is for optimizations when core is a bottleneck, break into two commands:
#define SB_STRIDE(stride, access_size) \
  __asm__ __volatile__("sb_stride   %0, %1" : : "r"(stride), "r"(access_size)); 

#define SB_DMA_WRITE_SIMP(output_port, num_strides, mem_addr) \
  __asm__ __volatile__("sb_wr_dma   %0, %1, %2"   : : "r"(mem_addr), "r"(num_strides), "i"(output_port)); 





//Write to DMA, but throw away all but the last 16-bits from each word
#define SB_DMA_WRITE_SHF16(output_port, stride, access_size, num_strides, mem_addr) \
  __asm__ __volatile__("sb_stride   %0, %1" : : "r"(stride), "r"(access_size)); \
  __asm__ __volatile__("sb_wr_dma   %0, %1, %2"   : : "r"(mem_addr),  "r"(num_strides), "i"(output_port|0x40)); 

//Write to DMA, but throw away all but the last 16-bits from each word
//WARNING -- (NOT IMPLEMENTED IN SIMULTOR YET)
#define SB_DMA_WRITE_SHF32(output_port, stride, access_size, num_strides, mem_addr) \
  __asm__ __volatile__("sb_stride   %0, %1" : : "r"(stride), "r"(access_size)); \
  __asm__ __volatile__("sb_wr_dma   %0, %1, %2"   : : "r"(mem_addr),  "r"(num_stirides), "i"(output_port|0x80)); 

//  __asm__ __volatile__("sb_dma_addr %0, %1" : : "r"(access_size), "r"(stride)); 
//  __asm__ __volatile__("sb_wr %0 "          : : "i"(output_port)); 
//  __asm__ __volatile__("sb_stride   %0, %1" : : "r"(mem_addr), "r"(stride)); 
//  __asm__ __volatile__("sb_dma_addr_p %0, %1, " #output_port : : "r"(mem_addr), "r"(stride_size)); 
//  __asm__ __volatile__("sb_dma_wr   %0, " : : "r"(num_strides)); 

//Send a constant value, repetated num_elements times to a port
#define SB_CONST(port, val, num_elements) \
  __asm__ __volatile__("sb_const %0, %1, %2 " : : "r"(val), "r"(num_elements), "i"(port)); 

//Write to Scratch from a CGRA output port.  Note that only linear writes are currently allowed
//#define SB_SCRATCH_WRITE(output_port, num_bytes, scratch_addr) 
//%  __asm__ __volatile__("sb_scr_wr   %0, %1, %2" : : "r"(scratch_addr), "r"(num_bytes), "i"(output_port)); 

//Write from output to input port
#define SB_RECURRENCE(output_port, input_port, num_strides) \
  __asm__ __volatile__("sb_wr_rd %0, %1" : : "r"(num_strides), "i"((input_port<<6) | (output_port)));

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

//Wait for all softbrain commands to be done -- This will block the processor indefinately if there is
//unbalanced commands
#define SB_WAIT_ALL() \
  __asm__ __volatile__("sb_wait t0, t0, 0" : : : "memory"); \

//Wait for all prior scratch writes to be complete.
#define SB_WAIT_SCR_WR() \
  __asm__ __volatile__("sb_wait t0, t0, 1"); \

//wait for everything except outputs to be complete. (useful for debugging)
#define SB_WAIT_COMPUTE() \
  __asm__ __volatile__("sb_wait t0, t0, 2"); \

//wait for all prior scratch reads to be complete (NOT IMPLEMENTED IN SIMULTOR YET)
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