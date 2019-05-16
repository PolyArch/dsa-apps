## Multi-core ssim functions/primitives supported


* SS_REM_PORT(output_port, num_elem, mask, remote_port)



* SS_IND_REM_SCRATCH(val_port, addr_port, num_elem, scr_base_addr, scratch_type)

* SS_REM_SCRATCH(scr_base_addr, stride, access_size, num_strides, val_port, scratch_type)

* SS_SCR_REM_SCR(src_scr_base_addr, stride, access_size, num_strides, dest_scr_base_addr, scratch_type)

* SS_SCR_REM_PORT(scr_base_addr, num_strides, mask, remote_port)

* SS_ATOMIC_SCR_OP(addr_port, val_port, offset, iters, opcode)

* SS_WAIT_DF(num_rem_writes, scratch_type)

* SS_GLOBAL_BARRIER
