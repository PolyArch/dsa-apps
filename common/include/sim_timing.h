#ifdef __x86_64__
__attribute__ ((noinline)) static void begin_roi() {
}
__attribute__ ((noinline)) static void end_roi()   {
}

#else
__attribute__ ((noinline)) static void begin_roi() {
    __asm__ __volatile__("add x0, x0, 1"); \
}
__attribute__ ((noinline)) static void end_roi()   {
     __asm__ __volatile__("add x0, x0, 2"); \
}
#endif

