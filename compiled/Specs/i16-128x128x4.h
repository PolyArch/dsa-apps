#ifndef FAKE

#define TYPE int16_t

#define row_size 128
#define col_size 128
#define C 4
#define N (row_size*col_size*C)

#else

#define TYPE int64_t

#define row_size 128
#define col_size 128
#define C 1
#define N (row_size*col_size*C)

#endif
