#define TYPE int16_t

#define stencil_row 3
#define stencil_col 3

#define o_row_size 128
#define i_row_size (o_row_size + stencil_row - 1)
#define o_col_size 128
#define i_col_size (o_col_size + stencil_col - 1)
#define C 4
#define o_size (o_row_size*o_col_size*C)
#define i_size (i_row_size*i_col_size*C)
