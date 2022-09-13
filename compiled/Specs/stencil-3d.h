
//Define input sizes
#define height_size 34
#define col_size 34
#define row_size 34
//Data Bounds
#define TYPE int64_t
#define MAX 1000
#define MIN 1
#define SIZE (row_size * col_size * height_size)
#define INDX(_row_size,_col_size,_i,_j,_k) ((_i)+_row_size*((_j)+_col_size*(_k)))
