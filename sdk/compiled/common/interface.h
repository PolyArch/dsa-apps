struct Arguments;

struct Arguments *init_data();

void run_reference(struct Arguments *);

void run_accelerator(struct Arguments *, int);

int sanity_check(struct Arguments *);


#define NO_SANITY_CHECK \
  void __attribute__((weak)) run_reference(struct Arguments *args) {} \
  int __attribute__((weak)) sanity_check(struct Arguments *args) { return 1; }

#define NO_INIT_DATA \
  struct Arguments *init_data() { return &args_; }
