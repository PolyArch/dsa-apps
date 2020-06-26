#pragma once

// implement some wheels to unify c++ abi.

#include <cstdlib>
#include <cstdio>

template<typename T>
void random_shuffle(int n, T *a) {
  for (int i = 0; i < n; ++i) {
    int x = rand() % n;
    int y = rand() % n;
    T t;
    t = a[x];
    a[x] = a[y];
    a[y] = t;
  }
}

template<typename T>
void merge_sort_impl(T *workspace, int n, T *a) {
  if (n == 1) {
    return;
  }
  int na = n / 2;
  merge_sort_impl(workspace, n / 2, a);
  int nb = n - na;
  T *b = a + n / 2;
  merge_sort_impl(workspace, n - n / 2, a + n / 2);
  int i, j, append;
  for (i = 0, j = 0, append = 0; i < na && j < nb; ) {
    if (a[i] < b[j]) {
      workspace[append++] = a[i++];
    } else {
      workspace[append++] = b[j++];
    }
  }
  while (i < na)
    workspace[append++] = a[i++];
  while (j < nb)
    workspace[append++] = b[j++];
  for (int _ = 0; _ < n; ++_) {
    a[_] = workspace[_];
  }
}

template<typename T>
void merge_sort(int n, T *a) {
  T *workspace = new T[n];
  merge_sort_impl(workspace, n, a);
  delete []workspace;
}
