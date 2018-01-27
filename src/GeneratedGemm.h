void noarchm3n3k3a1b0(const double* A, const double* B, double* C) {
#pragma message ("LIBXSMM KERNEL COMPILATION WARNING: compiling arch-independent gemm kernel in: " __FILE__)
  unsigned int l_m = 0;
  unsigned int l_n = 0;
  unsigned int l_k = 0;

  for ( l_n = 0; l_n < 3; l_n++ ) {
    for ( l_m = 0; l_m < 3; l_m++ ) { C[(l_n*3)+l_m] = 0.0; }

    for ( l_k = 0; l_k < 3; l_k++ ) {
      #pragma simd
      for ( l_m = 0; l_m < 3; l_m++ ) {
        C[(l_n*3)+l_m] += A[(l_k*3)+l_m] * B[(l_n*3)+l_k];
      }
    }
  }
}

void knlm3n3k3a1b0(const double* A, const double* B, double* C) {
#ifdef __AVX512F__
  __asm__ __volatile__("movq %0, %%rdi\n\t"
                       "movq %1, %%rsi\n\t"
                       "movq %2, %%rdx\n\t"
                       "movq $0, %%r12\n\t"
                       "movq $0, %%r13\n\t"
                       "movq $0, %%r14\n\t"
                       "movq $7, %%r9\n\t"
                       "kmovw %%r9d, %%k1\n\t"
                       "vpxord %%zmm29, %%zmm29, %%zmm29\n\t"
                       "vpxord %%zmm30, %%zmm30, %%zmm30\n\t"
                       "vpxord %%zmm31, %%zmm31, %%zmm31\n\t"
                       "movq $24, %%r15\n\t"
                       "movq $72, %%rax\n\t"
                       "movq $120, %%rbx\n\t"
                       "movq $168, %%r11\n\t"
                       "vpxord %%zmm26, %%zmm26, %%zmm26\n\t"
                       "vpxord %%zmm27, %%zmm27, %%zmm27\n\t"
                       "vpxord %%zmm28, %%zmm28, %%zmm28\n\t"
                       "vpxord %%zmm23, %%zmm23, %%zmm23\n\t"
                       "vpxord %%zmm24, %%zmm24, %%zmm24\n\t"
                       "vpxord %%zmm25, %%zmm25, %%zmm25\n\t"
                       "vpxord %%zmm20, %%zmm20, %%zmm20\n\t"
                       "vpxord %%zmm21, %%zmm21, %%zmm21\n\t"
                       "vpxord %%zmm22, %%zmm22, %%zmm22\n\t"
                       "vmovupd 0(%%rdi), %%zmm0%{%%k1%}%{z%}\n\t"
                       "vmovupd 24(%%rdi), %%zmm1%{%%k1%}%{z%}\n\t"
                       "vfmadd231pd 0(%%rsi)%{1to8%}, %%zmm0, %%zmm29\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm30\n\t"
                       "vfmadd231pd 0(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm31\n\t"
                       "vmovupd 48(%%rdi), %%zmm0%{%%k1%}%{z%}\n\t"
                       "vfmadd231pd 8(%%rsi)%{1to8%}, %%zmm1, %%zmm26\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,1)%{1to8%}, %%zmm1, %%zmm27\n\t"
                       "vfmadd231pd 8(%%rsi,%%r15,2)%{1to8%}, %%zmm1, %%zmm28\n\t"
                       "vfmadd231pd 16(%%rsi)%{1to8%}, %%zmm0, %%zmm23\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,1)%{1to8%}, %%zmm0, %%zmm24\n\t"
                       "vfmadd231pd 16(%%rsi,%%r15,2)%{1to8%}, %%zmm0, %%zmm25\n\t"
                       "vaddpd %%zmm26, %%zmm29, %%zmm29\n\t"
                       "vaddpd %%zmm27, %%zmm30, %%zmm30\n\t"
                       "vaddpd %%zmm28, %%zmm31, %%zmm31\n\t"
                       "vaddpd %%zmm23, %%zmm29, %%zmm29\n\t"
                       "vaddpd %%zmm24, %%zmm30, %%zmm30\n\t"
                       "vaddpd %%zmm25, %%zmm31, %%zmm31\n\t"
                       "vaddpd %%zmm20, %%zmm29, %%zmm29\n\t"
                       "vaddpd %%zmm21, %%zmm30, %%zmm30\n\t"
                       "vaddpd %%zmm22, %%zmm31, %%zmm31\n\t"
                       "vmovupd %%zmm29, 0(%%rdx)%{%%k1%}\n\t"
                       "vmovupd %%zmm30, 24(%%rdx)%{%%k1%}\n\t"
                       "vmovupd %%zmm31, 48(%%rdx)%{%%k1%}\n\t"
                       "addq $24, %%rdx\n\t"
                       "addq $24, %%rdi\n\t"
                       : : "m"(A), "m"(B), "m"(C) : "k1","rax","rbx","rcx","rdx","rdi","rsi","r8","r9","r10","r11","r12","r13","r14","r15","zmm0","zmm1","zmm2","zmm3","zmm4","zmm5","zmm6","zmm7","zmm8","zmm9","zmm10","zmm11","zmm12","zmm13","zmm14","zmm15","zmm16","zmm17","zmm18","zmm19","zmm20","zmm21","zmm22","zmm23","zmm24","zmm25","zmm26","zmm27","zmm28","zmm29","zmm30","zmm31");
#else
#pragma message ("LIBXSMM KERNEL COMPILATION ERROR in: " __FILE__)
#error No kernel was compiled, lacking support for current architecture?
#endif
}

