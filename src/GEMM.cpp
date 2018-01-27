#include "GEMM.h"
#include "immintrin.h" //avx512
// macros for better readability
#define A(i,j,lda) A[ (j)*lda + (i)]
#define B(i,j,ldb) B[ (j)*ldb + (i)]
#define C(i,j,ldc) C[ (j)*ldc + (i)]

#define A_tmp(i,j,lda) A_tmp[ (j)*lda + (i)]
#define B_tmp(i,j,ldb) B_tmp[ (j)*ldb + (i)]
#define C_tmp(i,j,ldc) C_tmp[ (j)*ldc + (i)]
#define C_tmp1(i,j,ldc) C_tmp1[ (j)*ldc + (i)]

#include <iostream>
#include <cmath>
#include "GeneratedGemm.h"
#include "constants.h"

void DGEMM_old(unsigned M, 
	unsigned N, 
	unsigned K, 
	double alpha, 
	double const* A, 
	unsigned ldA, 
	double const* B, 
	unsigned ldB, 
	double beta, 
	double* C, 
	unsigned ldC) {
  	for (unsigned j = 0; j < N; ++j) {
    	for (unsigned i = 0; i < M; ++i) {
    		double cij = 0.0;
    		for (unsigned k = 0; k < K; ++k) {
    			cij += A[k*ldA + i] * B[j*ldB + k];
	    	}
	    	C[j*ldC + i] = alpha * cij + beta * C[j*ldC + i];
		}
	}
}

void inner512_8x8(int K_, 
	const double* A, 
	int lda, 
	const double* B, 
	int ldb, 
	double* C, 
	int ldc, 
	const __m512d alpha_v, 
	const __m512d beta_v);

void DGEMM(unsigned M_, 
	unsigned N_, 
	unsigned K_, 
	double alpha, 
	double const* A, 
	unsigned lda, 
	double const* B, 
	unsigned ldb,
	double beta,
	double* C, 
	unsigned ldc) {
	
	// if switches call LIBXSMM generated kernels for all possible matrices with alpha=1 and beta=0
	if (M_ == 3 && N_ == 3 && K_ == 3 && alpha == 1 && beta == 0 && lda == 3 && ldb == 3 && ldc == 3) {		
		DGEMMm3n3k3a1b0(A,B,C);
	}
	else if (M_ == 6 && N_ == 3 && K_ == 6 && alpha == 1 && beta == 0 && lda == 6 && ldb == 6 && ldc == 6) {
		DGEMMm6n3k6a1b0(A,B,C);
	}
	else if (M_ == 10 && N_ == 3 && K_ == 10 && alpha == 1 && beta == 0 && lda == 10 && ldb == 10 && ldc == 10) {
		DGEMMm10n3k10a1b0(A,B,C);
	}
	else if (M_ == 15 && N_ == 3 && K_ == 15 && alpha == 1 && beta == 0 && lda == 15 && ldb == 15 && ldc == 15) {
		DGEMMm15n3k15a1b0(A,B,C);
	}
	else if (M_ == 21 && N_ == 3 && K_ == 21 && alpha == 1 && beta == 0 && lda == 21 && ldb == 21 && ldc == 21) {
		DGEMMm21n3k21a1b0(A,B,C);
	}
	else if (M_ == 28 && N_ == 3 && K_ == 28 && alpha == 1 && beta == 0 && lda == 28 && ldb == 28 && ldc == 28) {
		DGEMMm28n3k28a1b0(A,B,C);
	}
	else if (M_ == 36 && N_ == 3 && K_ == 36 && alpha == 1 && beta == 0 && lda == 36 && ldb == 36 && ldc == 36) {
		DGEMMm36n3k36a1b0(A,B,C);
	}
	else if (M_ == 45 && N_ == 3 && K_ == 45 && alpha == 1 && beta == 0 && lda == 45 && ldb == 45 && ldc == 45) {
		DGEMMm45n3k45a1b0(A,B,C);
	}
	else if (M_ == 55 && N_ == 3 && K_ == 55 && alpha == 1 && beta == 0 && lda == 55 && ldb == 55 && ldc == 55) {
		DGEMMm55n3k55a1b0(A,B,C);
	}
	else if (M_ == 66 && N_ == 3 && K_ == 66 && alpha == 1 && beta == 0 && lda == 66 && ldb == 66 && ldc == 66) {
		DGEMMm66n3k66a1b0(A,B,C);
	}
	else if (M_ == 78 && N_ == 3 && K_ == 78 && alpha == 1 && beta == 0 && lda == 78 && ldb == 78 && ldc == 78) {
		DGEMMm78n3k78a1b0(A,B,C);
	}
	else if (M_ == 4 && N_ == 3 && K_ == 3 && alpha == 1 && beta == 0 && lda == 4 && ldb == 3 && ldc == 4) {
		DGEMMm4n3k3a1b0(A,B,C);
	}
	else if (M_ == 9 && N_ == 3 && K_ == 6 && alpha == 1 && beta == 0 && lda == 9 && ldb == 6 && ldc == 9) {
		DGEMMm9n3k6a1b0(A,B,C);
	}
	else if (M_ == 16 && N_ == 3 && K_ == 10 && alpha == 1 && beta == 0 && lda == 16 && ldb == 10 && ldc == 16) {
		DGEMMm16n3k10a1b0(A,B,C);
	}
	else if (M_ == 16 && N_ == 3 && K_ == 15 && alpha == 1 && beta == 0 && lda == 16 && ldb == 15 && ldc == 16) {
		DGEMMm16n3k15a1b0(A,B,C);
	}
	else if (M_ == 25 && N_ == 3 && K_ == 21 && alpha == 1 && beta == 0 && lda == 25 && ldb == 21 && ldc == 25) {
		DGEMMm25n3k21a1b0(A,B,C);
	}
	else if (M_ == 36 && N_ == 3 && K_ == 28 && alpha == 1 && beta == 0 && lda == 36 && ldb == 28 && ldc == 36) {
		DGEMMm36n3k28a1b0(A,B,C);
	}
	else if (M_ == 49 && N_ == 3 && K_ == 45 && alpha == 1 && beta == 0 && lda == 49 && ldb == 45 && ldc == 49) {
		DGEMMm49n3k45a1b0(A,B,C);
	}
	else if (M_ == 64 && N_ == 3 && K_ == 55 && alpha == 1 && beta == 0 && lda == 64 && ldb == 55 && ldc == 64) {
		DGEMMm64n3k55a1b0(A,B,C);
	}
	else if (M_ == 81 && N_ == 3 && K_ == 66 && alpha == 1 && beta == 0 && lda == 81 && ldb == 66 && ldc == 81) {
		DGEMMm81n3k66a1b0(A,B,C);
	}
	else if (M_ == 81 && N_ == 3 && K_ == 78 && alpha == 1 && beta == 0 && lda == 81 && ldb == 78 && ldc == 81) {
		DGEMMm6n3k6a1b0(A,B,C);
	}
	else {
		//DGEMM_old(M_, N_, K_, alpha, A, lda, B, ldb, beta, C, ldc);
		
		/**
		* creates larger buffers for DGEMM calls with alpha different from 1 or -1
		* all these matrixes have sizes A: NUMBER_OF_BASIS_FUNCTIONS B: NUMBER_OF_QUANTITIES C: NUMBER_OF_QUANTITIES
		* so B is always 3*3 and the largest possible matrix is 78x3
		* M and N have to be multiples of 8 because of 512 vectorisation
		* K is unrolled by 4 in the microkernel so it has to be a multiple of 4
		* 320 and 32 are the maximum needed sizes (order 12)
		* */
		
		const __m512d alpha_v = _mm512_set1_pd(alpha);
		const __m512d beta_v = _mm512_set1_pd(beta);
		
		int ac_ld = ceil((double)NUMBER_OF_BASIS_FUNCTIONS/8.0)*8;
		int ac_size = ac_ld*4;
		
		double A_tmp[ac_size];
		double B_tmp[32];
		double C_tmp[ac_size];
		
		#pragma vector always
		for (int i = 0; i < ac_size; i++) 
		{
			A_tmp[i] = 0;
			C_tmp[i] = 0;
		}
		#pragma vector always
		for (int i = 0; i < 32; i++)
		{
			B_tmp[i] = 0;
		}
		
		int lda_new = ac_ld;
		int ldb_new = 4;
		int ldc_new = ac_ld;
		
		for (int j=0; j < K_; j++)
		{
			for (int i = 0; i < lda; i++) 
			{
				A_tmp[j*lda_new+i] = A[j*lda + i];
			}
		}
		for (int j=0; j < N_; j++)
		{
			for (int i = 0; i < ldb; i++) 
			{
				B_tmp[j*ldb_new+i] = B[j*ldb + i];
			}
		}
		for (int j=0; j < N_; j++)
		{
			for (int i = 0; i < ldc; i++) 
			{
				C_tmp[j*ldc_new+i] = C[j*ldc + i];
			}
		}
		
		for (int n = 0; n < N_; n+=8) {
			for (int m = 0; m < M_; m+=8) {
			  inner512_8x8(K_, &A_tmp(m,0,lda), lda_new, &B_tmp(0,n,ldb),ldb_new ,&C_tmp(m,n,ldc), ldc_new, alpha_v, beta_v);
			}
		}
		
		for (int j=0; j < N_; j++)
		{
			for (int i = 0; i < ldc; i++) 
			{
				C[j*ldc+i] = C_tmp[j*ldc_new + i];
			}
		}
	}
	
	
	// put everything in buffers of 8x8 multiples, twice as slow as completely unoptimized code, so the idea was aborted
	/*
		int M_new = ceil((double) M_/8.0)*8;
		int N_new = ceil((double)N_/8.0)*8;
		int K_new = ceil((double)K_/8.0)*8;
	
		//std::cout << "M_: " << M_ << " M_new: " << M_new << std::endl;
		//std::cout << "N_: " << M_ << " N_new: " << M_new << std::endl;
		//std::cout << "K_: " << M_ << " K_new: " << M_new << std::endl;
		
		//double ceilv = ceil((double) M_/8.0);
		
		//std::cout << "ceil(M_/8): " << ceilv << std::endl;
	
		const __m512d alpha_v = _mm512_set1_pd(alpha);
		const __m512d beta_v = _mm512_set1_pd(beta);
		
		int A_size = M_new*K_new;
		int B_size = K_new*N_new;
		int C_size = M_new*N_new;
		
		double A_tmp[A_size];
		double B_tmp[B_size];
		double C_tmp[C_size];
		
		#pragma vector always
		for (int i = 0; i < A_size; i++) 
		{
			A_tmp[i] = 0;
		}
		#pragma vector always
		for (int i = 0; i < B_size; i++)
		{
			B_tmp[i] = 0;
		}
		#pragma vector always
		for (int i = 0; i < C_size; i++)
		{
			C_tmp[i] = 0;
		}
		
		int lda_new = M_new;
		int ldb_new = K_new;
		int ldc_new = M_new;
		
		for (int j=0; j < K_; j++)
		{
			for (int i = 0; i < M_; i++) 
			{
				A_tmp[j*lda_new+i] = A[j*lda + i];
			}
		}
		for (int j=0; j < N_; j++)
		{
			for (int i = 0; i < K_; i++) 
			{
				B_tmp[j*ldb_new+i] = B[j*ldb + i];
			}
		}
		for (int j=0; j < N_; j++)
		{
			for (int i = 0; i < M_; i++) 
			{
				C_tmp[j*ldc_new+i] = C[j*ldc + i];
			}
		}
		
		for (int n = 0; n < N_; n+=8) {
			for (int m = 0; m < M_; m+=8) {
			  inner512_8x8(K_, &A_tmp(m,0,lda), lda_new, &B_tmp(0,n,ldb),ldb_new ,&C_tmp(m,n,ldc), ldc_new, alpha_v, beta_v);
			}
		}
		
		for (int j=0; j < N_; j++)
		{
			for (int i = 0; i < M_; i++) 
			{
				C[j*ldc+i] = C_tmp[j*ldc_new + i];
			}
		}*/
}


/**
 * Kernel computing C += A*B
 * Uses intrinsics for computation
 */
void inner512_8x8(int K_, 
	const double* A, 
	int lda, 
	const double* B, 
	int ldb, 
	double* C, 
	int ldc, 
	const __m512d alpha_v, 
	const __m512d beta_v) {
	
	// vector registers
	__m512d
	c_00_10_20_30_40_50_60_70_v,
	c_01_11_21_31_41_51_61_71_v,
	c_02_12_22_32_42_52_62_72_v,
	c_03_13_23_33_43_53_63_73_v,
	c_04_14_24_34_44_54_64_74_v,
	c_05_15_25_35_45_55_65_75_v,
	c_06_16_26_36_46_56_66_76_v,
	c_07_17_27_37_47_57_67_77_v,
	a_0k_1k_2k_3k_4k_5k_6k_7k_v,
	b_k0_v2, b_k1_v2, b_k2_v2, b_k3_v2,
	b_k4_v2, b_k5_v2, b_k6_v2, b_k7_v2,
	C_0_0, C_0_1, C_0_2, C_0_3, C_0_4, C_0_5, C_0_6, C_0_7,
	tmp_v;

	C_0_0 = _mm512_load_pd( (double *) &C(0,0,ldc));
	C_0_1 = _mm512_load_pd( (double *) &C(0,1,ldc));
	C_0_2 = _mm512_load_pd( (double *) &C(0,2,ldc));
	C_0_3 = _mm512_load_pd( (double *) &C(0,3,ldc));
	C_0_4 = _mm512_load_pd( (double *) &C(0,4,ldc));
	C_0_5 = _mm512_load_pd( (double *) &C(0,5,ldc));
	C_0_6 = _mm512_load_pd( (double *) &C(0,6,ldc));
	C_0_7 = _mm512_load_pd( (double *) &C(0,7,ldc));
	
	c_00_10_20_30_40_50_60_70_v = _mm512_setzero_pd();
	c_01_11_21_31_41_51_61_71_v = _mm512_setzero_pd();
	c_02_12_22_32_42_52_62_72_v = _mm512_setzero_pd();
	c_03_13_23_33_43_53_63_73_v = _mm512_setzero_pd();
	c_04_14_24_34_44_54_64_74_v = _mm512_setzero_pd();
	c_05_15_25_35_45_55_65_75_v = _mm512_setzero_pd();
	c_06_16_26_36_46_56_66_76_v = _mm512_setzero_pd();
	c_07_17_27_37_47_57_67_77_v = _mm512_setzero_pd();
	
	// int b = 0;
	
	#pragma vector always
	for (int k = 0; k < K_; k+=4) {
		
		// Unroll 0
		a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k,lda));
		
		/** we left this here to show that we tried to use packing for A and B*/
		//use for packing B
		//b_k0_v2 = _mm512_set1_pd(B[b]);
		//b_k1_v2 = _mm512_set1_pd(B[b+1]);
		//b_k2_v2 = _mm512_set1_pd(B[b+2]);
		//b_k3_v2 = _mm512_set1_pd(B[b+3]);
		//b_k4_v2 = _mm512_set1_pd(B[b+4]);
		//b_k5_v2 = _mm512_set1_pd(B[b+5]);
		//b_k6_v2 = _mm512_set1_pd(B[b+6]);
		//b_k7_v2 = _mm512_set1_pd(B[b+7]);	
	    //b+=8;
		
		b_k0_v2 = _mm512_set1_pd(B(k,0,ldb));
		b_k1_v2 = _mm512_set1_pd(B(k,1,ldb));
		b_k2_v2 = _mm512_set1_pd(B(k,2,ldb));
		b_k3_v2 = _mm512_set1_pd(B(k,3,ldb));
		b_k4_v2 = _mm512_set1_pd(B(k,4,ldb));
		b_k5_v2 = _mm512_set1_pd(B(k,5,ldb));
		b_k6_v2 = _mm512_set1_pd(B(k,6,ldb));
		b_k7_v2 = _mm512_set1_pd(B(k,7,ldb));
		
		c_00_10_20_30_40_50_60_70_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k0_v2, c_00_10_20_30_40_50_60_70_v);
		c_01_11_21_31_41_51_61_71_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k1_v2, c_01_11_21_31_41_51_61_71_v);
		c_02_12_22_32_42_52_62_72_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k2_v2, c_02_12_22_32_42_52_62_72_v);
		c_03_13_23_33_43_53_63_73_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k3_v2, c_03_13_23_33_43_53_63_73_v);
		c_04_14_24_34_44_54_64_74_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k4_v2, c_04_14_24_34_44_54_64_74_v);
		c_05_15_25_35_45_55_65_75_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k5_v2, c_05_15_25_35_45_55_65_75_v);
		c_06_16_26_36_46_56_66_76_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k6_v2, c_06_16_26_36_46_56_66_76_v);
		c_07_17_27_37_47_57_67_77_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k7_v2, c_07_17_27_37_47_57_67_77_v);
		
		// Unroll 1
		a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+1,lda));
		
		b_k0_v2 = _mm512_set1_pd(B(k+1,0,ldb));
		b_k1_v2 = _mm512_set1_pd(B(k+1,1,ldb));
		b_k2_v2 = _mm512_set1_pd(B(k+1,2,ldb));
		b_k3_v2 = _mm512_set1_pd(B(k+1,3,ldb));
		b_k4_v2 = _mm512_set1_pd(B(k+1,4,ldb));
		b_k5_v2 = _mm512_set1_pd(B(k+1,5,ldb));
		b_k6_v2 = _mm512_set1_pd(B(k+1,6,ldb));
		b_k7_v2 = _mm512_set1_pd(B(k+1,7,ldb));
		
		c_00_10_20_30_40_50_60_70_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k0_v2, c_00_10_20_30_40_50_60_70_v);
		c_01_11_21_31_41_51_61_71_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k1_v2, c_01_11_21_31_41_51_61_71_v);
		c_02_12_22_32_42_52_62_72_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k2_v2, c_02_12_22_32_42_52_62_72_v);
		c_03_13_23_33_43_53_63_73_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k3_v2, c_03_13_23_33_43_53_63_73_v);
		c_04_14_24_34_44_54_64_74_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k4_v2, c_04_14_24_34_44_54_64_74_v);
		c_05_15_25_35_45_55_65_75_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k5_v2, c_05_15_25_35_45_55_65_75_v);
		c_06_16_26_36_46_56_66_76_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k6_v2, c_06_16_26_36_46_56_66_76_v);
		c_07_17_27_37_47_57_67_77_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k7_v2, c_07_17_27_37_47_57_67_77_v);
		
		// Unroll 2
		a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+2,lda));
		
		b_k0_v2 = _mm512_set1_pd(B(k+2,0,ldb));
		b_k1_v2 = _mm512_set1_pd(B(k+2,1,ldb));
		b_k2_v2 = _mm512_set1_pd(B(k+2,2,ldb));
		b_k3_v2 = _mm512_set1_pd(B(k+2,3,ldb));
		b_k4_v2 = _mm512_set1_pd(B(k+2,4,ldb));
		b_k5_v2 = _mm512_set1_pd(B(k+2,5,ldb));
		b_k6_v2 = _mm512_set1_pd(B(k+2,6,ldb));
		b_k7_v2 = _mm512_set1_pd(B(k+2,7,ldb));
		
		c_00_10_20_30_40_50_60_70_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k0_v2, c_00_10_20_30_40_50_60_70_v);
		c_01_11_21_31_41_51_61_71_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k1_v2, c_01_11_21_31_41_51_61_71_v);
		c_02_12_22_32_42_52_62_72_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k2_v2, c_02_12_22_32_42_52_62_72_v);
		c_03_13_23_33_43_53_63_73_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k3_v2, c_03_13_23_33_43_53_63_73_v);
		c_04_14_24_34_44_54_64_74_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k4_v2, c_04_14_24_34_44_54_64_74_v);
		c_05_15_25_35_45_55_65_75_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k5_v2, c_05_15_25_35_45_55_65_75_v);
		c_06_16_26_36_46_56_66_76_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k6_v2, c_06_16_26_36_46_56_66_76_v);
		c_07_17_27_37_47_57_67_77_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k7_v2, c_07_17_27_37_47_57_67_77_v);
		
		// Unroll 3
		a_0k_1k_2k_3k_4k_5k_6k_7k_v = _mm512_load_pd( (double *) &A(0,k+3,lda));
		
		b_k0_v2 = _mm512_set1_pd(B(k+3,0,ldb));
		b_k1_v2 = _mm512_set1_pd(B(k+3,1,ldb));
		b_k2_v2 = _mm512_set1_pd(B(k+3,2,ldb));
		b_k3_v2 = _mm512_set1_pd(B(k+3,3,ldb));
		b_k4_v2 = _mm512_set1_pd(B(k+3,4,ldb));
		b_k5_v2 = _mm512_set1_pd(B(k+3,5,ldb));
		b_k6_v2 = _mm512_set1_pd(B(k+3,6,ldb));
		b_k7_v2 = _mm512_set1_pd(B(k+3,7,ldb));
		
		c_00_10_20_30_40_50_60_70_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k0_v2, c_00_10_20_30_40_50_60_70_v);
		c_01_11_21_31_41_51_61_71_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k1_v2, c_01_11_21_31_41_51_61_71_v);
		c_02_12_22_32_42_52_62_72_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k2_v2, c_02_12_22_32_42_52_62_72_v);
		c_03_13_23_33_43_53_63_73_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k3_v2, c_03_13_23_33_43_53_63_73_v);
		c_04_14_24_34_44_54_64_74_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k4_v2, c_04_14_24_34_44_54_64_74_v);
		c_05_15_25_35_45_55_65_75_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k5_v2, c_05_15_25_35_45_55_65_75_v);
		c_06_16_26_36_46_56_66_76_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k6_v2, c_06_16_26_36_46_56_66_76_v);
		c_07_17_27_37_47_57_67_77_v = _mm512_fmadd_pd(a_0k_1k_2k_3k_4k_5k_6k_7k_v, b_k7_v2, c_07_17_27_37_47_57_67_77_v);
		
	}

	// Multiply old result by beta
	C_0_0 = _mm512_mul_pd(C_0_0, beta_v);
	C_0_1 = _mm512_mul_pd(C_0_1, beta_v);
	C_0_2 = _mm512_mul_pd(C_0_2, beta_v);
	C_0_3 = _mm512_mul_pd(C_0_3, beta_v);
	C_0_4 = _mm512_mul_pd(C_0_4, beta_v);
	C_0_5 = _mm512_mul_pd(C_0_5, beta_v);
	C_0_6 = _mm512_mul_pd(C_0_6, beta_v);
	C_0_7 = _mm512_mul_pd(C_0_7, beta_v);

	// Multiply calculated result by alpha and add to old result
	C_0_0 = _mm512_fmadd_pd(c_00_10_20_30_40_50_60_70_v, alpha_v, C_0_0);
	C_0_1 = _mm512_fmadd_pd(c_01_11_21_31_41_51_61_71_v, alpha_v, C_0_1);
	C_0_2 = _mm512_fmadd_pd(c_02_12_22_32_42_52_62_72_v, alpha_v, C_0_2);
	C_0_3 = _mm512_fmadd_pd(c_03_13_23_33_43_53_63_73_v, alpha_v, C_0_3);
	C_0_4 = _mm512_fmadd_pd(c_04_14_24_34_44_54_64_74_v, alpha_v, C_0_4);
	C_0_5 = _mm512_fmadd_pd(c_05_15_25_35_45_55_65_75_v, alpha_v, C_0_5);
	C_0_6 = _mm512_fmadd_pd(c_06_16_26_36_46_56_66_76_v, alpha_v, C_0_6);
	C_0_7 = _mm512_fmadd_pd(c_07_17_27_37_47_57_67_77_v, alpha_v, C_0_7);

	
	// Store stuff back in the matrix. We can join this block with the above but will it lead to better performance?
	_mm512_store_pd(&C(0,0,ldc), C_0_0);
	_mm512_store_pd(&C(0,1,ldc), C_0_1);
	_mm512_store_pd(&C(0,2,ldc), C_0_2);
	_mm512_store_pd(&C(0,3,ldc), C_0_3);
	_mm512_store_pd(&C(0,4,ldc), C_0_4);
	_mm512_store_pd(&C(0,5,ldc), C_0_5);
	_mm512_store_pd(&C(0,6,ldc), C_0_6);
	_mm512_store_pd(&C(0,7,ldc), C_0_7);
}
