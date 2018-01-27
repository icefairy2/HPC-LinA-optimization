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

	//std::cout << "M: " << M_ << " N: " << N_ << " K: " << K_ << std::endl;
	//std::cout << "lda: " << lda << " ldb: " << ldb << " ldc: " << ldc << std::endl;
	
	if (M_ == 3 && N_ == 3 && K_ == 3 && alpha == 1 && beta == 0) {
		//std::cout << "execute generated" << std::endl;
		knlm3n3k3a1b0(A,B,C);
	}
	else {
		DGEMM_old(M_, N_, K_, alpha, A, lda, B, ldb, beta, C, ldc);
	}
	//DGEMM_old(M_, N_, K_, alpha, A, lda, B, ldb, beta, C, ldc);
	
	//if (M_ > 8 && N_ > 8 && K_ > 8) {
	//	std::cout << "larger than 8" << std::endl;
	//}
	
	//if (M_ < 8 || N_ < 8 || K_ < 8) {
	//	DGEMM_old(M_, N_, K_, alpha, A, lda, B, ldb, beta, C, ldc);
	//}
	//else {
	/*
	double A_tmp[512];
	double B_tmp[512];
	double C_tmp1[512];
	double C_tmp2[512];

	for (int i = 0; i < K_*M_; i++) {
		A_tmp[i] = A[i];
	}
	for (int i = 0; i < K_*N_; i++) {
		B_tmp[i] = B[i];
	}
	for (int i = 0; i < N_*M_; i++) {
		C_tmp1[i] = C[i];
		C_tmp2[i] = C[i];
	}*/
	//std::cout << "temp setup" << std::endl;
	
	//const __m512d alpha_v = _mm512_set1_pd(alpha);
	//const __m512d beta_v = _mm512_set1_pd(beta);
	
	/*
	for (int n = 0; n < N_; n+=8) {
		for (int m = 0; m < M_; m+=8) {
		  inner512_8x8(K_, &A(m,0,lda), lda, &B(0,n,ldb),ldb ,&C(m,n,ldc), ldc, alpha_v, beta_v);
		}
	}*/
	//}

	/*
	for (int n = 0; n < N_; n+=8) {
		for (int m = 0; m < M_; m+=8) {
		  inner512_8x8(K_, &A_tmp(m,0,lda), lda, &B_tmp(0,n,ldb),ldb ,&C_tmp1(m,n,ldc), ldc, alpha_v, beta_v);
		}
	}*/
	
	/*int errors = 0;
	for (int i = 0; i < N_*M_; i++) {
		if (std::abs(C_tmp2[i] - C_tmp1[i]) > 1e-8) {
			errors++;
			std::cout << "tmp1: " << C_tmp1[i] << " tmp2: " << C_tmp2[i] << std::endl;
		}
			
	}
	if (errors > 0)
	std::cout << "errors: " << errors << std::endl;*/
	//std::cout << "calc" << std::endl;
	/*
	for (int i = 0; i < N_*M_; i++) {
		//std::cout << "i: " << i << std::endl;
		//std::cout << "C_tmp[i] " << C_tmp[i] << std::endl;
		//std::cout << "C[i] " << C[i] << std::endl;
		C[i] = C_tmp1[i];
		//std::cout << "C[i] new" << C[i] << std::endl;
	}*/
	
	//std::cout << "write back" << std::endl;
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
