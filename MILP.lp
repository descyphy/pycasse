\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 b_a_1[0,0] + 0 b_a_2[0,0] + 0 b_a_3[0,0] + 0 b_a_4[0,0] + 0 b_a_7[0,0]
   + 0 b_g_10[0,0]
Subject To
 R0: b_a = 1
 R1: - 10000 b_a_5[0,0] - 1.644853626951472 b_a_5_sigma >= -10000.2
 R2: - 10000 b_a_5[0,0] - 1.644853626951472 b_a_5_sigma <= -0.2001
 R3: - 10000 b_g_9[0,0] + 2.326347874040841 b_g_9_sigma >= -9999.8999
 R4: - 10000 b_g_9[0,0] + 2.326347874040841 b_g_9_sigma <= 0.1
 R5: - 10000 b_g_11[0,0] - 1.644853626951472 b_g_11_sigma >= -10000.2
 R6: - 10000 b_g_11[0,0] - 1.644853626951472 b_g_11_sigma <= -0.2001
 UNSAT: b_g = 1
 qc0: b_a_5_variance + [ ] = 0.01
 qc1: b_a_6_variance + [ - sigma2 ^2 ] = 0
 qc2: c - 10000 b_a_6[0,0] + 469.2411377681197 b_a_6_sigma + [
   - 477.4506739197212 p * b_a_6_sigma ] >= -10000
 qc3: c - 10000 b_a_6[0,0] + 469.2411377681197 b_a_6_sigma + [
   - 477.4506739197212 p * b_a_6_sigma ] <= -0.0001
 qc4: b_g_9_variance + [ - 0.25 sigma2 ^2 ] = 0.0025
 qc5: b_g_11_variance + [ ] = 0.01
 qc6: b_g_12_variance + [ - sigma2 ^2 ] = 0
 qc7: c - 10000 b_g_12[0,0] + 469.2411377681197 b_g_12_sigma + [
   - 477.4506739197212 p * b_g_12_sigma ] >= -10000
 qc8: c - 10000 b_g_12[0,0] + 469.2411377681197 b_g_12_sigma + [
   - 477.4506739197212 p * b_g_12_sigma ] <= -0.0001
Bounds
 0.9875 <= p <= 1
 0.04625 <= sigma2 <= 0.0825
 0.4375 <= c <= 0.5
 b_a_5_variance <= 10000
 b_a_5_sigma <= 10000
 b_a_6_variance <= 10000
 b_a_6_sigma <= 10000
 b_g_9_variance <= 10000
 b_g_9_sigma <= 10000
 b_g_11_variance <= 10000
 b_g_11_sigma <= 10000
 b_g_12_variance <= 10000
 b_g_12_sigma <= 10000
Binaries
 b_a b_a_1[0,0] b_a_2[0,0] b_a_3[0,0] b_a_4[0,0] b_a_5[0,0] b_a_6[0,0]
 b_a_7[0,0] b_g b_g_9[0,0] b_g_10[0,0] b_g_11[0,0] b_g_12[0,0]
General Constraints
 GC0: b_a_2[0,0] = OR ( b_a_3[0,0] , b_a_4[0,0] )
 GC1: b_a_5_sigma = POW ( b_a_5_variance ^ 0.5 )
 GC2: b_a_6_sigma = POW ( b_a_6_variance ^ 0.5 )
 GC3: b_a_1[0,0] = AND ( b_a_2[0,0] , b_a_5[0,0] , b_a_6[0,0] )
 GC4: b_a = AND ( b_a_1[0,0] , b_a_7[0,0] )
 GC5: b_g_9_sigma = POW ( b_g_9_variance ^ 0.5 )
 GC6: b_g_11_sigma = POW ( b_g_11_variance ^ 0.5 )
 GC7: b_g_12_sigma = POW ( b_g_12_variance ^ 0.5 )
 GC8: b_g_10[0,0] = AND ( b_g_11[0,0] , b_g_12[0,0] )
 GC9: b_g = AND ( b_g_9[0,0] , b_g_10[0,0] )
End
