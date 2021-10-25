\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 b_a_1[0,0] + 0 b_a_2[0,0] + 0 b_a_3[0,0] + 0 b_a_4[0,0] + 0 b_a_7[0,0]
   + 0 b_g_10[0,0] + 0 b_g_11[0,0] + 0 b_g_12[0,0] + 0 b_g_13[0,0]
   + 0 b_g_14[0,0] + 0 b_g_17[0,0]
Subject To
 R0: b_a = 1
 R1: - 10000 b_a_5[0,0] - 1.644853626951472 b_a_5_sigma >= -10000.2
 R2: - 10000 b_a_5[0,0] - 1.644853626951472 b_a_5_sigma <= -0.2001
 R3: - 10000 b_g_9[0,0] + 2.326347874040841 b_g_9_sigma >= -9999.8999
 R4: - 10000 b_g_9[0,0] + 2.326347874040841 b_g_9_sigma <= 0.1
 R5: - 10000 b_g_15[0,0] - 1.644853626951472 b_g_15_sigma >= -10000.2
 R6: - 10000 b_g_15[0,0] - 1.644853626951472 b_g_15_sigma <= -0.2001
 R7: - 10000 b_g_18[0,0] - 1.644853626951472 b_g_18_sigma >= -10000.2
 R8: - 10000 b_g_18[0,0] - 1.644853626951472 b_g_18_sigma <= -0.2001
 SAT: b_g = 0
 qc0: b_a_5_variance + [ ] = 0.01
 qc1: b_a_6_variance + [ - sigma2 ^2 ] = 0
 qc2: c - 10000 b_a_6[0,0] + 1.461206020336355 b_a_6_sigma + [
   - 3.146865080561092 p * b_a_6_sigma ] >= -10000
 qc3: c - 10000 b_a_6[0,0] + 1.461206020336355 b_a_6_sigma + [
   - 3.146865080561092 p * b_a_6_sigma ] <= -0.0001
 qc4: b_g_9_variance + [ - 0.25 sigma2 ^2 ] = 0.0025
 qc5: b_g_15_variance + [ ] = 0.01
 qc6: b_g_16_variance + [ - sigma2 ^2 ] = 0
 qc7: c - 10000 b_g_16[0,0] + 1.461206020336355 b_g_16_sigma + [
   - 3.146865080561092 p * b_g_16_sigma ] >= -10000
 qc8: c - 10000 b_g_16[0,0] + 1.461206020336355 b_g_16_sigma + [
   - 3.146865080561092 p * b_g_16_sigma ] <= -0.0001
 qc9: b_g_18_variance + [ ] = 0.01
 qc10: b_g_19_variance + [ - sigma2 ^2 ] = 0
 qc11: c - 10000 b_g_19[0,0] + 1.461206020336355 b_g_19_sigma + [
   - 3.146865080561092 p * b_g_19_sigma ] >= -10000
 qc12: c - 10000 b_g_19[0,0] + 1.461206020336355 b_g_19_sigma + [
   - 3.146865080561092 p * b_g_19_sigma ] <= -0.0001
Bounds
 p <= 0.5
 0.1 <= sigma2 <= 0.3
 -4 <= c <= 4
 b_a_5_variance <= 10000
 b_a_5_sigma <= 10000
 b_a_6_variance <= 10000
 b_a_6_sigma <= 10000
 b_g_9_variance <= 10000
 b_g_9_sigma <= 10000
 b_g_15_variance <= 10000
 b_g_15_sigma <= 10000
 b_g_16_variance <= 10000
 b_g_16_sigma <= 10000
 b_g_18_variance <= 10000
 b_g_18_sigma <= 10000
 b_g_19_variance <= 10000
 b_g_19_sigma <= 10000
Binaries
 b_a b_a_1[0,0] b_a_2[0,0] b_a_3[0,0] b_a_4[0,0] b_a_5[0,0] b_a_6[0,0]
 b_a_7[0,0] b_g b_g_9[0,0] b_g_10[0,0] b_g_11[0,0] b_g_12[0,0] b_g_13[0,0]
 b_g_14[0,0] b_g_15[0,0] b_g_16[0,0] b_g_17[0,0] b_g_18[0,0] b_g_19[0,0]
General Constraints
 GC0: b_a_2[0,0] = OR ( b_a_3[0,0] , b_a_4[0,0] )
 GC1: b_a_5_sigma = POW ( b_a_5_variance ^ 0.5 )
 GC2: b_a_6_sigma = POW ( b_a_6_variance ^ 0.5 )
 GC3: b_a_1[0,0] = AND ( b_a_2[0,0] , b_a_5[0,0] , b_a_6[0,0] )
 GC4: b_a = AND ( b_a_1[0,0] , b_a_7[0,0] )
 GC5: b_g_9_sigma = POW ( b_g_9_variance ^ 0.5 )
 GC6: b_g_12[0,0] = OR ( b_g_13[0,0] , b_g_14[0,0] )
 GC7: b_g_15_sigma = POW ( b_g_15_variance ^ 0.5 )
 GC8: b_g_16_sigma = POW ( b_g_16_variance ^ 0.5 )
 GC9: b_g_11[0,0] = AND ( b_g_12[0,0] , b_g_15[0,0] , b_g_16[0,0] )
 GC10: b_g_18_sigma = POW ( b_g_18_variance ^ 0.5 )
 GC11: b_g_19_sigma = POW ( b_g_19_variance ^ 0.5 )
 GC12: b_g_17[0,0] = AND ( b_g_18[0,0] , b_g_19[0,0] )
 GC13: b_g_10[0,0] = OR ( b_g_11[0,0] , b_g_17[0,0] )
 GC14: b_g = AND ( b_g_9[0,0] , b_g_10[0,0] )
End
