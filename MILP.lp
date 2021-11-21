\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 
Subject To
 R0: b_a >= 0
 R1: b_a_1[0,0] = -10000
 R2: b_a_2[0,0] = 10000
 R3: b_g_4[0,0] + 1.2815515655446 b_g_4_sigma = 1.5
 UNSAT: b_g >= 0
 qc0: b_g_4_variance + [ - sigma ^2 ] = 0
 qc1: b_g_5_variance + [ - sigma ^2 ] = 0
 qc2: b_g_5[0,0] + 1.685659060224737 b_g_5_sigma + [
   3.146865080561092 p * b_g_5_sigma ] = 1.5
Bounds
 0.05 <= sigma <= 2
 p <= 0.5
 -10000 <= b_a <= 10000
 -10000 <= b_a_1[0,0] <= 10000
 -10000 <= b_a_2[0,0] <= 10000
 -10000 <= b_g <= 10000
 -10000 <= b_g_4[0,0] <= 10000
 b_g_4_variance <= 10000
 b_g_4_sigma <= 10000
 -10000 <= b_g_5[0,0] <= 10000
 b_g_5_variance <= 10000
 b_g_5_sigma <= 10000
General Constraints
 GC0: b_a = MAX ( b_a_1[0,0] , b_a_2[0,0] )
 GC1: b_g_4_sigma = POW ( b_g_4_variance ^ 0.5 )
 GC2: b_g_5_sigma = POW ( b_g_5_variance ^ 0.5 )
 GC3: b_g = MIN ( b_g_4[0,0] , b_g_5[0,0] )
End
