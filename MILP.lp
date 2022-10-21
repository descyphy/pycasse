\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  b_g
Subject To
 R0: b_a >= 0
 R1: b_a_1[0,0] = 10000
 R2: b_a_2[0,0] = 10000
 R3: b_g_4[0,0] + 1.2815515655446 b_g_4_sigma__0 = 1.5
 R4: - c + b_g_5[0,0] + 1.2815515655446 b_g_5_sigma__0 = 0
 R5: b_sigma >= 0
 R6: sigma + b_sigma = 1.146875
 R7: sigma = 1.146875
 R8: b_c >= 0
 R9: c + b_c = 1.5
 R10: c = 1.5
 qc0: b_g_4_variance__0 + [ - sigma ^2 ] = 0
 qc1: b_g_5_variance__0 + [ - sigma ^2 ] = 0
Bounds
 1.025 <= sigma <= 1.146875
 1.5 <= c <= 1.625
 -10000 <= b_a <= 10000
 -10000 <= b_a_1[0,0] <= 10000
 -10000 <= b_a_2[0,0] <= 10000
 -10000 <= b_g <= 10000
 -10000 <= b_g_4[0,0] <= 10000
 b_g_4_variance__0 <= 10000
 b_g_4_sigma__0 <= 10000
 -10000 <= b_g_5[0,0] <= 10000
 b_g_5_variance__0 <= 10000
 b_g_5_sigma__0 <= 10000
 -10000 <= b_sigma <= 10000
 -10000 <= b_c <= 10000
General Constraints
 GC0: b_a = MIN ( b_a_1[0,0] , b_a_2[0,0] )
 GC1: b_g_4_sigma__0 = POW ( b_g_4_variance__0 ^ 0.5 )
 GC2: b_g_5_sigma__0 = POW ( b_g_5_variance__0 ^ 0.5 )
 GC3: b_g = MIN ( b_g_4[0,0] , b_g_5[0,0] )
End
