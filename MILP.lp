\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 
Subject To
 R0: b_a >= 0
 R1: b_a_1[0,0] = 10000
 R2: b_a_2[0,0] = 10000
 R3: b_g_4[0,0] = 0.2184484344553996
 R4: - c + 209.4139054596916 p + b_g_5[0,0] = 3.28177937965769
 SAT: b_g <= -0.0001
Bounds
 -2.796875 <= c <= -2.78125
 9.765625e-04 <= p <= 0.001953125
 -10000 <= b_a <= 10000
 -10000 <= b_a_1[0,0] <= 10000
 -10000 <= b_a_2[0,0] <= 10000
 -10000 <= b_g <= 10000
 -10000 <= b_g_4[0,0] <= 10000
 -10000 <= b_g_5[0,0] <= 10000
General Constraints
 GC0: b_a = MAX ( b_a_1[0,0] , b_a_2[0,0] )
 GC1: b_g = MAX ( b_g_4[0,0] , b_g_5[0,0] )
End
