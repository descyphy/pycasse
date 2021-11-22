\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 a__0 + 0 a__1 + 0 a__2 + 0 a__3 + 0 b_g_3[0,0] + 0 b_g_3[0,1]
   + 0 b_g_3[0,2] + 0 b_g_3[0,3]
Subject To
 R0: b_0 = 1
 R1: - s__0 - 10000 b_0 >= -10000
 R2: - s__0 - 10000 b_0 <= -0.0001
 R3: s__0 - 10000 b_0 >= -10000
 R4: s__0 - 10000 b_0 <= -0.0001
 R5: b_1 = 1
 R6: - v__0 - 10000 b_1 >= -10000
 R7: - v__0 - 10000 b_1 <= -0.0001
 R8: v__0 - 10000 b_1 >= -10000
 R9: v__0 - 10000 b_1 <= -0.0001
 R10: b_g = 1
Bounds
 -100 <= s__0 <= 2000
 -5 <= v__0 <= 10
 -1 <= a__0 <= 1
 -1 <= a__1 <= 1
 -1 <= a__2 <= 1
 -1 <= a__3 <= 1
Binaries
 b_0 b_1 b_g b_g_3[0,0] b_g_3[0,1] b_g_3[0,2] b_g_3[0,3]
General Constraints
 GC0: b_g = OR ( b_g_3[0,0] , b_g_3[0,1] , b_g_3[0,2] , b_g_3[0,3] )
End
