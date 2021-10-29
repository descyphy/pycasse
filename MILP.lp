\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 
Subject To
 R0: b_a = 1
 R1: b_a = 1
 R2: - mean - 10000 b_g - 1.2815515655446 b_g_sigma >= -10000
 R3: - mean - 10000 b_g - 1.2815515655446 b_g_sigma <= -0.0001
 UNSAT: b_g = 1
 qc0: b_g_variance + [ - sigma ^2 ] = 0
Bounds
 -0.02890625 <= mean <= -0.028125
 0.0223046875 <= sigma <= 0.02265625
 b_g_variance <= 10000
 b_g_sigma <= 10000
Binaries
 b_a b_g
General Constraints
 GC0: b_g_sigma = POW ( b_g_variance ^ 0.5 )
End
