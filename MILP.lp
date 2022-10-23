\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  b_g
Subject To
 R0: b_a >= 0
 R1: b_a = 10000
 R2: mean + b_g + 2.326347874040841 b_g_sigma__0 = 0
 R3: b_mean >= 0
 R4: mean + b_mean = -0.1
 R5: mean = -0.1
 R6: b_sigma >= 0
 R7: sigma + b_sigma = 0.0409375
 R8: sigma = 0.0409375
 qc0: b_g_variance__0 + [ - sigma ^2 ] = 0
Bounds
 -0.1 <= mean <= -0.096875
 0.03953125 <= sigma <= 0.0409375
 -10000 <= b_a <= 10000
 -10000 <= b_g <= 10000
 b_g_variance__0 <= 10000
 b_g_sigma__0 <= 10000
 -10000 <= b_mean <= 10000
 -10000 <= b_sigma <= 10000
General Constraints
 GC0: b_g_sigma__0 = POW ( b_g_variance__0 ^ 0.5 )
End
