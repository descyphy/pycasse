\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 u[0,0]
Subject To
 R0: b_a >= 0
 R1: b_a = 10000
 R2: 2.530611841988826 p - c + b_g = 1.266158476508544
 SAT: b_g <= -0.0001
Bounds
 -0.1 <= u[0,0] <= 0.1
 0.546875 <= p <= 0.5625
 0.25 <= c <= 0.5
 -10000 <= b_a <= 10000
 -10000 <= b_g <= 10000
End
