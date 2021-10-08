\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 
Subject To
 R0: node_0_0 = 10000
 R1: node_0_0 >= 0
 R2: node_1_0 - contract_1_0 + 2.530611841988826 contract_0_0
   = 1.266158476508544
 SAT: node_1_0 <= -0.0001
Bounds
 -10000 <= node_0_0 <= 10000
 -10000 <= node_1_0 <= 10000
 0.25 <= contract_1_0 <= 0.5
 0.546875 <= contract_0_0 <= 0.5625
End
