\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 node_1_0 + 0 node_3_0
Subject To
 R0: 10000 node_2_0 - contract_0_0 <= 9995
 R1: 10000 node_2_0 - contract_0_0 >= -5
 R2: 10000 node_2_1 - contract_0_1 <= 9995
 R3: 10000 node_2_1 - contract_0_1 >= -5
 R4: 10000 node_2_2 - contract_0_2 <= 9995
 R5: 10000 node_2_2 - contract_0_2 >= -5
 R6: 10000 node_2_3 - contract_0_3 <= 9995
 R7: 10000 node_2_3 - contract_0_3 >= -5
 R8: contract_0_1 + 10000 node_4_1 <= 10003.9999
 R9: contract_0_1 + 10000 node_4_1 >= 3.9999
 R10: contract_0_2 + 10000 node_4_2 <= 10003.9999
 R11: contract_0_2 + 10000 node_4_2 >= 3.9999
 R12: node_0_0 = 1
Bounds
 -10000 <= contract_0_0 <= 10000
 -10000 <= contract_0_1 <= 10000
 -10000 <= contract_0_2 <= 10000
 -10000 <= contract_0_3 <= 10000
Binaries
 node_0_0 node_1_0 node_2_0 node_2_1 node_2_2 node_2_3 node_3_0 node_4_1
 node_4_2
General Constraints
 GC0: node_1_0 = AND ( node_2_0 , node_2_1 , node_2_2 , node_2_3 )
 GC1: node_3_0 = AND ( node_4_1 , node_4_2 )
 GC2: node_0_0 = AND ( node_1_0 , node_3_0 )
End
