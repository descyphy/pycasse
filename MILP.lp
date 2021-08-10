\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 node_1_0 + 0 node_4_0
Subject To
 R0: 10000 node_2_0 - contract_2_0 <= 9995
 R1: 10000 node_2_0 - contract_2_0 >= -5
 R2: contract_2_0 + 10000 node_3_0 <= 10009
 R3: contract_2_0 + 10000 node_3_0 >= 9
 R4: 10000 node_5_0 + contract_1_0 <= 10002.9999
 R5: 10000 node_5_0 + contract_1_0 >= 2.9999
 R6: - contract_1_0 + 10000 node_6_0 <= 9990.999900000001
 R7: - contract_1_0 + 10000 node_6_0 >= -9.0001
 R8: node_0_0 = 1
Bounds
 -10000 <= contract_2_0 <= 10000
 -10000 <= contract_1_0 <= 10000
Binaries
 node_0_0 node_1_0 node_2_0 node_3_0 node_4_0 node_5_0 node_6_0
General Constraints
 GC0: node_1_0 = AND ( node_2_0 , node_3_0 )
 GC1: node_4_0 = OR ( node_5_0 , node_6_0 )
 GC2: node_0_0 = AND ( node_1_0 , node_4_0 )
End
