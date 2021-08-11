\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 node_1_0 + 0 node_4_0 + 0 node_7_0
Subject To
 R0: 10000 node_2_0 + contract_0_0 <= 10000.9999
 R1: 10000 node_2_0 + contract_0_0 >= 1
 R2: 10000 node_3_0 - contract_1_0 <= 9997
 R3: 10000 node_3_0 - contract_1_0 >= -2.9999
 R4: - contract_0_0 + 10000 node_5_0 <= 9995
 R5: - contract_0_0 + 10000 node_5_0 >= -4.9999
 R6: contract_0_0 + 10000 node_6_0 <= 10009
 R7: contract_0_0 + 10000 node_6_0 >= 9.0001
 R8: contract_1_0 + 10000 node_8_0 <= 10001.9999
 R9: contract_1_0 + 10000 node_8_0 >= 2
 R10: 10000 node_9_0 - contract_2_0 <= 9997.999900000001
 R11: 10000 node_9_0 - contract_2_0 >= -2
 R12: node_0_0 = 1
Bounds
 -10000 <= contract_0_0 <= 10000
 -10000 <= contract_1_0 <= 10000
 -10000 <= contract_2_0 <= 10000
Binaries
 node_0_0 node_1_0 node_2_0 node_3_0 node_4_0 node_5_0 node_6_0 node_7_0
 node_8_0 node_9_0
General Constraints
 GC0: node_1_0 = OR ( node_2_0 , node_3_0 )
 GC1: node_4_0 = AND ( node_5_0 , node_6_0 )
 GC2: node_7_0 = OR ( node_8_0 , node_9_0 )
 GC3: node_0_0 = AND ( node_1_0 , node_4_0 , node_7_0 )
End
