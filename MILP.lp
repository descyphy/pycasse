\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 node_1_0 + 0 node_2_0 + 0 node_6_0
Subject To
 R0: 10000 node_3_0 + contract_1_0 <= 10002.9999
 R1: 10000 node_3_0 + contract_1_0 >= 2.9999
 R2: - contract_1_0 + 10000 node_4_0 <= 9990.999900000001
 R3: - contract_1_0 + 10000 node_4_0 >= -9.0001
 R4: 10000 node_5_0 - contract_0_0 <= 9997
 R5: 10000 node_5_0 - contract_0_0 >= -3
 R6: - contract_1_0 + 10000 node_7_0 <= 9995
 R7: - contract_1_0 + 10000 node_7_0 >= -5
 R8: contract_1_0 + 10000 node_8_0 <= 10009
 R9: contract_1_0 + 10000 node_8_0 >= 9
 R10: contract_0_0 + 10000 node_9_0 <= 10001.9999
 R11: contract_0_0 + 10000 node_9_0 >= 1.9999
 R12: node_0_0 = 1
Bounds
 -10000 <= contract_1_0 <= 10000
 -10000 <= contract_0_0 <= 10000
Binaries
 node_0_0 node_1_0 node_2_0 node_3_0 node_4_0 node_5_0 node_6_0 node_7_0
 node_8_0 node_9_0
General Constraints
 GC0: node_2_0 = OR ( node_3_0 , node_4_0 )
 GC1: node_1_0 = OR ( node_2_0 , node_5_0 )
 GC2: node_6_0 = AND ( node_7_0 , node_8_0 )
 GC3: node_0_0 = AND ( node_1_0 , node_6_0 , node_9_0 )
End
