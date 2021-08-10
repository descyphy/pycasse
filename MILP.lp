\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 node_1_0 + 0 node_2_0 + 0 node_4_0 + 0 node_6_0
Subject To
 R0: 10000 node_3_0 + contract_1_0 <= 10004.9999
 R1: 10000 node_3_0 + contract_1_0 >= 4.9999
 R2: 10000 node_3_1 + contract_1_1 <= 10004.9999
 R3: 10000 node_3_1 + contract_1_1 >= 4.9999
 R4: 10000 node_3_2 + contract_1_2 <= 10004.9999
 R5: 10000 node_3_2 + contract_1_2 >= 4.9999
 R6: 10000 node_3_3 + contract_1_3 <= 10004.9999
 R7: 10000 node_3_3 + contract_1_3 >= 4.9999
 R8: 10000 node_5_1 - contract_2_1 <= 9999
 R9: 10000 node_5_1 - contract_2_1 >= -1
 R10: 10000 node_5_2 - contract_2_2 <= 9999
 R11: 10000 node_5_2 - contract_2_2 >= -1
 R12: 10000 node_5_3 - contract_2_3 <= 9999
 R13: 10000 node_5_3 - contract_2_3 >= -1
 R14: 10000 node_5_4 - contract_2_4 <= 9999
 R15: 10000 node_5_4 - contract_2_4 >= -1
 R16: 10000 node_7_0 + contract_2_0 <= 10002
 R17: 10000 node_7_0 + contract_2_0 >= 2
 R18: contract_2_1 + 10000 node_7_1 <= 10002
 R19: contract_2_1 + 10000 node_7_1 >= 2
 R20: contract_2_2 + 10000 node_7_2 <= 10002
 R21: contract_2_2 + 10000 node_7_2 >= 2
 R22: contract_2_3 + 10000 node_7_3 <= 10002
 R23: contract_2_3 + 10000 node_7_3 >= 2
 R24: contract_2_4 + 10000 node_7_4 <= 10002
 R25: contract_2_4 + 10000 node_7_4 >= 2
 R26: node_0_0 = 1
Bounds
 -10000 <= contract_1_0 <= 10000
 -10000 <= contract_1_1 <= 10000
 -10000 <= contract_1_2 <= 10000
 -10000 <= contract_1_3 <= 10000
 -10000 <= contract_2_1 <= 10000
 -10000 <= contract_2_2 <= 10000
 -10000 <= contract_2_3 <= 10000
 -10000 <= contract_2_4 <= 10000
 -10000 <= contract_2_0 <= 10000
Binaries
 node_0_0 node_1_0 node_2_0 node_3_0 node_3_1 node_3_2 node_3_3 node_4_0
 node_5_1 node_5_2 node_5_3 node_5_4 node_6_0 node_7_0 node_7_1 node_7_2
 node_7_3 node_7_4
General Constraints
 GC0: node_2_0 = OR ( node_3_0 , node_3_1 , node_3_2 , node_3_3 )
 GC1: node_4_0 = OR ( node_5_1 , node_5_2 , node_5_3 , node_5_4 )
 GC2: node_1_0 = OR ( node_2_0 , node_4_0 )
 GC3: node_6_0 = AND ( node_7_0 , node_7_1 , node_7_2 , node_7_3 , node_7_4 )
 GC4: node_0_0 = AND ( node_1_0 , node_6_0 )
End
