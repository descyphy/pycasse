\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 node_1_0 + 0 node_2_0 + 0 node_3_0 + 0 node_5_0 + 0 node_7_0
   + 0 node_9_0 + 0 node_11_0 + 0 node_13_0
Subject To
 R0: 10000 node_4_0 + contract_0_0 <= 10004.9999
 R1: 10000 node_4_0 + contract_0_0 >= 4.9999
 R2: 10000 node_4_1 + contract_0_1 <= 10004.9999
 R3: 10000 node_4_1 + contract_0_1 >= 4.9999
 R4: 10000 node_4_2 + contract_0_2 <= 10004.9999
 R5: 10000 node_4_2 + contract_0_2 >= 4.9999
 R6: 10000 node_4_3 + contract_0_3 <= 10004.9999
 R7: 10000 node_4_3 + contract_0_3 >= 4.9999
 R8: 10000 node_6_1 - contract_1_1 <= 9997
 R9: 10000 node_6_1 - contract_1_1 >= -3
 R10: 10000 node_6_2 - contract_1_2 <= 9997
 R11: 10000 node_6_2 - contract_1_2 >= -3
 R12: 10000 node_6_3 - contract_1_3 <= 9997
 R13: 10000 node_6_3 - contract_1_3 >= -3
 R14: 10000 node_6_4 - contract_1_4 <= 9997
 R15: 10000 node_6_4 - contract_1_4 >= -3
 R16: contract_0_0 + 10000 node_8_0 <= 10004.9999
 R17: contract_0_0 + 10000 node_8_0 >= 4.9999
 R18: contract_0_1 + 10000 node_8_1 <= 10004.9999
 R19: contract_0_1 + 10000 node_8_1 >= 4.9999
 R20: contract_0_2 + 10000 node_8_2 <= 10004.9999
 R21: contract_0_2 + 10000 node_8_2 >= 4.9999
 R22: contract_0_3 + 10000 node_8_3 <= 10004.9999
 R23: contract_0_3 + 10000 node_8_3 >= 4.9999
 R24: 10000 node_10_0 - contract_1_0 <= 9995
 R25: 10000 node_10_0 - contract_1_0 >= -5
 R26: - contract_1_1 + 10000 node_10_1 <= 9995
 R27: - contract_1_1 + 10000 node_10_1 >= -5
 R28: - contract_1_2 + 10000 node_10_2 <= 9995
 R29: - contract_1_2 + 10000 node_10_2 >= -5
 R30: - contract_1_3 + 10000 node_10_3 <= 9995
 R31: - contract_1_3 + 10000 node_10_3 >= -5
 R32: - contract_1_4 + 10000 node_10_4 <= 9995
 R33: - contract_1_4 + 10000 node_10_4 >= -5
 R34: - contract_0_0 + 10000 node_12_0 <= 9995
 R35: - contract_0_0 + 10000 node_12_0 >= -5
 R36: - contract_0_1 + 10000 node_12_1 <= 9995
 R37: - contract_0_1 + 10000 node_12_1 >= -5
 R38: - contract_0_2 + 10000 node_12_2 <= 9995
 R39: - contract_0_2 + 10000 node_12_2 >= -5
 R40: - contract_0_3 + 10000 node_12_3 <= 9995
 R41: - contract_0_3 + 10000 node_12_3 >= -5
 R42: contract_1_1 + 10000 node_14_1 <= 10002.9999
 R43: contract_1_1 + 10000 node_14_1 >= 2.9999
 R44: contract_1_2 + 10000 node_14_2 <= 10002.9999
 R45: contract_1_2 + 10000 node_14_2 >= 2.9999
 R46: contract_1_3 + 10000 node_14_3 <= 10002.9999
 R47: contract_1_3 + 10000 node_14_3 >= 2.9999
 R48: contract_1_4 + 10000 node_14_4 <= 10002.9999
 R49: contract_1_4 + 10000 node_14_4 >= 2.9999
 R50: node_0_0 = 1
Bounds
 -10000 <= contract_0_0 <= 10000
 -10000 <= contract_0_1 <= 10000
 -10000 <= contract_0_2 <= 10000
 -10000 <= contract_0_3 <= 10000
 -10000 <= contract_1_1 <= 10000
 -10000 <= contract_1_2 <= 10000
 -10000 <= contract_1_3 <= 10000
 -10000 <= contract_1_4 <= 10000
 -10000 <= contract_1_0 <= 10000
Binaries
 node_0_0 node_1_0 node_2_0 node_3_0 node_4_0 node_4_1 node_4_2 node_4_3
 node_5_0 node_6_1 node_6_2 node_6_3 node_6_4 node_7_0 node_8_0 node_8_1
 node_8_2 node_8_3 node_9_0 node_10_0 node_10_1 node_10_2 node_10_3
 node_10_4 node_11_0 node_12_0 node_12_1 node_12_2 node_12_3 node_13_0
 node_14_1 node_14_2 node_14_3 node_14_4
General Constraints
 GC0: node_3_0 = OR ( node_4_0 , node_4_1 , node_4_2 , node_4_3 )
 GC1: node_5_0 = AND ( node_6_1 , node_6_2 , node_6_3 , node_6_4 )
 GC2: node_7_0 = OR ( node_8_0 , node_8_1 , node_8_2 , node_8_3 )
 GC3: node_2_0 = OR ( node_3_0 , node_5_0 , node_7_0 )
 GC4: node_9_0 = AND ( node_10_0 , node_10_1 , node_10_2 , node_10_3 ,
   node_10_4 )
 GC5: node_1_0 = AND ( node_2_0 , node_9_0 )
 GC6: node_11_0 = AND ( node_12_0 , node_12_1 , node_12_2 , node_12_3 )
 GC7: node_13_0 = OR ( node_14_1 , node_14_2 , node_14_3 , node_14_4 )
 GC8: node_0_0 = AND ( node_1_0 , node_11_0 , node_13_0 )
End
