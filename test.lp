\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 node_1_0 + 0 node_3_0 + 0 node_5_0
Subject To
 R0: node_0_0 = 1
 R1: 10000 node_6_0 - contract_1_0 <= 9998
 R2: 10000 node_6_0 - contract_1_0 >= -1.9999
 R3: 10000 node_6_1 - contract_1_1 <= 9998
 R4: 10000 node_6_1 - contract_1_1 >= -1.9999
 R5: 10000 node_6_2 - contract_1_2 <= 9998
 R6: 10000 node_6_2 - contract_1_2 >= -1.9999
 R7: 10000 node_6_3 - contract_1_3 <= 9998
 R8: 10000 node_6_3 - contract_1_3 >= -1.9999
 R9: 10000 node_6_4 - contract_1_4 <= 9998
 R10: 10000 node_6_4 - contract_1_4 >= -1.9999
 qc0: 10000 node_2_0 + [ - contract_0_0 ^2 ] <= 9996
 qc1: 10000 node_2_0 + [ - contract_0_0 ^2 ] >= -3.9999
 qc2: 10000 node_2_1 + [ - contract_0_1 ^2 ] <= 9996
 qc3: 10000 node_2_1 + [ - contract_0_1 ^2 ] >= -3.9999
 qc4: 10000 node_2_2 + [ - contract_0_2 ^2 ] <= 9996
 qc5: 10000 node_2_2 + [ - contract_0_2 ^2 ] >= -3.9999
 qc6: 10000 node_2_3 + [ - contract_0_3 ^2 ] <= 9996
 qc7: 10000 node_2_3 + [ - contract_0_3 ^2 ] >= -3.9999
 qc8: 10000 node_4_0 + [ contract_0_0 ^2 ] <= 10009
 qc9: 10000 node_4_0 + [ contract_0_0 ^2 ] >= 9.0001
 qc10: 10000 node_4_1 + [ contract_0_1 ^2 ] <= 10009
 qc11: 10000 node_4_1 + [ contract_0_1 ^2 ] >= 9.0001
 qc12: 10000 node_4_2 + [ contract_0_2 ^2 ] <= 10009
 qc13: 10000 node_4_2 + [ contract_0_2 ^2 ] >= 9.0001
 qc14: 10000 node_4_3 + [ contract_0_3 ^2 ] <= 10009
 qc15: 10000 node_4_3 + [ contract_0_3 ^2 ] >= 9.0001
Bounds
 -10000 <= contract_0_0 <= 10000
 -10000 <= contract_0_1 <= 10000
 -10000 <= contract_0_2 <= 10000
 -10000 <= contract_0_3 <= 10000
 -10000 <= contract_1_0 <= 10000
 -10000 <= contract_1_1 <= 10000
 -10000 <= contract_1_2 <= 10000
 -10000 <= contract_1_3 <= 10000
 -10000 <= contract_1_4 <= 10000
Binaries
 node_0_0 node_1_0 node_2_0 node_2_1 node_2_2 node_2_3 node_3_0 node_4_0
 node_4_1 node_4_2 node_4_3 node_5_0 node_6_0 node_6_1 node_6_2 node_6_3
 node_6_4
General Constraints
 GC0: node_1_0 = AND ( node_2_0 , node_2_1 , node_2_2 , node_2_3 )
 GC1: node_3_0 = AND ( node_4_0 , node_4_1 , node_4_2 , node_4_3 )
 GC2: node_0_0 = AND ( node_1_0 , node_3_0 )
 GC3: node_5_0 = AND ( node_6_0 , node_6_1 , node_6_2 , node_6_3 , node_6_4 )
End
