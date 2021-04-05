\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 node_1_0 + 0 node_2_0 + 0 node_2_1 + 0 node_2_2 + 0 node_2_3
   + 0 node_2_4 + 0 contract_2_0 + 0 contract_2_1 + 0 contract_2_2
   + 0 contract_2_3 + 0 contract_2_4 + 0 node_3_0 + 0 node_4_1 + 0 node_4_2
   + 0 node_4_3 + 0 node_4_4
Subject To
 R0: node_0_0 = 1
 GC0: node_2_0 = 1 -> - contract_2_0 <= -1
 GC1: node_2_0 = 0 -> - contract_2_0 >= -0.9999
 GC2: node_2_1 = 1 -> - contract_2_1 <= -1
 GC3: node_2_1 = 0 -> - contract_2_1 >= -0.9999
 GC4: node_2_2 = 1 -> - contract_2_2 <= -1
 GC5: node_2_2 = 0 -> - contract_2_2 >= -0.9999
 GC6: node_2_3 = 1 -> - contract_2_3 <= -1
 GC7: node_2_3 = 0 -> - contract_2_3 >= -0.9999
 GC8: node_2_4 = 1 -> - contract_2_4 <= -1
 GC9: node_2_4 = 0 -> - contract_2_4 >= -0.9999
 GC11: node_4_1 = 1 -> contract_2_1 <= 2
 GC12: node_4_1 = 0 -> contract_2_1 >= 2.0001
 GC13: node_4_2 = 1 -> contract_2_2 <= 2
 GC14: node_4_2 = 0 -> contract_2_2 >= 2.0001
 GC15: node_4_3 = 1 -> contract_2_3 <= 2
 GC16: node_4_3 = 0 -> contract_2_3 >= 2.0001
 GC17: node_4_4 = 1 -> contract_2_4 <= 2
 GC18: node_4_4 = 0 -> contract_2_4 >= 2.0001
Bounds
 -10000 <= contract_2_0 <= 10000
 -10000 <= contract_2_1 <= 10000
 -10000 <= contract_2_2 <= 10000
 -10000 <= contract_2_3 <= 10000
 -10000 <= contract_2_4 <= 10000
Binaries
 node_0_0 node_1_0 node_2_0 node_2_1 node_2_2 node_2_3 node_2_4 node_3_0
 node_4_1 node_4_2 node_4_3 node_4_4
General Constraints
 GC10: node_1_0 = AND ( node_2_0 , node_2_1 , node_2_2 , node_2_3 ,
   node_2_4 )
 GC19: node_3_0 = OR ( node_4_1 , node_4_2 , node_4_3 , node_4_4 )
 GC20: node_0_0 = AND ( node_1_0 , node_3_0 )
End
