\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 x_0 + 0 x_1 + 0 x_2 + 0 x_3 + 0 y_1 + 0 y_2 + 0 y_3 + 0 y_4
Subject To
 R0: node_2_0 - node_1_0 - node_1_1 - node_1_2 - node_1_3 <= 0
 R1: 4 node_2_0 - node_1_0 - node_1_1 - node_1_2 - node_1_3 >= 0
 R2: 4 node_4_0 - node_3_1 - node_3_2 - node_3_3 - node_3_4 <= 0
 R3: node_4_0 - node_3_1 - node_3_2 - node_3_3 - node_3_4 >= -3
 R4: node_5_0 - node_2_0 - node_4_0 <= 0
 R5: 2 node_5_0 - node_2_0 - node_4_0 >= 0
 R6: node_5_0 = 1
 GC0: node_1_0 = 1 -> x_0 <= 4.9999
 GC1: node_1_0 = 0 -> x_0 >= 5
 GC2: node_1_1 = 1 -> x_1 <= 4.9999
 GC3: node_1_1 = 0 -> x_1 >= 5
 GC4: node_1_2 = 1 -> x_2 <= 4.9999
 GC5: node_1_2 = 0 -> x_2 >= 5
 GC6: node_1_3 = 1 -> x_3 <= 4.9999
 GC7: node_1_3 = 0 -> x_3 >= 5
 GC8: node_3_1 = 1 -> - y_1 <= -2
 GC9: node_3_1 = 0 -> - y_1 >= -1.9999
 GC10: node_3_2 = 1 -> - y_2 <= -2
 GC11: node_3_2 = 0 -> - y_2 >= -1.9999
 GC12: node_3_3 = 1 -> - y_3 <= -2
 GC13: node_3_3 = 0 -> - y_3 >= -1.9999
 GC14: node_3_4 = 1 -> - y_4 <= -2
 GC15: node_3_4 = 0 -> - y_4 >= -1.9999
Bounds
 -10000 <= x_0 <= 10000
 -10000 <= x_1 <= 10000
 -10000 <= x_2 <= 10000
 -10000 <= x_3 <= 10000
 -10000 <= y_1 <= 10000
 -10000 <= y_2 <= 10000
 -10000 <= y_3 <= 10000
 -10000 <= y_4 <= 10000
Binaries
 node_5_0 node_2_0 node_1_0 node_1_1 node_1_2 node_1_3 node_4_0 node_3_1
 node_3_2 node_3_3 node_3_4
End
