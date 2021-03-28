\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 node_3_0 + 0 node_3_1 + 0 node_3_2 + 0 node_3_3 + 0 node_1_0
   + 0 node_1_1 + 0 node_1_2 + 0 node_1_3 + 0 node_1_4 + 0 node_1_5
   + 0 node_1_6 + 0 x_0 + 0 x_1 + 0 x_2 + 0 x_3 + 0 x_4 + 0 x_5 + 0 x_6
   + 0 node_2_0 + 0 node_2_1 + 0 node_2_2 + 0 node_2_3 + 0 node_2_4
   + 0 node_2_5 + 0 node_2_6 + 0 node_3_aux_0_0 + 0 node_3_aux_0_1
   + 0 node_3_aux_0_2 + 0 node_3_aux_0_3 + 0 node_3_aux_1_1
   + 0 node_3_aux_1_2 + 0 node_3_aux_1_3 + 0 node_3_aux_1_4
   + 0 node_3_aux_2_2 + 0 node_3_aux_2_3 + 0 node_3_aux_2_4
   + 0 node_3_aux_2_5 + 0 node_3_aux_3_3 + 0 node_3_aux_3_4
   + 0 node_3_aux_3_5 + 0 node_3_aux_3_6
Subject To
 R0: node_4_0 = 1
 GC0: node_1_0 = 1 -> - x_0 <= -3
 GC1: node_1_0 = 0 -> - x_0 >= -2.9999
 GC2: node_1_1 = 1 -> - x_1 <= -3
 GC3: node_1_1 = 0 -> - x_1 >= -2.9999
 GC4: node_1_2 = 1 -> - x_2 <= -3
 GC5: node_1_2 = 0 -> - x_2 >= -2.9999
 GC6: node_1_3 = 1 -> - x_3 <= -3
 GC7: node_1_3 = 0 -> - x_3 >= -2.9999
 GC8: node_1_4 = 1 -> - x_4 <= -3
 GC9: node_1_4 = 0 -> - x_4 >= -2.9999
 GC10: node_1_5 = 1 -> - x_5 <= -3
 GC11: node_1_5 = 0 -> - x_5 >= -2.9999
 GC12: node_1_6 = 1 -> - x_6 <= -3
 GC13: node_1_6 = 0 -> - x_6 >= -2.9999
 GC14: node_2_0 = 1 -> - x_0 <= -5
 GC15: node_2_0 = 0 -> - x_0 >= -4.9999
 GC16: node_2_1 = 1 -> - x_1 <= -5
 GC17: node_2_1 = 0 -> - x_1 >= -4.9999
 GC18: node_2_2 = 1 -> - x_2 <= -5
 GC19: node_2_2 = 0 -> - x_2 >= -4.9999
 GC20: node_2_3 = 1 -> - x_3 <= -5
 GC21: node_2_3 = 0 -> - x_3 >= -4.9999
 GC22: node_2_4 = 1 -> - x_4 <= -5
 GC23: node_2_4 = 0 -> - x_4 >= -4.9999
 GC24: node_2_5 = 1 -> - x_5 <= -5
 GC25: node_2_5 = 0 -> - x_5 >= -4.9999
 GC26: node_2_6 = 1 -> - x_6 <= -5
 GC27: node_2_6 = 0 -> - x_6 >= -4.9999
Bounds
 -10000 <= x_0 <= 10000
 -10000 <= x_1 <= 10000
 -10000 <= x_2 <= 10000
 -10000 <= x_3 <= 10000
 -10000 <= x_4 <= 10000
 -10000 <= x_5 <= 10000
 -10000 <= x_6 <= 10000
Binaries
 node_4_0 node_3_0 node_3_1 node_3_2 node_3_3 node_1_0 node_1_1 node_1_2
 node_1_3 node_1_4 node_1_5 node_1_6 node_2_0 node_2_1 node_2_2 node_2_3
 node_2_4 node_2_5 node_2_6 node_3_aux_0_0 node_3_aux_0_1 node_3_aux_0_2
 node_3_aux_0_3 node_3_aux_1_1 node_3_aux_1_2 node_3_aux_1_3 node_3_aux_1_4
 node_3_aux_2_2 node_3_aux_2_3 node_3_aux_2_4 node_3_aux_2_5 node_3_aux_3_3
 node_3_aux_3_4 node_3_aux_3_5 node_3_aux_3_6
General Constraints
 GC28: node_3_aux_0_0 = AND ( node_2_0 )
 GC29: node_3_aux_0_0 = AND ( node_1_0 , node_2_1 )
 GC30: node_3_aux_0_0 = AND ( node_1_0 , node_1_1 , node_2_2 )
 GC31: node_3_aux_0_0 = AND ( node_1_0 , node_1_1 , node_1_2 , node_2_3 )
 GC32: node_3_0 = OR ( node_3_aux_0_0 , node_3_aux_0_1 , node_3_aux_0_2 ,
   node_3_aux_0_3 )
 GC33: node_3_aux_1_1 = AND ( node_2_1 )
 GC34: node_3_aux_1_1 = AND ( node_1_1 , node_2_2 )
 GC35: node_3_aux_1_1 = AND ( node_1_1 , node_1_2 , node_2_3 )
 GC36: node_3_aux_1_1 = AND ( node_1_1 , node_1_2 , node_1_3 , node_2_4 )
 GC37: node_3_1 = OR ( node_3_aux_1_1 , node_3_aux_1_2 , node_3_aux_1_3 ,
   node_3_aux_1_4 )
 GC38: node_3_aux_2_2 = AND ( node_2_2 )
 GC39: node_3_aux_2_2 = AND ( node_1_2 , node_2_3 )
 GC40: node_3_aux_2_2 = AND ( node_1_2 , node_1_3 , node_2_4 )
 GC41: node_3_aux_2_2 = AND ( node_1_2 , node_1_3 , node_1_4 , node_2_5 )
 GC42: node_3_2 = OR ( node_3_aux_2_2 , node_3_aux_2_3 , node_3_aux_2_4 ,
   node_3_aux_2_5 )
 GC43: node_3_aux_3_3 = AND ( node_2_3 )
 GC44: node_3_aux_3_3 = AND ( node_1_3 , node_2_4 )
 GC45: node_3_aux_3_3 = AND ( node_1_3 , node_1_4 , node_2_5 )
 GC46: node_3_aux_3_3 = AND ( node_1_3 , node_1_4 , node_1_5 , node_2_6 )
 GC47: node_3_3 = OR ( node_3_aux_3_3 , node_3_aux_3_4 , node_3_aux_3_5 ,
   node_3_aux_3_6 )
 GC48: node_4_0 = AND ( node_3_0 , node_3_1 , node_3_2 , node_3_3 )
End
