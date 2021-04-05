\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 node_1_0 + 0 node_2_0 + 0 node_2_1 + 0 node_2_2 + 0 node_2_3
   + 0 contract_1_0 + 0 contract_1_1 + 0 contract_1_2 + 0 contract_1_3
   + 0 node_3_0 + 0 node_4_0 + 0 node_5_0 + 0 node_5_1 + 0 node_5_2
   + 0 node_5_3 + 0 node_6_0 + 0 node_7_0 + 0 node_7_1 + 0 node_7_2
   + 0 node_7_3 + 0 node_7_4 + 0 contract_2_0 + 0 contract_2_1
   + 0 contract_2_2 + 0 contract_2_3 + 0 contract_2_4 + 0 node_8_0
   + 0 node_9_0 + 0 node_10_0 + 0 node_11_0 + 0 node_11_1 + 0 node_11_2
   + 0 node_11_3 + 0 node_12_0 + 0 node_13_1 + 0 node_13_2 + 0 node_13_3
   + 0 node_13_4 + 0 node_14_0 + 0 node_15_0 + 0 node_16_0 + 0 node_16_1
   + 0 node_16_2 + 0 node_16_3 + 0 node_17_0 + 0 node_18_0 + 0 node_18_1
   + 0 node_18_2 + 0 node_18_3 + 0 node_18_4 + 0 node_19_0 + 0 node_20_0
   + 0 node_20_1 + 0 node_20_2 + 0 node_20_3 + 0 node_20_4
Subject To
 R0: node_0_0 = 1
 GC0: node_2_0 = 1 -> - contract_1_0 <= -5
 GC1: node_2_0 = 0 -> - contract_1_0 >= -4.9999
 GC2: node_2_1 = 1 -> - contract_1_1 <= -5
 GC3: node_2_1 = 0 -> - contract_1_1 >= -4.9999
 GC4: node_2_2 = 1 -> - contract_1_2 <= -5
 GC5: node_2_2 = 0 -> - contract_1_2 >= -4.9999
 GC6: node_2_3 = 1 -> - contract_1_3 <= -5
 GC7: node_2_3 = 0 -> - contract_1_3 >= -4.9999
 GC9: node_5_0 = 1 -> contract_1_0 <= 5
 GC10: node_5_0 = 0 -> contract_1_0 >= 5.0001
 GC11: node_5_1 = 1 -> contract_1_1 <= 5
 GC12: node_5_1 = 0 -> contract_1_1 >= 5.0001
 GC13: node_5_2 = 1 -> contract_1_2 <= 5
 GC14: node_5_2 = 0 -> contract_1_2 >= 5.0001
 GC15: node_5_3 = 1 -> contract_1_3 <= 5
 GC16: node_5_3 = 0 -> contract_1_3 >= 5.0001
 GC18: node_7_0 = 1 -> - contract_2_0 <= -2
 GC19: node_7_0 = 0 -> - contract_2_0 >= -1.9999
 GC20: node_7_1 = 1 -> - contract_2_1 <= -2
 GC21: node_7_1 = 0 -> - contract_2_1 >= -1.9999
 GC22: node_7_2 = 1 -> - contract_2_2 <= -2
 GC23: node_7_2 = 0 -> - contract_2_2 >= -1.9999
 GC24: node_7_3 = 1 -> - contract_2_3 <= -2
 GC25: node_7_3 = 0 -> - contract_2_3 >= -1.9999
 GC26: node_7_4 = 1 -> - contract_2_4 <= -2
 GC27: node_7_4 = 0 -> - contract_2_4 >= -1.9999
 GC30: node_11_0 = 1 -> - contract_1_0 <= -5
 GC31: node_11_0 = 0 -> - contract_1_0 >= -4.9999
 GC32: node_11_1 = 1 -> - contract_1_1 <= -5
 GC33: node_11_1 = 0 -> - contract_1_1 >= -4.9999
 GC34: node_11_2 = 1 -> - contract_1_2 <= -5
 GC35: node_11_2 = 0 -> - contract_1_2 >= -4.9999
 GC36: node_11_3 = 1 -> - contract_1_3 <= -5
 GC37: node_11_3 = 0 -> - contract_1_3 >= -4.9999
 GC39: node_13_1 = 1 -> contract_2_1 <= 3
 GC40: node_13_1 = 0 -> contract_2_1 >= 3.0001
 GC41: node_13_2 = 1 -> contract_2_2 <= 3
 GC42: node_13_2 = 0 -> contract_2_2 >= 3.0001
 GC43: node_13_3 = 1 -> contract_2_3 <= 3
 GC44: node_13_3 = 0 -> contract_2_3 >= 3.0001
 GC45: node_13_4 = 1 -> contract_2_4 <= 3
 GC46: node_13_4 = 0 -> contract_2_4 >= 3.0001
 GC49: node_16_0 = 1 -> - contract_1_0 <= -5
 GC50: node_16_0 = 0 -> - contract_1_0 >= -4.9999
 GC51: node_16_1 = 1 -> - contract_1_1 <= -5
 GC52: node_16_1 = 0 -> - contract_1_1 >= -4.9999
 GC53: node_16_2 = 1 -> - contract_1_2 <= -5
 GC54: node_16_2 = 0 -> - contract_1_2 >= -4.9999
 GC55: node_16_3 = 1 -> - contract_1_3 <= -5
 GC56: node_16_3 = 0 -> - contract_1_3 >= -4.9999
 GC58: node_18_0 = 1 -> contract_2_0 <= 2
 GC59: node_18_0 = 0 -> contract_2_0 >= 2.0001
 GC60: node_18_1 = 1 -> contract_2_1 <= 2
 GC61: node_18_1 = 0 -> contract_2_1 >= 2.0001
 GC62: node_18_2 = 1 -> contract_2_2 <= 2
 GC63: node_18_2 = 0 -> contract_2_2 >= 2.0001
 GC64: node_18_3 = 1 -> contract_2_3 <= 2
 GC65: node_18_3 = 0 -> contract_2_3 >= 2.0001
 GC66: node_18_4 = 1 -> contract_2_4 <= 2
 GC67: node_18_4 = 0 -> contract_2_4 >= 2.0001
 GC71: node_20_0 = 1 -> contract_2_0 <= 2
 GC72: node_20_0 = 0 -> contract_2_0 >= 2.0001
 GC73: node_20_1 = 1 -> contract_2_1 <= 2
 GC74: node_20_1 = 0 -> contract_2_1 >= 2.0001
 GC75: node_20_2 = 1 -> contract_2_2 <= 2
 GC76: node_20_2 = 0 -> contract_2_2 >= 2.0001
 GC77: node_20_3 = 1 -> contract_2_3 <= 2
 GC78: node_20_3 = 0 -> contract_2_3 >= 2.0001
 GC79: node_20_4 = 1 -> contract_2_4 <= 2
 GC80: node_20_4 = 0 -> contract_2_4 >= 2.0001
Bounds
 -10000 <= contract_1_0 <= 10000
 -10000 <= contract_1_1 <= 10000
 -10000 <= contract_1_2 <= 10000
 -10000 <= contract_1_3 <= 10000
 -10000 <= contract_2_0 <= 10000
 -10000 <= contract_2_1 <= 10000
 -10000 <= contract_2_2 <= 10000
 -10000 <= contract_2_3 <= 10000
 -10000 <= contract_2_4 <= 10000
Binaries
 node_0_0 node_1_0 node_2_0 node_2_1 node_2_2 node_2_3 node_3_0 node_4_0
 node_5_0 node_5_1 node_5_2 node_5_3 node_6_0 node_7_0 node_7_1 node_7_2
 node_7_3 node_7_4 node_8_0 node_9_0 node_10_0 node_11_0 node_11_1
 node_11_2 node_11_3 node_12_0 node_13_1 node_13_2 node_13_3 node_13_4
 node_14_0 node_15_0 node_16_0 node_16_1 node_16_2 node_16_3 node_17_0
 node_18_0 node_18_1 node_18_2 node_18_3 node_18_4 node_19_0 node_20_0
 node_20_1 node_20_2 node_20_3 node_20_4
General Constraints
 GC8: node_1_0 = AND ( node_2_0 , node_2_1 , node_2_2 , node_2_3 )
 GC17: node_4_0 = OR ( node_5_0 , node_5_1 , node_5_2 , node_5_3 )
 GC28: node_6_0 = OR ( node_7_0 , node_7_1 , node_7_2 , node_7_3 , node_7_4 )
 GC29: node_3_0 = OR ( node_4_0 , node_6_0 )
 GC38: node_10_0 = AND ( node_11_0 , node_11_1 , node_11_2 , node_11_3 )
 GC47: node_12_0 = OR ( node_13_1 , node_13_2 , node_13_3 , node_13_4 )
 GC48: node_9_0 = AND ( node_10_0 , node_12_0 )
 GC57: node_15_0 = AND ( node_16_0 , node_16_1 , node_16_2 , node_16_3 )
 GC68: node_17_0 = AND ( node_18_0 , node_18_1 , node_18_2 , node_18_3 ,
   node_18_4 )
 GC69: node_14_0 = AND ( node_15_0 , node_17_0 )
 GC70: node_8_0 = OR ( node_9_0 , node_14_0 )
 GC81: node_19_0 = AND ( node_20_0 , node_20_1 , node_20_2 , node_20_3 ,
   node_20_4 )
 GC82: node_0_0 = AND ( node_1_0 , node_3_0 , node_8_0 , node_19_0 )
End
