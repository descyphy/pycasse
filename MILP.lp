\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 node_1_0 + 0 node_1_1 + 0 node_1_2 + 0 node_1_3 + 0 node_1_4
   + 0 node_1_5 + 0 node_1_6 + 0 node_11_0 + 0 node_12_0 + 0 node_12_1
   + 0 node_12_2 + 0 node_12_3 + 0 node_12_4 + 0 node_12_5 + 0 node_12_6
   + 0 node_17_0 + 0 node_18_0 + 0 node_18_1 + 0 node_18_2 + 0 node_18_3
   + 0 node_18_4 + 0 node_18_5 + 0 node_18_6 + 0 node_23_0 + 0 node_24_0
   + 0 node_29_0 + 0 node_30_1 + 0 node_35_0 + 0 node_36_2 + 0 node_41_0
   + 0 node_42_3 + 0 node_47_0 + 0 node_48_4 + 0 node_53_0 + 0 node_54_5
   + 0 node_59_0 + 0 node_60_0 + 0 node_65_0 + 0 node_66_1 + 0 node_71_0
   + 0 node_72_2 + 0 node_77_0 + 0 node_78_3 + 0 node_83_0 + 0 node_84_4
   + 0 node_89_0 + 0 node_90_5 + 0 node_95_0 + 0 node_96_0 + 0 node_101_0
   + 0 node_102_1 + 0 node_107_0 + 0 node_108_2 + 0 node_113_0
   + 0 node_114_3 + 0 node_119_0 + 0 node_120_4 + 0 node_125_0
   + 0 node_126_5 + 0 node_131_0 + 0 node_132_0 + 0 node_137_0
   + 0 node_138_1 + 0 node_143_0 + 0 node_144_2 + 0 node_149_0
   + 0 node_150_3 + 0 node_155_0 + 0 node_156_4 + 0 node_161_0
   + 0 node_162_5 + 10000 goal_x_0 + 10000 goal_y_0 + 10000 goal_x_1
   + 10000 goal_y_1 + 10000 goal_x_2 + 10000 goal_y_2 + 10000 goal_x_3
   + 10000 goal_y_3 + 10000 goal_x_4 + 10000 goal_y_4 + 10000 goal_x_5
   + 10000 goal_y_5 + fuel_x_0 + fuel_y_0 + fuel_x_1 + fuel_y_1 + fuel_x_2
   + fuel_y_2 + fuel_x_3 + fuel_y_3 + fuel_x_4 + fuel_y_4 + fuel_x_5
   + fuel_y_5
Subject To
 R0: 10000 node_2_0 + contract_5_0 <= 10000
 R1: 10000 node_2_0 + contract_5_0 >= 0
 R2: 10000 node_2_1 + contract_5_1 <= 10000
 R3: 10000 node_2_1 + contract_5_1 >= 0
 R4: 10000 node_2_2 + contract_5_2 <= 10000
 R5: 10000 node_2_2 + contract_5_2 >= 0
 R6: 10000 node_2_3 + contract_5_3 <= 10000
 R7: 10000 node_2_3 + contract_5_3 >= 0
 R8: 10000 node_2_4 + contract_5_4 <= 10000
 R9: 10000 node_2_4 + contract_5_4 >= 0
 R10: 10000 node_2_5 + contract_5_5 <= 10000
 R11: 10000 node_2_5 + contract_5_5 >= 0
 R12: 10000 node_2_6 + contract_5_6 <= 10000
 R13: 10000 node_2_6 + contract_5_6 >= 0
 R14: - contract_5_0 + 10000 node_3_0 <= 10000
 R15: - contract_5_0 + 10000 node_3_0 >= 0
 R16: - contract_5_1 + 10000 node_3_1 <= 10000
 R17: - contract_5_1 + 10000 node_3_1 >= 0
 R18: - contract_5_2 + 10000 node_3_2 <= 10000
 R19: - contract_5_2 + 10000 node_3_2 >= 0
 R20: - contract_5_3 + 10000 node_3_3 <= 10000
 R21: - contract_5_3 + 10000 node_3_3 >= 0
 R22: - contract_5_4 + 10000 node_3_4 <= 10000
 R23: - contract_5_4 + 10000 node_3_4 >= 0
 R24: - contract_5_5 + 10000 node_3_5 <= 10000
 R25: - contract_5_5 + 10000 node_3_5 >= 0
 R26: - contract_5_6 + 10000 node_3_6 <= 10000
 R27: - contract_5_6 + 10000 node_3_6 >= 0
 R28: 10000 node_4_0 + contract_6_0 <= 10000
 R29: 10000 node_4_0 + contract_6_0 >= 0
 R30: 10000 node_4_1 + contract_6_1 <= 10000
 R31: 10000 node_4_1 + contract_6_1 >= 0
 R32: 10000 node_4_2 + contract_6_2 <= 10000
 R33: 10000 node_4_2 + contract_6_2 >= 0
 R34: 10000 node_4_3 + contract_6_3 <= 10000
 R35: 10000 node_4_3 + contract_6_3 >= 0
 R36: 10000 node_4_4 + contract_6_4 <= 10000
 R37: 10000 node_4_4 + contract_6_4 >= 0
 R38: 10000 node_4_5 + contract_6_5 <= 10000
 R39: 10000 node_4_5 + contract_6_5 >= 0
 R40: 10000 node_4_6 + contract_6_6 <= 10000
 R41: 10000 node_4_6 + contract_6_6 >= 0
 R42: - contract_6_0 + 10000 node_5_0 <= 10000
 R43: - contract_6_0 + 10000 node_5_0 >= 0
 R44: - contract_6_1 + 10000 node_5_1 <= 10000
 R45: - contract_6_1 + 10000 node_5_1 >= 0
 R46: - contract_6_2 + 10000 node_5_2 <= 10000
 R47: - contract_6_2 + 10000 node_5_2 >= 0
 R48: - contract_6_3 + 10000 node_5_3 <= 10000
 R49: - contract_6_3 + 10000 node_5_3 >= 0
 R50: - contract_6_4 + 10000 node_5_4 <= 10000
 R51: - contract_6_4 + 10000 node_5_4 >= 0
 R52: - contract_6_5 + 10000 node_5_5 <= 10000
 R53: - contract_6_5 + 10000 node_5_5 >= 0
 R54: - contract_6_6 + 10000 node_5_6 <= 10000
 R55: - contract_6_6 + 10000 node_5_6 >= 0
 R56: 10000 node_6_0 + contract_11_0 <= 10000
 R57: 10000 node_6_0 + contract_11_0 >= 0
 R58: 10000 node_6_1 + contract_11_1 <= 10000
 R59: 10000 node_6_1 + contract_11_1 >= 0
 R60: 10000 node_6_2 + contract_11_2 <= 10000
 R61: 10000 node_6_2 + contract_11_2 >= 0
 R62: 10000 node_6_3 + contract_11_3 <= 10000
 R63: 10000 node_6_3 + contract_11_3 >= 0
 R64: 10000 node_6_4 + contract_11_4 <= 10000
 R65: 10000 node_6_4 + contract_11_4 >= 0
 R66: 10000 node_6_5 + contract_11_5 <= 10000
 R67: 10000 node_6_5 + contract_11_5 >= 0
 R68: 10000 node_6_6 + contract_11_6 <= 10000
 R69: 10000 node_6_6 + contract_11_6 >= 0
 R70: - contract_11_0 + 10000 node_7_0 <= 10000
 R71: - contract_11_0 + 10000 node_7_0 >= 0
 R72: - contract_11_1 + 10000 node_7_1 <= 10000
 R73: - contract_11_1 + 10000 node_7_1 >= 0
 R74: - contract_11_2 + 10000 node_7_2 <= 10000
 R75: - contract_11_2 + 10000 node_7_2 >= 0
 R76: - contract_11_3 + 10000 node_7_3 <= 10000
 R77: - contract_11_3 + 10000 node_7_3 >= 0
 R78: - contract_11_4 + 10000 node_7_4 <= 10000
 R79: - contract_11_4 + 10000 node_7_4 >= 0
 R80: - contract_11_5 + 10000 node_7_5 <= 10000
 R81: - contract_11_5 + 10000 node_7_5 >= 0
 R82: - contract_11_6 + 10000 node_7_6 <= 10000
 R83: - contract_11_6 + 10000 node_7_6 >= 0
 R84: 10000 node_8_0 + contract_12_0 <= 10000
 R85: 10000 node_8_0 + contract_12_0 >= 0
 R86: 10000 node_8_1 + contract_12_1 <= 10000
 R87: 10000 node_8_1 + contract_12_1 >= 0
 R88: 10000 node_8_2 + contract_12_2 <= 10000
 R89: 10000 node_8_2 + contract_12_2 >= 0
 R90: 10000 node_8_3 + contract_12_3 <= 10000
 R91: 10000 node_8_3 + contract_12_3 >= 0
 R92: 10000 node_8_4 + contract_12_4 <= 10000
 R93: 10000 node_8_4 + contract_12_4 >= 0
 R94: 10000 node_8_5 + contract_12_5 <= 10000
 R95: 10000 node_8_5 + contract_12_5 >= 0
 R96: 10000 node_8_6 + contract_12_6 <= 10000
 R97: 10000 node_8_6 + contract_12_6 >= 0
 R98: - contract_12_0 + 10000 node_9_0 <= 10000
 R99: - contract_12_0 + 10000 node_9_0 >= 0
 R100: - contract_12_1 + 10000 node_9_1 <= 10000
 R101: - contract_12_1 + 10000 node_9_1 >= 0
 R102: - contract_12_2 + 10000 node_9_2 <= 10000
 R103: - contract_12_2 + 10000 node_9_2 >= 0
 R104: - contract_12_3 + 10000 node_9_3 <= 10000
 R105: - contract_12_3 + 10000 node_9_3 >= 0
 R106: - contract_12_4 + 10000 node_9_4 <= 10000
 R107: - contract_12_4 + 10000 node_9_4 >= 0
 R108: - contract_12_5 + 10000 node_9_5 <= 10000
 R109: - contract_12_5 + 10000 node_9_5 >= 0
 R110: - contract_12_6 + 10000 node_9_6 <= 10000
 R111: - contract_12_6 + 10000 node_9_6 >= 0
 R112: node_0_0 = 1
 R113: 10000 node_13_0 + contract_1_0 - contract_13_0 <= 9995
 R114: 10000 node_13_0 + contract_1_0 - contract_13_0 >= -5
 R115: 10000 node_13_1 + contract_1_1 - contract_13_1 <= 9995
 R116: 10000 node_13_1 + contract_1_1 - contract_13_1 >= -5
 R117: 10000 node_13_2 + contract_1_2 - contract_13_2 <= 9995
 R118: 10000 node_13_2 + contract_1_2 - contract_13_2 >= -5
 R119: 10000 node_13_3 + contract_1_3 - contract_13_3 <= 9995
 R120: 10000 node_13_3 + contract_1_3 - contract_13_3 >= -5
 R121: 10000 node_13_4 + contract_1_4 - contract_13_4 <= 9995
 R122: 10000 node_13_4 + contract_1_4 - contract_13_4 >= -5
 R123: 10000 node_13_5 + contract_1_5 - contract_13_5 <= 9995
 R124: 10000 node_13_5 + contract_1_5 - contract_13_5 >= -5
 R125: 10000 node_13_6 + contract_1_6 - contract_13_6 <= 9995
 R126: 10000 node_13_6 + contract_1_6 - contract_13_6 >= -5
 R127: - contract_1_0 + contract_13_0 + 10000 node_14_0 <= 9995
 R128: - contract_1_0 + contract_13_0 + 10000 node_14_0 >= -5
 R129: - contract_1_1 + contract_13_1 + 10000 node_14_1 <= 9995
 R130: - contract_1_1 + contract_13_1 + 10000 node_14_1 >= -5
 R131: - contract_1_2 + contract_13_2 + 10000 node_14_2 <= 9995
 R132: - contract_1_2 + contract_13_2 + 10000 node_14_2 >= -5
 R133: - contract_1_3 + contract_13_3 + 10000 node_14_3 <= 9995
 R134: - contract_1_3 + contract_13_3 + 10000 node_14_3 >= -5
 R135: - contract_1_4 + contract_13_4 + 10000 node_14_4 <= 9995
 R136: - contract_1_4 + contract_13_4 + 10000 node_14_4 >= -5
 R137: - contract_1_5 + contract_13_5 + 10000 node_14_5 <= 9995
 R138: - contract_1_5 + contract_13_5 + 10000 node_14_5 >= -5
 R139: - contract_1_6 + contract_13_6 + 10000 node_14_6 <= 9995
 R140: - contract_1_6 + contract_13_6 + 10000 node_14_6 >= -5
 R141: 10000 node_15_0 + contract_2_0 - contract_14_0 <= 9998
 R142: 10000 node_15_0 + contract_2_0 - contract_14_0 >= -2
 R143: 10000 node_15_1 + contract_2_1 - contract_14_1 <= 9998
 R144: 10000 node_15_1 + contract_2_1 - contract_14_1 >= -2
 R145: 10000 node_15_2 + contract_2_2 - contract_14_2 <= 9998
 R146: 10000 node_15_2 + contract_2_2 - contract_14_2 >= -2
 R147: 10000 node_15_3 + contract_2_3 - contract_14_3 <= 9998
 R148: 10000 node_15_3 + contract_2_3 - contract_14_3 >= -2
 R149: 10000 node_15_4 + contract_2_4 - contract_14_4 <= 9998
 R150: 10000 node_15_4 + contract_2_4 - contract_14_4 >= -2
 R151: 10000 node_15_5 + contract_2_5 - contract_14_5 <= 9998
 R152: 10000 node_15_5 + contract_2_5 - contract_14_5 >= -2
 R153: 10000 node_15_6 + contract_2_6 - contract_14_6 <= 9998
 R154: 10000 node_15_6 + contract_2_6 - contract_14_6 >= -2
 R155: - contract_2_0 + contract_14_0 + 10000 node_16_0 <= 9998
 R156: - contract_2_0 + contract_14_0 + 10000 node_16_0 >= -2
 R157: - contract_2_1 + contract_14_1 + 10000 node_16_1 <= 9998
 R158: - contract_2_1 + contract_14_1 + 10000 node_16_1 >= -2
 R159: - contract_2_2 + contract_14_2 + 10000 node_16_2 <= 9998
 R160: - contract_2_2 + contract_14_2 + 10000 node_16_2 >= -2
 R161: - contract_2_3 + contract_14_3 + 10000 node_16_3 <= 9998
 R162: - contract_2_3 + contract_14_3 + 10000 node_16_3 >= -2
 R163: - contract_2_4 + contract_14_4 + 10000 node_16_4 <= 9998
 R164: - contract_2_4 + contract_14_4 + 10000 node_16_4 >= -2
 R165: - contract_2_5 + contract_14_5 + 10000 node_16_5 <= 9998
 R166: - contract_2_5 + contract_14_5 + 10000 node_16_5 >= -2
 R167: - contract_2_6 + contract_14_6 + 10000 node_16_6 <= 9998
 R168: - contract_2_6 + contract_14_6 + 10000 node_16_6 >= -2
 R169: - contract_13_0 + 10000 node_19_0 + contract_7_0 <= 9995
 R170: - contract_13_0 + 10000 node_19_0 + contract_7_0 >= -5
 R171: - contract_13_1 + 10000 node_19_1 + contract_7_1 <= 9995
 R172: - contract_13_1 + 10000 node_19_1 + contract_7_1 >= -5
 R173: - contract_13_2 + 10000 node_19_2 + contract_7_2 <= 9995
 R174: - contract_13_2 + 10000 node_19_2 + contract_7_2 >= -5
 R175: - contract_13_3 + 10000 node_19_3 + contract_7_3 <= 9995
 R176: - contract_13_3 + 10000 node_19_3 + contract_7_3 >= -5
 R177: - contract_13_4 + 10000 node_19_4 + contract_7_4 <= 9995
 R178: - contract_13_4 + 10000 node_19_4 + contract_7_4 >= -5
 R179: - contract_13_5 + 10000 node_19_5 + contract_7_5 <= 9995
 R180: - contract_13_5 + 10000 node_19_5 + contract_7_5 >= -5
 R181: - contract_13_6 + 10000 node_19_6 + contract_7_6 <= 9995
 R182: - contract_13_6 + 10000 node_19_6 + contract_7_6 >= -5
 R183: contract_13_0 - contract_7_0 + 10000 node_20_0 <= 9995
 R184: contract_13_0 - contract_7_0 + 10000 node_20_0 >= -5
 R185: contract_13_1 - contract_7_1 + 10000 node_20_1 <= 9995
 R186: contract_13_1 - contract_7_1 + 10000 node_20_1 >= -5
 R187: contract_13_2 - contract_7_2 + 10000 node_20_2 <= 9995
 R188: contract_13_2 - contract_7_2 + 10000 node_20_2 >= -5
 R189: contract_13_3 - contract_7_3 + 10000 node_20_3 <= 9995
 R190: contract_13_3 - contract_7_3 + 10000 node_20_3 >= -5
 R191: contract_13_4 - contract_7_4 + 10000 node_20_4 <= 9995
 R192: contract_13_4 - contract_7_4 + 10000 node_20_4 >= -5
 R193: contract_13_5 - contract_7_5 + 10000 node_20_5 <= 9995
 R194: contract_13_5 - contract_7_5 + 10000 node_20_5 >= -5
 R195: contract_13_6 - contract_7_6 + 10000 node_20_6 <= 9995
 R196: contract_13_6 - contract_7_6 + 10000 node_20_6 >= -5
 R197: - contract_14_0 + 10000 node_21_0 + contract_8_0 <= 9998
 R198: - contract_14_0 + 10000 node_21_0 + contract_8_0 >= -2
 R199: - contract_14_1 + 10000 node_21_1 + contract_8_1 <= 9998
 R200: - contract_14_1 + 10000 node_21_1 + contract_8_1 >= -2
 R201: - contract_14_2 + 10000 node_21_2 + contract_8_2 <= 9998
 R202: - contract_14_2 + 10000 node_21_2 + contract_8_2 >= -2
 R203: - contract_14_3 + 10000 node_21_3 + contract_8_3 <= 9998
 R204: - contract_14_3 + 10000 node_21_3 + contract_8_3 >= -2
 R205: - contract_14_4 + 10000 node_21_4 + contract_8_4 <= 9998
 R206: - contract_14_4 + 10000 node_21_4 + contract_8_4 >= -2
 R207: - contract_14_5 + 10000 node_21_5 + contract_8_5 <= 9998
 R208: - contract_14_5 + 10000 node_21_5 + contract_8_5 >= -2
 R209: - contract_14_6 + 10000 node_21_6 + contract_8_6 <= 9998
 R210: - contract_14_6 + 10000 node_21_6 + contract_8_6 >= -2
 R211: contract_14_0 - contract_8_0 + 10000 node_22_0 <= 9998
 R212: contract_14_0 - contract_8_0 + 10000 node_22_0 >= -2
 R213: contract_14_1 - contract_8_1 + 10000 node_22_1 <= 9998
 R214: contract_14_1 - contract_8_1 + 10000 node_22_1 >= -2
 R215: contract_14_2 - contract_8_2 + 10000 node_22_2 <= 9998
 R216: contract_14_2 - contract_8_2 + 10000 node_22_2 >= -2
 R217: contract_14_3 - contract_8_3 + 10000 node_22_3 <= 9998
 R218: contract_14_3 - contract_8_3 + 10000 node_22_3 >= -2
 R219: contract_14_4 - contract_8_4 + 10000 node_22_4 <= 9998
 R220: contract_14_4 - contract_8_4 + 10000 node_22_4 >= -2
 R221: contract_14_5 - contract_8_5 + 10000 node_22_5 <= 9998
 R222: contract_14_5 - contract_8_5 + 10000 node_22_5 >= -2
 R223: contract_14_6 - contract_8_6 + 10000 node_22_6 <= 9998
 R224: contract_14_6 - contract_8_6 + 10000 node_22_6 >= -2
 R225: node_10_0 = 1
 R226: contract_14_0 + 10000 node_25_0 <= 10013
 R227: contract_14_0 + 10000 node_25_0 >= 13
 R228: - contract_14_0 + 10000 node_26_0 <= 9990
 R229: - contract_14_0 + 10000 node_26_0 >= -10
 R230: - contract_13_0 + 10000 node_27_0 <= 10000
 R231: - contract_13_0 + 10000 node_27_0 >= 0
 R232: contract_13_0 + 10000 node_28_0 <= 20000
 R233: contract_13_0 + 10000 node_28_0 >= 10000
 R234: contract_14_1 + 10000 node_31_1 <= 10013
 R235: contract_14_1 + 10000 node_31_1 >= 13
 R236: - contract_14_1 + 10000 node_32_1 <= 9990
 R237: - contract_14_1 + 10000 node_32_1 >= -10
 R238: - contract_13_1 + 10000 node_33_1 <= 10000
 R239: - contract_13_1 + 10000 node_33_1 >= 0
 R240: contract_13_1 + 10000 node_34_1 <= 20000
 R241: contract_13_1 + 10000 node_34_1 >= 10000
 R242: contract_14_2 + 10000 node_37_2 <= 10013
 R243: contract_14_2 + 10000 node_37_2 >= 13
 R244: - contract_14_2 + 10000 node_38_2 <= 9990
 R245: - contract_14_2 + 10000 node_38_2 >= -10
 R246: - contract_13_2 + 10000 node_39_2 <= 10000
 R247: - contract_13_2 + 10000 node_39_2 >= 0
 R248: contract_13_2 + 10000 node_40_2 <= 20000
 R249: contract_13_2 + 10000 node_40_2 >= 10000
 R250: contract_14_3 + 10000 node_43_3 <= 10013
 R251: contract_14_3 + 10000 node_43_3 >= 13
 R252: - contract_14_3 + 10000 node_44_3 <= 9990
 R253: - contract_14_3 + 10000 node_44_3 >= -10
 R254: - contract_13_3 + 10000 node_45_3 <= 10000
 R255: - contract_13_3 + 10000 node_45_3 >= 0
 R256: contract_13_3 + 10000 node_46_3 <= 20000
 R257: contract_13_3 + 10000 node_46_3 >= 10000
 R258: contract_14_4 + 10000 node_49_4 <= 10013
 R259: contract_14_4 + 10000 node_49_4 >= 13
 R260: - contract_14_4 + 10000 node_50_4 <= 9990
 R261: - contract_14_4 + 10000 node_50_4 >= -10
 R262: - contract_13_4 + 10000 node_51_4 <= 10000
 R263: - contract_13_4 + 10000 node_51_4 >= 0
 R264: contract_13_4 + 10000 node_52_4 <= 20000
 R265: contract_13_4 + 10000 node_52_4 >= 10000
 R266: contract_14_5 + 10000 node_55_5 <= 10013
 R267: contract_14_5 + 10000 node_55_5 >= 13
 R268: - contract_14_5 + 10000 node_56_5 <= 9990
 R269: - contract_14_5 + 10000 node_56_5 >= -10
 R270: - contract_13_5 + 10000 node_57_5 <= 10000
 R271: - contract_13_5 + 10000 node_57_5 >= 0
 R272: contract_13_5 + 10000 node_58_5 <= 20000
 R273: contract_13_5 + 10000 node_58_5 >= 10000
 R274: contract_14_0 + 10000 node_61_0 <= 10010
 R275: contract_14_0 + 10000 node_61_0 >= 10
 R276: - contract_14_0 + 10000 node_62_0 <= 9994
 R277: - contract_14_0 + 10000 node_62_0 >= -6
 R278: - contract_13_0 + 10000 node_63_0 <= 10000
 R279: - contract_13_0 + 10000 node_63_0 >= 0
 R280: contract_13_0 + 10000 node_64_0 <= 20000
 R281: contract_13_0 + 10000 node_64_0 >= 10000
 R282: contract_14_1 + 10000 node_67_1 <= 10010
 R283: contract_14_1 + 10000 node_67_1 >= 10
 R284: - contract_14_1 + 10000 node_68_1 <= 9994
 R285: - contract_14_1 + 10000 node_68_1 >= -6
 R286: - contract_13_1 + 10000 node_69_1 <= 10000
 R287: - contract_13_1 + 10000 node_69_1 >= 0
 R288: contract_13_1 + 10000 node_70_1 <= 20000
 R289: contract_13_1 + 10000 node_70_1 >= 10000
 R290: contract_14_2 + 10000 node_73_2 <= 10010
 R291: contract_14_2 + 10000 node_73_2 >= 10
 R292: - contract_14_2 + 10000 node_74_2 <= 9994
 R293: - contract_14_2 + 10000 node_74_2 >= -6
 R294: - contract_13_2 + 10000 node_75_2 <= 10000
 R295: - contract_13_2 + 10000 node_75_2 >= 0
 R296: contract_13_2 + 10000 node_76_2 <= 20000
 R297: contract_13_2 + 10000 node_76_2 >= 10000
 R298: contract_14_3 + 10000 node_79_3 <= 10010
 R299: contract_14_3 + 10000 node_79_3 >= 10
 R300: - contract_14_3 + 10000 node_80_3 <= 9994
 R301: - contract_14_3 + 10000 node_80_3 >= -6
 R302: - contract_13_3 + 10000 node_81_3 <= 10000
 R303: - contract_13_3 + 10000 node_81_3 >= 0
 R304: contract_13_3 + 10000 node_82_3 <= 20000
 R305: contract_13_3 + 10000 node_82_3 >= 10000
 R306: contract_14_4 + 10000 node_85_4 <= 10010
 R307: contract_14_4 + 10000 node_85_4 >= 10
 R308: - contract_14_4 + 10000 node_86_4 <= 9994
 R309: - contract_14_4 + 10000 node_86_4 >= -6
 R310: - contract_13_4 + 10000 node_87_4 <= 10000
 R311: - contract_13_4 + 10000 node_87_4 >= 0
 R312: contract_13_4 + 10000 node_88_4 <= 20000
 R313: contract_13_4 + 10000 node_88_4 >= 10000
 R314: contract_14_5 + 10000 node_91_5 <= 10010
 R315: contract_14_5 + 10000 node_91_5 >= 10
 R316: - contract_14_5 + 10000 node_92_5 <= 9994
 R317: - contract_14_5 + 10000 node_92_5 >= -6
 R318: - contract_13_5 + 10000 node_93_5 <= 10000
 R319: - contract_13_5 + 10000 node_93_5 >= 0
 R320: contract_13_5 + 10000 node_94_5 <= 20000
 R321: contract_13_5 + 10000 node_94_5 >= 10000
 R322: contract_14_0 + 10000 node_97_0 <= 10006
 R323: contract_14_0 + 10000 node_97_0 >= 6
 R324: - contract_14_0 + 10000 node_98_0 <= 9998
 R325: - contract_14_0 + 10000 node_98_0 >= -2
 R326: - contract_13_0 + 10000 node_99_0 <= 10000
 R327: - contract_13_0 + 10000 node_99_0 >= 0
 R328: contract_13_0 + 10000 node_100_0 <= 20000
 R329: contract_13_0 + 10000 node_100_0 >= 10000
 R330: contract_14_1 + 10000 node_103_1 <= 10006
 R331: contract_14_1 + 10000 node_103_1 >= 6
 R332: - contract_14_1 + 10000 node_104_1 <= 9998
 R333: - contract_14_1 + 10000 node_104_1 >= -2
 R334: - contract_13_1 + 10000 node_105_1 <= 10000
 R335: - contract_13_1 + 10000 node_105_1 >= 0
 R336: contract_13_1 + 10000 node_106_1 <= 20000
 R337: contract_13_1 + 10000 node_106_1 >= 10000
 R338: contract_14_2 + 10000 node_109_2 <= 10006
 R339: contract_14_2 + 10000 node_109_2 >= 6
 R340: - contract_14_2 + 10000 node_110_2 <= 9998
 R341: - contract_14_2 + 10000 node_110_2 >= -2
 R342: - contract_13_2 + 10000 node_111_2 <= 10000
 R343: - contract_13_2 + 10000 node_111_2 >= 0
 R344: contract_13_2 + 10000 node_112_2 <= 20000
 R345: contract_13_2 + 10000 node_112_2 >= 10000
 R346: contract_14_3 + 10000 node_115_3 <= 10006
 R347: contract_14_3 + 10000 node_115_3 >= 6
 R348: - contract_14_3 + 10000 node_116_3 <= 9998
 R349: - contract_14_3 + 10000 node_116_3 >= -2
 R350: - contract_13_3 + 10000 node_117_3 <= 10000
 R351: - contract_13_3 + 10000 node_117_3 >= 0
 R352: contract_13_3 + 10000 node_118_3 <= 20000
 R353: contract_13_3 + 10000 node_118_3 >= 10000
 R354: contract_14_4 + 10000 node_121_4 <= 10006
 R355: contract_14_4 + 10000 node_121_4 >= 6
 R356: - contract_14_4 + 10000 node_122_4 <= 9998
 R357: - contract_14_4 + 10000 node_122_4 >= -2
 R358: - contract_13_4 + 10000 node_123_4 <= 10000
 R359: - contract_13_4 + 10000 node_123_4 >= 0
 R360: contract_13_4 + 10000 node_124_4 <= 20000
 R361: contract_13_4 + 10000 node_124_4 >= 10000
 R362: contract_14_5 + 10000 node_127_5 <= 10006
 R363: contract_14_5 + 10000 node_127_5 >= 6
 R364: - contract_14_5 + 10000 node_128_5 <= 9998
 R365: - contract_14_5 + 10000 node_128_5 >= -2
 R366: - contract_13_5 + 10000 node_129_5 <= 10000
 R367: - contract_13_5 + 10000 node_129_5 >= 0
 R368: contract_13_5 + 10000 node_130_5 <= 20000
 R369: contract_13_5 + 10000 node_130_5 >= 10000
 R370: contract_14_0 + 10000 node_133_0 <= 10002
 R371: contract_14_0 + 10000 node_133_0 >= 2
 R372: - contract_14_0 + 10000 node_134_0 <= 10001
 R373: - contract_14_0 + 10000 node_134_0 >= 1
 R374: - contract_13_0 + 10000 node_135_0 <= 10000
 R375: - contract_13_0 + 10000 node_135_0 >= 0
 R376: contract_13_0 + 10000 node_136_0 <= 20000
 R377: contract_13_0 + 10000 node_136_0 >= 10000
 R378: contract_14_1 + 10000 node_139_1 <= 10002
 R379: contract_14_1 + 10000 node_139_1 >= 2
 R380: - contract_14_1 + 10000 node_140_1 <= 10001
 R381: - contract_14_1 + 10000 node_140_1 >= 1
 R382: - contract_13_1 + 10000 node_141_1 <= 10000
 R383: - contract_13_1 + 10000 node_141_1 >= 0
 R384: contract_13_1 + 10000 node_142_1 <= 20000
 R385: contract_13_1 + 10000 node_142_1 >= 10000
 R386: contract_14_2 + 10000 node_145_2 <= 10002
 R387: contract_14_2 + 10000 node_145_2 >= 2
 R388: - contract_14_2 + 10000 node_146_2 <= 10001
 R389: - contract_14_2 + 10000 node_146_2 >= 1
 R390: - contract_13_2 + 10000 node_147_2 <= 10000
 R391: - contract_13_2 + 10000 node_147_2 >= 0
 R392: contract_13_2 + 10000 node_148_2 <= 20000
 R393: contract_13_2 + 10000 node_148_2 >= 10000
 R394: contract_14_3 + 10000 node_151_3 <= 10002
 R395: contract_14_3 + 10000 node_151_3 >= 2
 R396: - contract_14_3 + 10000 node_152_3 <= 10001
 R397: - contract_14_3 + 10000 node_152_3 >= 1
 R398: - contract_13_3 + 10000 node_153_3 <= 10000
 R399: - contract_13_3 + 10000 node_153_3 >= 0
 R400: contract_13_3 + 10000 node_154_3 <= 20000
 R401: contract_13_3 + 10000 node_154_3 >= 10000
 R402: contract_14_4 + 10000 node_157_4 <= 10002
 R403: contract_14_4 + 10000 node_157_4 >= 2
 R404: - contract_14_4 + 10000 node_158_4 <= 10001
 R405: - contract_14_4 + 10000 node_158_4 >= 1
 R406: - contract_13_4 + 10000 node_159_4 <= 10000
 R407: - contract_13_4 + 10000 node_159_4 >= 0
 R408: contract_13_4 + 10000 node_160_4 <= 20000
 R409: contract_13_4 + 10000 node_160_4 >= 10000
 R410: contract_14_5 + 10000 node_163_5 <= 10002
 R411: contract_14_5 + 10000 node_163_5 >= 2
 R412: - contract_14_5 + 10000 node_164_5 <= 10001
 R413: - contract_14_5 + 10000 node_164_5 >= 1
 R414: - contract_13_5 + 10000 node_165_5 <= 10000
 R415: - contract_13_5 + 10000 node_165_5 >= 0
 R416: contract_13_5 + 10000 node_166_5 <= 20000
 R417: contract_13_5 + 10000 node_166_5 >= 10000
 R418: - contract_1_0 + contract_1_1 - 0.5 contract_3_0 = 0
 R419: - contract_1_1 + contract_1_2 - 0.5 contract_3_1 = 0
 R420: - contract_1_2 + contract_1_3 - 0.5 contract_3_2 = 0
 R421: - contract_1_3 + contract_1_4 - 0.5 contract_3_3 = 0
 R422: - contract_1_4 + contract_1_5 - 0.5 contract_3_4 = 0
 R423: - contract_1_5 + contract_1_6 - 0.5 contract_3_5 = 0
 R424: - contract_2_0 + contract_2_1 - 0.5 contract_4_0 = 0
 R425: - contract_2_1 + contract_2_2 - 0.5 contract_4_1 = 0
 R426: - contract_2_2 + contract_2_3 - 0.5 contract_4_2 = 0
 R427: - contract_2_3 + contract_2_4 - 0.5 contract_4_3 = 0
 R428: - contract_2_4 + contract_2_5 - 0.5 contract_4_4 = 0
 R429: - contract_2_5 + contract_2_6 - 0.5 contract_4_5 = 0
 R430: - 0.5 contract_5_0 - contract_3_0 + contract_3_1 = 0
 R431: - 0.5 contract_5_1 - contract_3_1 + contract_3_2 = 0
 R432: - 0.5 contract_5_2 - contract_3_2 + contract_3_3 = 0
 R433: - 0.5 contract_5_3 - contract_3_3 + contract_3_4 = 0
 R434: - 0.5 contract_5_4 - contract_3_4 + contract_3_5 = 0
 R435: - 0.5 contract_5_5 - contract_3_5 + contract_3_6 = 0
 R436: - 0.5 contract_6_0 - contract_4_0 + contract_4_1 = 0
 R437: - 0.5 contract_6_1 - contract_4_1 + contract_4_2 = 0
 R438: - 0.5 contract_6_2 - contract_4_2 + contract_4_3 = 0
 R439: - 0.5 contract_6_3 - contract_4_3 + contract_4_4 = 0
 R440: - 0.5 contract_6_4 - contract_4_4 + contract_4_5 = 0
 R441: - 0.5 contract_6_5 - contract_4_5 + contract_4_6 = 0
 R442: - contract_7_0 + contract_7_1 - 0.5 contract_9_0 = 0
 R443: - contract_7_1 + contract_7_2 - 0.5 contract_9_1 = 0
 R444: - contract_7_2 + contract_7_3 - 0.5 contract_9_2 = 0
 R445: - contract_7_3 + contract_7_4 - 0.5 contract_9_3 = 0
 R446: - contract_7_4 + contract_7_5 - 0.5 contract_9_4 = 0
 R447: - contract_7_5 + contract_7_6 - 0.5 contract_9_5 = 0
 R448: - contract_8_0 + contract_8_1 - 0.5 contract_10_0 = 0
 R449: - contract_8_1 + contract_8_2 - 0.5 contract_10_1 = 0
 R450: - contract_8_2 + contract_8_3 - 0.5 contract_10_2 = 0
 R451: - contract_8_3 + contract_8_4 - 0.5 contract_10_3 = 0
 R452: - contract_8_4 + contract_8_5 - 0.5 contract_10_4 = 0
 R453: - contract_8_5 + contract_8_6 - 0.5 contract_10_5 = 0
 R454: - 0.5 contract_11_0 - contract_9_0 + contract_9_1 = 0
 R455: - 0.5 contract_11_1 - contract_9_1 + contract_9_2 = 0
 R456: - 0.5 contract_11_2 - contract_9_2 + contract_9_3 = 0
 R457: - 0.5 contract_11_3 - contract_9_3 + contract_9_4 = 0
 R458: - 0.5 contract_11_4 - contract_9_4 + contract_9_5 = 0
 R459: - 0.5 contract_11_5 - contract_9_5 + contract_9_6 = 0
 R460: - 0.5 contract_12_0 - contract_10_0 + contract_10_1 = 0
 R461: - 0.5 contract_12_1 - contract_10_1 + contract_10_2 = 0
 R462: - 0.5 contract_12_2 - contract_10_2 + contract_10_3 = 0
 R463: - 0.5 contract_12_3 - contract_10_3 + contract_10_4 = 0
 R464: - 0.5 contract_12_4 - contract_10_4 + contract_10_5 = 0
 R465: - 0.5 contract_12_5 - contract_10_5 + contract_10_6 = 0
 R466: - contract_13_0 + contract_13_1 - 0.5 contract_15_0 = 0
 R467: - contract_13_1 + contract_13_2 - 0.5 contract_15_1 = 0
 R468: - contract_13_2 + contract_13_3 - 0.5 contract_15_2 = 0
 R469: - contract_13_3 + contract_13_4 - 0.5 contract_15_3 = 0
 R470: - contract_13_4 + contract_13_5 - 0.5 contract_15_4 = 0
 R471: - contract_13_5 + contract_13_6 - 0.5 contract_15_5 = 0
 R472: - contract_14_0 + contract_14_1 - 0.5 contract_16_0 = 0
 R473: - contract_14_1 + contract_14_2 - 0.5 contract_16_1 = 0
 R474: - contract_14_2 + contract_14_3 - 0.5 contract_16_2 = 0
 R475: - contract_14_3 + contract_14_4 - 0.5 contract_16_3 = 0
 R476: - contract_14_4 + contract_14_5 - 0.5 contract_16_4 = 0
 R477: - contract_14_5 + contract_14_6 - 0.5 contract_16_5 = 0
 R478: - contract_15_0 + contract_15_1 - 0.5 contract_17_0 = 0
 R479: - contract_15_1 + contract_15_2 - 0.5 contract_17_1 = 0
 R480: - contract_15_2 + contract_15_3 - 0.5 contract_17_2 = 0
 R481: - contract_15_3 + contract_15_4 - 0.5 contract_17_3 = 0
 R482: - contract_15_4 + contract_15_5 - 0.5 contract_17_4 = 0
 R483: - contract_15_5 + contract_15_6 - 0.5 contract_17_5 = 0
 R484: - contract_16_0 + contract_16_1 - 0.5 contract_18_0 = 0
 R485: - contract_16_1 + contract_16_2 - 0.5 contract_18_1 = 0
 R486: - contract_16_2 + contract_16_3 - 0.5 contract_18_2 = 0
 R487: - contract_16_3 + contract_16_4 - 0.5 contract_18_3 = 0
 R488: - contract_16_4 + contract_16_5 - 0.5 contract_18_4 = 0
 R489: - contract_16_5 + contract_16_6 - 0.5 contract_18_5 = 0
 R490: regions_0 = 1
 R491: regions_1 = 1
 R492: regions_2 = 1
 R493: regions_3 = 1
 R494: regions_4 = 1
 R495: regions_5 = 1
 R496: contract_13_0 - goal_x_0 <= 10000
 R497: - contract_13_0 - goal_x_0 <= -10000
 R498: contract_14_0 - goal_y_0 <= 0
 R499: - contract_14_0 - goal_y_0 <= 0
 R500: contract_13_1 - goal_x_1 <= 10000
 R501: - contract_13_1 - goal_x_1 <= -10000
 R502: contract_14_1 - goal_y_1 <= 0
 R503: - contract_14_1 - goal_y_1 <= 0
 R504: contract_13_2 - goal_x_2 <= 10000
 R505: - contract_13_2 - goal_x_2 <= -10000
 R506: contract_14_2 - goal_y_2 <= 0
 R507: - contract_14_2 - goal_y_2 <= 0
 R508: contract_13_3 - goal_x_3 <= 10000
 R509: - contract_13_3 - goal_x_3 <= -10000
 R510: contract_14_3 - goal_y_3 <= 0
 R511: - contract_14_3 - goal_y_3 <= 0
 R512: contract_13_4 - goal_x_4 <= 10000
 R513: - contract_13_4 - goal_x_4 <= -10000
 R514: contract_14_4 - goal_y_4 <= 0
 R515: - contract_14_4 - goal_y_4 <= 0
 R516: contract_13_5 - goal_x_5 <= 10000
 R517: - contract_13_5 - goal_x_5 <= -10000
 R518: contract_14_5 - goal_y_5 <= 0
 R519: - contract_14_5 - goal_y_5 <= 0
 R520: contract_17_0 - fuel_x_0 <= 0
 R521: - contract_17_0 - fuel_x_0 <= 0
 R522: contract_18_0 - fuel_y_0 <= 0
 R523: - contract_18_0 - fuel_y_0 <= 0
 R524: contract_17_1 - fuel_x_1 <= 0
 R525: - contract_17_1 - fuel_x_1 <= 0
 R526: contract_18_1 - fuel_y_1 <= 0
 R527: - contract_18_1 - fuel_y_1 <= 0
 R528: contract_17_2 - fuel_x_2 <= 0
 R529: - contract_17_2 - fuel_x_2 <= 0
 R530: contract_18_2 - fuel_y_2 <= 0
 R531: - contract_18_2 - fuel_y_2 <= 0
 R532: contract_17_3 - fuel_x_3 <= 0
 R533: - contract_17_3 - fuel_x_3 <= 0
 R534: contract_18_3 - fuel_y_3 <= 0
 R535: - contract_18_3 - fuel_y_3 <= 0
 R536: contract_17_4 - fuel_x_4 <= 0
 R537: - contract_17_4 - fuel_x_4 <= 0
 R538: contract_18_4 - fuel_y_4 <= 0
 R539: - contract_18_4 - fuel_y_4 <= 0
 R540: contract_17_5 - fuel_x_5 <= 0
 R541: - contract_17_5 - fuel_x_5 <= 0
 R542: contract_18_5 - fuel_y_5 <= 0
 R543: - contract_18_5 - fuel_y_5 <= 0
 state_0_0: contract_1_0 = 85.94840595323788
 state_0_1: contract_2_0 = 0
 state_0_2: contract_3_0 = 15
 state_0_3: contract_4_0 = 0
 state_1_0: contract_7_0 = 94.37582205056178
 state_1_1: contract_8_0 = 1.125
 state_1_2: contract_9_0 = 15
 state_1_3: contract_10_0 = -0.25
 state_2_0: contract_13_0 = 82.73507939716094
 state_2_1: contract_14_0 = -12.8125
 state_2_2: contract_15_0 = 11.67962867329958
 state_2_3: contract_16_0 = -4.375
Bounds
 -5 <= contract_5_0 <= 5
 -5 <= contract_5_1 <= 5
 -5 <= contract_5_2 <= 5
 -5 <= contract_5_3 <= 5
 -5 <= contract_5_4 <= 5
 -5 <= contract_5_5 <= 5
 -5 <= contract_5_6 <= 5
 -5 <= contract_6_0 <= 5
 -5 <= contract_6_1 <= 5
 -5 <= contract_6_2 <= 5
 -5 <= contract_6_3 <= 5
 -5 <= contract_6_4 <= 5
 -5 <= contract_6_5 <= 5
 -5 <= contract_6_6 <= 5
 -5 <= contract_11_0 <= 5
 -5 <= contract_11_1 <= 5
 -5 <= contract_11_2 <= 5
 -5 <= contract_11_3 <= 5
 -5 <= contract_11_4 <= 5
 -5 <= contract_11_5 <= 5
 -5 <= contract_11_6 <= 5
 -5 <= contract_12_0 <= 5
 -5 <= contract_12_1 <= 5
 -5 <= contract_12_2 <= 5
 -5 <= contract_12_3 <= 5
 -5 <= contract_12_4 <= 5
 -5 <= contract_12_5 <= 5
 -5 <= contract_12_6 <= 5
 -10000 <= contract_1_0 <= 10000
 -10000 <= contract_13_0 <= 10000
 -10000 <= contract_1_1 <= 10000
 -10000 <= contract_13_1 <= 10000
 -10000 <= contract_1_2 <= 10000
 -10000 <= contract_13_2 <= 10000
 -10000 <= contract_1_3 <= 10000
 -10000 <= contract_13_3 <= 10000
 -10000 <= contract_1_4 <= 10000
 -10000 <= contract_13_4 <= 10000
 -10000 <= contract_1_5 <= 10000
 -10000 <= contract_13_5 <= 10000
 -10000 <= contract_1_6 <= 10000
 -10000 <= contract_13_6 <= 10000
 -10000 <= contract_2_0 <= 10000
 -10000 <= contract_14_0 <= 10000
 -10000 <= contract_2_1 <= 10000
 -10000 <= contract_14_1 <= 10000
 -10000 <= contract_2_2 <= 10000
 -10000 <= contract_14_2 <= 10000
 -10000 <= contract_2_3 <= 10000
 -10000 <= contract_14_3 <= 10000
 -10000 <= contract_2_4 <= 10000
 -10000 <= contract_14_4 <= 10000
 -10000 <= contract_2_5 <= 10000
 -10000 <= contract_14_5 <= 10000
 -10000 <= contract_2_6 <= 10000
 -10000 <= contract_14_6 <= 10000
 -10000 <= contract_7_0 <= 10000
 -10000 <= contract_7_1 <= 10000
 -10000 <= contract_7_2 <= 10000
 -10000 <= contract_7_3 <= 10000
 -10000 <= contract_7_4 <= 10000
 -10000 <= contract_7_5 <= 10000
 -10000 <= contract_7_6 <= 10000
 -10000 <= contract_8_0 <= 10000
 -10000 <= contract_8_1 <= 10000
 -10000 <= contract_8_2 <= 10000
 -10000 <= contract_8_3 <= 10000
 -10000 <= contract_8_4 <= 10000
 -10000 <= contract_8_5 <= 10000
 -10000 <= contract_8_6 <= 10000
 -15 <= contract_3_0 <= 15
 -15 <= contract_3_1 <= 15
 -15 <= contract_3_2 <= 15
 -15 <= contract_3_3 <= 15
 -15 <= contract_3_4 <= 15
 -15 <= contract_3_5 <= 15
 -15 <= contract_4_0 <= 15
 -15 <= contract_4_1 <= 15
 -15 <= contract_4_2 <= 15
 -15 <= contract_4_3 <= 15
 -15 <= contract_4_4 <= 15
 -15 <= contract_4_5 <= 15
 -15 <= contract_3_6 <= 15
 -15 <= contract_4_6 <= 15
 -15 <= contract_9_0 <= 15
 -15 <= contract_9_1 <= 15
 -15 <= contract_9_2 <= 15
 -15 <= contract_9_3 <= 15
 -15 <= contract_9_4 <= 15
 -15 <= contract_9_5 <= 15
 -15 <= contract_10_0 <= 15
 -15 <= contract_10_1 <= 15
 -15 <= contract_10_2 <= 15
 -15 <= contract_10_3 <= 15
 -15 <= contract_10_4 <= 15
 -15 <= contract_10_5 <= 15
 -15 <= contract_9_6 <= 15
 -15 <= contract_10_6 <= 15
 -15 <= contract_15_0 <= 15
 -15 <= contract_15_1 <= 15
 -15 <= contract_15_2 <= 15
 -15 <= contract_15_3 <= 15
 -15 <= contract_15_4 <= 15
 -15 <= contract_15_5 <= 15
 -15 <= contract_16_0 <= 15
 -15 <= contract_16_1 <= 15
 -15 <= contract_16_2 <= 15
 -15 <= contract_16_3 <= 15
 -15 <= contract_16_4 <= 15
 -15 <= contract_16_5 <= 15
 -5 <= contract_17_0 <= 5
 -5 <= contract_17_1 <= 5
 -5 <= contract_17_2 <= 5
 -5 <= contract_17_3 <= 5
 -5 <= contract_17_4 <= 5
 -15 <= contract_15_6 <= 15
 -5 <= contract_17_5 <= 5
 -5 <= contract_18_0 <= 5
 -5 <= contract_18_1 <= 5
 -5 <= contract_18_2 <= 5
 -5 <= contract_18_3 <= 5
 -5 <= contract_18_4 <= 5
 -15 <= contract_16_6 <= 15
 -5 <= contract_18_5 <= 5
 -10000 <= goal_x_0 <= 10000
 -10000 <= goal_y_0 <= 10000
 -10000 <= goal_x_1 <= 10000
 -10000 <= goal_y_1 <= 10000
 -10000 <= goal_x_2 <= 10000
 -10000 <= goal_y_2 <= 10000
 -10000 <= goal_x_3 <= 10000
 -10000 <= goal_y_3 <= 10000
 -10000 <= goal_x_4 <= 10000
 -10000 <= goal_y_4 <= 10000
 -10000 <= goal_x_5 <= 10000
 -10000 <= goal_y_5 <= 10000
 fuel_x_0 <= 5
 fuel_y_0 <= 5
 fuel_x_1 <= 5
 fuel_y_1 <= 5
 fuel_x_2 <= 5
 fuel_y_2 <= 5
 fuel_x_3 <= 5
 fuel_y_3 <= 5
 fuel_x_4 <= 5
 fuel_y_4 <= 5
 fuel_x_5 <= 5
 fuel_y_5 <= 5
Binaries
 node_0_0 node_1_0 node_1_1 node_1_2 node_1_3 node_1_4 node_1_5 node_1_6
 node_2_0 node_2_1 node_2_2 node_2_3 node_2_4 node_2_5 node_2_6 node_3_0
 node_3_1 node_3_2 node_3_3 node_3_4 node_3_5 node_3_6 node_4_0 node_4_1
 node_4_2 node_4_3 node_4_4 node_4_5 node_4_6 node_5_0 node_5_1 node_5_2
 node_5_3 node_5_4 node_5_5 node_5_6 node_6_0 node_6_1 node_6_2 node_6_3
 node_6_4 node_6_5 node_6_6 node_7_0 node_7_1 node_7_2 node_7_3 node_7_4
 node_7_5 node_7_6 node_8_0 node_8_1 node_8_2 node_8_3 node_8_4 node_8_5
 node_8_6 node_9_0 node_9_1 node_9_2 node_9_3 node_9_4 node_9_5 node_9_6
 node_10_0 node_11_0 node_12_0 node_12_1 node_12_2 node_12_3 node_12_4
 node_12_5 node_12_6 node_13_0 node_13_1 node_13_2 node_13_3 node_13_4
 node_13_5 node_13_6 node_14_0 node_14_1 node_14_2 node_14_3 node_14_4
 node_14_5 node_14_6 node_15_0 node_15_1 node_15_2 node_15_3 node_15_4
 node_15_5 node_15_6 node_16_0 node_16_1 node_16_2 node_16_3 node_16_4
 node_16_5 node_16_6 node_17_0 node_18_0 node_18_1 node_18_2 node_18_3
 node_18_4 node_18_5 node_18_6 node_19_0 node_19_1 node_19_2 node_19_3
 node_19_4 node_19_5 node_19_6 node_20_0 node_20_1 node_20_2 node_20_3
 node_20_4 node_20_5 node_20_6 node_21_0 node_21_1 node_21_2 node_21_3
 node_21_4 node_21_5 node_21_6 node_22_0 node_22_1 node_22_2 node_22_3
 node_22_4 node_22_5 node_22_6 node_23_0 node_24_0 node_25_0 node_26_0
 node_27_0 node_28_0 node_29_0 node_30_1 node_31_1 node_32_1 node_33_1
 node_34_1 node_35_0 node_36_2 node_37_2 node_38_2 node_39_2 node_40_2
 node_41_0 node_42_3 node_43_3 node_44_3 node_45_3 node_46_3 node_47_0
 node_48_4 node_49_4 node_50_4 node_51_4 node_52_4 node_53_0 node_54_5
 node_55_5 node_56_5 node_57_5 node_58_5 node_59_0 node_60_0 node_61_0
 node_62_0 node_63_0 node_64_0 node_65_0 node_66_1 node_67_1 node_68_1
 node_69_1 node_70_1 node_71_0 node_72_2 node_73_2 node_74_2 node_75_2
 node_76_2 node_77_0 node_78_3 node_79_3 node_80_3 node_81_3 node_82_3
 node_83_0 node_84_4 node_85_4 node_86_4 node_87_4 node_88_4 node_89_0
 node_90_5 node_91_5 node_92_5 node_93_5 node_94_5 node_95_0 node_96_0
 node_97_0 node_98_0 node_99_0 node_100_0 node_101_0 node_102_1 node_103_1
 node_104_1 node_105_1 node_106_1 node_107_0 node_108_2 node_109_2
 node_110_2 node_111_2 node_112_2 node_113_0 node_114_3 node_115_3
 node_116_3 node_117_3 node_118_3 node_119_0 node_120_4 node_121_4
 node_122_4 node_123_4 node_124_4 node_125_0 node_126_5 node_127_5
 node_128_5 node_129_5 node_130_5 node_131_0 node_132_0 node_133_0
 node_134_0 node_135_0 node_136_0 node_137_0 node_138_1 node_139_1
 node_140_1 node_141_1 node_142_1 node_143_0 node_144_2 node_145_2
 node_146_2 node_147_2 node_148_2 node_149_0 node_150_3 node_151_3
 node_152_3 node_153_3 node_154_3 node_155_0 node_156_4 node_157_4
 node_158_4 node_159_4 node_160_4 node_161_0 node_162_5 node_163_5
 node_164_5 node_165_5 node_166_5 regions_0 regions_1 regions_2 regions_3
 regions_4 regions_5
General Constraints
 GC0: node_1_0 = AND ( node_2_0 , node_3_0 , node_4_0 , node_5_0 ,
   node_6_0 , node_7_0 , node_8_0 , node_9_0 )
 GC1: node_1_1 = AND ( node_2_1 , node_3_1 , node_4_1 , node_5_1 ,
   node_6_1 , node_7_1 , node_8_1 , node_9_1 )
 GC2: node_1_2 = AND ( node_2_2 , node_3_2 , node_4_2 , node_5_2 ,
   node_6_2 , node_7_2 , node_8_2 , node_9_2 )
 GC3: node_1_3 = AND ( node_2_3 , node_3_3 , node_4_3 , node_5_3 ,
   node_6_3 , node_7_3 , node_8_3 , node_9_3 )
 GC4: node_1_4 = AND ( node_2_4 , node_3_4 , node_4_4 , node_5_4 ,
   node_6_4 , node_7_4 , node_8_4 , node_9_4 )
 GC5: node_1_5 = AND ( node_2_5 , node_3_5 , node_4_5 , node_5_5 ,
   node_6_5 , node_7_5 , node_8_5 , node_9_5 )
 GC6: node_1_6 = AND ( node_2_6 , node_3_6 , node_4_6 , node_5_6 ,
   node_6_6 , node_7_6 , node_8_6 , node_9_6 )
 GC7: node_0_0 = AND ( node_1_0 , node_1_1 , node_1_2 , node_1_3 ,
   node_1_4 , node_1_5 , node_1_6 )
 GC8: node_12_0 = OR ( node_13_0 , node_14_0 , node_15_0 , node_16_0 )
 GC9: node_12_1 = OR ( node_13_1 , node_14_1 , node_15_1 , node_16_1 )
 GC10: node_12_2 = OR ( node_13_2 , node_14_2 , node_15_2 , node_16_2 )
 GC11: node_12_3 = OR ( node_13_3 , node_14_3 , node_15_3 , node_16_3 )
 GC12: node_12_4 = OR ( node_13_4 , node_14_4 , node_15_4 , node_16_4 )
 GC13: node_12_5 = OR ( node_13_5 , node_14_5 , node_15_5 , node_16_5 )
 GC14: node_12_6 = OR ( node_13_6 , node_14_6 , node_15_6 , node_16_6 )
 GC15: node_11_0 = AND ( node_12_0 , node_12_1 , node_12_2 , node_12_3 ,
   node_12_4 , node_12_5 , node_12_6 )
 GC16: node_18_0 = OR ( node_19_0 , node_20_0 , node_21_0 , node_22_0 )
 GC17: node_18_1 = OR ( node_19_1 , node_20_1 , node_21_1 , node_22_1 )
 GC18: node_18_2 = OR ( node_19_2 , node_20_2 , node_21_2 , node_22_2 )
 GC19: node_18_3 = OR ( node_19_3 , node_20_3 , node_21_3 , node_22_3 )
 GC20: node_18_4 = OR ( node_19_4 , node_20_4 , node_21_4 , node_22_4 )
 GC21: node_18_5 = OR ( node_19_5 , node_20_5 , node_21_5 , node_22_5 )
 GC22: node_18_6 = OR ( node_19_6 , node_20_6 , node_21_6 , node_22_6 )
 GC23: node_17_0 = AND ( node_18_0 , node_18_1 , node_18_2 , node_18_3 ,
   node_18_4 , node_18_5 , node_18_6 )
 GC24: node_10_0 = AND ( node_11_0 , node_17_0 )
 GC25: node_24_0 = AND ( node_25_0 , node_26_0 , node_27_0 , node_28_0 )
 GC26: node_23_0 = AND ( node_24_0 )
 GC27: node_30_1 = AND ( node_31_1 , node_32_1 , node_33_1 , node_34_1 )
 GC28: node_29_0 = AND ( node_30_1 )
 GC29: node_36_2 = AND ( node_37_2 , node_38_2 , node_39_2 , node_40_2 )
 GC30: node_35_0 = AND ( node_36_2 )
 GC31: node_42_3 = AND ( node_43_3 , node_44_3 , node_45_3 , node_46_3 )
 GC32: node_41_0 = AND ( node_42_3 )
 GC33: node_48_4 = AND ( node_49_4 , node_50_4 , node_51_4 , node_52_4 )
 GC34: node_47_0 = AND ( node_48_4 )
 GC35: node_54_5 = AND ( node_55_5 , node_56_5 , node_57_5 , node_58_5 )
 GC36: node_53_0 = AND ( node_54_5 )
 GC37: node_60_0 = AND ( node_61_0 , node_62_0 , node_63_0 , node_64_0 )
 GC38: node_59_0 = AND ( node_60_0 )
 GC39: node_66_1 = AND ( node_67_1 , node_68_1 , node_69_1 , node_70_1 )
 GC40: node_65_0 = AND ( node_66_1 )
 GC41: node_72_2 = AND ( node_73_2 , node_74_2 , node_75_2 , node_76_2 )
 GC42: node_71_0 = AND ( node_72_2 )
 GC43: node_78_3 = AND ( node_79_3 , node_80_3 , node_81_3 , node_82_3 )
 GC44: node_77_0 = AND ( node_78_3 )
 GC45: node_84_4 = AND ( node_85_4 , node_86_4 , node_87_4 , node_88_4 )
 GC46: node_83_0 = AND ( node_84_4 )
 GC47: node_90_5 = AND ( node_91_5 , node_92_5 , node_93_5 , node_94_5 )
 GC48: node_89_0 = AND ( node_90_5 )
 GC49: node_96_0 = AND ( node_97_0 , node_98_0 , node_99_0 , node_100_0 )
 GC50: node_95_0 = AND ( node_96_0 )
 GC51: node_102_1 = AND ( node_103_1 , node_104_1 , node_105_1 , node_106_1 )
 GC52: node_101_0 = AND ( node_102_1 )
 GC53: node_108_2 = AND ( node_109_2 , node_110_2 , node_111_2 , node_112_2 )
 GC54: node_107_0 = AND ( node_108_2 )
 GC55: node_114_3 = AND ( node_115_3 , node_116_3 , node_117_3 , node_118_3 )
 GC56: node_113_0 = AND ( node_114_3 )
 GC57: node_120_4 = AND ( node_121_4 , node_122_4 , node_123_4 , node_124_4 )
 GC58: node_119_0 = AND ( node_120_4 )
 GC59: node_126_5 = AND ( node_127_5 , node_128_5 , node_129_5 , node_130_5 )
 GC60: node_125_0 = AND ( node_126_5 )
 GC61: node_132_0 = AND ( node_133_0 , node_134_0 , node_135_0 , node_136_0 )
 GC62: node_131_0 = AND ( node_132_0 )
 GC63: node_138_1 = AND ( node_139_1 , node_140_1 , node_141_1 , node_142_1 )
 GC64: node_137_0 = AND ( node_138_1 )
 GC65: node_144_2 = AND ( node_145_2 , node_146_2 , node_147_2 , node_148_2 )
 GC66: node_143_0 = AND ( node_144_2 )
 GC67: node_150_3 = AND ( node_151_3 , node_152_3 , node_153_3 , node_154_3 )
 GC68: node_149_0 = AND ( node_150_3 )
 GC69: node_156_4 = AND ( node_157_4 , node_158_4 , node_159_4 , node_160_4 )
 GC70: node_155_0 = AND ( node_156_4 )
 GC71: node_162_5 = AND ( node_163_5 , node_164_5 , node_165_5 , node_166_5 )
 GC72: node_161_0 = AND ( node_162_5 )
 GC73: regions_0 = OR ( node_23_0 , node_59_0 , node_95_0 , node_131_0 )
 GC74: regions_1 = OR ( node_29_0 , node_65_0 , node_101_0 , node_137_0 )
 GC75: regions_2 = OR ( node_35_0 , node_71_0 , node_107_0 , node_143_0 )
 GC76: regions_3 = OR ( node_41_0 , node_77_0 , node_113_0 , node_149_0 )
 GC77: regions_4 = OR ( node_47_0 , node_83_0 , node_119_0 , node_155_0 )
 GC78: regions_5 = OR ( node_53_0 , node_89_0 , node_125_0 , node_161_0 )
End
