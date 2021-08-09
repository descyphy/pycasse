\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  10000 node_0_0 + 0 node_1_0 + 0 node_1_1 + 0 node_1_2 + 0 node_1_3
   + 0 node_1_4 + 0 node_1_5 + 0 node_1_6 + 0 node_1_7 + 0 node_1_8
   + 0 node_1_9 + 0 node_1_10 + 0 node_11_0 + 0 node_12_0 + 0 node_12_1
   + 0 node_12_2 + 0 node_12_3 + 0 node_12_4 + 0 node_12_5 + 0 node_12_6
   + 0 node_12_7 + 0 node_12_8 + 0 node_12_9 + 0 node_12_10 + 0 node_17_0
   + 0 node_18_0 + 0 node_18_1 + 0 node_18_2 + 0 node_18_3 + 0 node_18_4
   + 0 node_18_5 + 0 node_18_6 + 0 node_18_7 + 0 node_18_8 + 0 node_18_9
   + 0 node_18_10 + 0 node_23_0 + 0 node_24_0 + 0 node_29_0 + 0 node_30_1
   + 0 node_35_0 + 0 node_36_2 + 0 node_41_0 + 0 node_42_3 + 0 node_47_0
   + 0 node_48_4 + 0 node_53_0 + 0 node_54_5 + 0 node_59_0 + 0 node_60_6
   + 0 node_65_0 + 0 node_66_7 + 0 node_71_0 + 0 node_72_8 + 0 node_77_0
   + 0 node_78_9 + 0 node_83_0 + 0 node_84_0 + 0 node_89_0 + 0 node_90_1
   + 0 node_95_0 + 0 node_96_2 + 0 node_101_0 + 0 node_102_3 + 0 node_107_0
   + 0 node_108_4 + 0 node_113_0 + 0 node_114_5 + 0 node_119_0
   + 0 node_120_6 + 0 node_125_0 + 0 node_126_7 + 0 node_131_0
   + 0 node_132_8 + 0 node_137_0 + 0 node_138_9 + 0 node_143_0
   + 0 node_144_0 + 0 node_149_0 + 0 node_150_1 + 0 node_155_0
   + 0 node_156_2 + 0 node_161_0 + 0 node_162_3 + 0 node_167_0
   + 0 node_168_4 + 0 node_173_0 + 0 node_174_5 + 0 node_179_0
   + 0 node_180_6 + 0 node_185_0 + 0 node_186_7 + 0 node_191_0
   + 0 node_192_8 + 0 node_197_0 + 0 node_198_9 + goal_x_0 + goal_y_0
   + goal_x_1 + goal_y_1 + goal_x_2 + goal_y_2 + goal_x_3 + goal_y_3
   + goal_x_4 + goal_y_4 + goal_x_5 + goal_y_5 + goal_x_6 + goal_y_6
   + goal_x_7 + goal_y_7 + goal_x_8 + goal_y_8 + goal_x_9 + goal_y_9
   + fuel_x_0 + fuel_y_0 + fuel_x_1 + fuel_y_1 + fuel_x_2 + fuel_y_2
   + fuel_x_3 + fuel_y_3 + fuel_x_4 + fuel_y_4 + fuel_x_5 + fuel_y_5
   + fuel_x_6 + fuel_y_6 + fuel_x_7 + fuel_y_7 + fuel_x_8 + fuel_y_8
   + fuel_x_9 + fuel_y_9
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
 R14: 10000 node_2_7 + contract_5_7 <= 10000
 R15: 10000 node_2_7 + contract_5_7 >= 0
 R16: 10000 node_2_8 + contract_5_8 <= 10000
 R17: 10000 node_2_8 + contract_5_8 >= 0
 R18: 10000 node_2_9 + contract_5_9 <= 10000
 R19: 10000 node_2_9 + contract_5_9 >= 0
 R20: 10000 node_2_10 + contract_5_10 <= 10000
 R21: 10000 node_2_10 + contract_5_10 >= 0
 R22: - contract_5_0 + 10000 node_3_0 <= 10000
 R23: - contract_5_0 + 10000 node_3_0 >= 0
 R24: - contract_5_1 + 10000 node_3_1 <= 10000
 R25: - contract_5_1 + 10000 node_3_1 >= 0
 R26: - contract_5_2 + 10000 node_3_2 <= 10000
 R27: - contract_5_2 + 10000 node_3_2 >= 0
 R28: - contract_5_3 + 10000 node_3_3 <= 10000
 R29: - contract_5_3 + 10000 node_3_3 >= 0
 R30: - contract_5_4 + 10000 node_3_4 <= 10000
 R31: - contract_5_4 + 10000 node_3_4 >= 0
 R32: - contract_5_5 + 10000 node_3_5 <= 10000
 R33: - contract_5_5 + 10000 node_3_5 >= 0
 R34: - contract_5_6 + 10000 node_3_6 <= 10000
 R35: - contract_5_6 + 10000 node_3_6 >= 0
 R36: - contract_5_7 + 10000 node_3_7 <= 10000
 R37: - contract_5_7 + 10000 node_3_7 >= 0
 R38: - contract_5_8 + 10000 node_3_8 <= 10000
 R39: - contract_5_8 + 10000 node_3_8 >= 0
 R40: - contract_5_9 + 10000 node_3_9 <= 10000
 R41: - contract_5_9 + 10000 node_3_9 >= 0
 R42: - contract_5_10 + 10000 node_3_10 <= 10000
 R43: - contract_5_10 + 10000 node_3_10 >= 0
 R44: 10000 node_4_0 + contract_6_0 <= 10000
 R45: 10000 node_4_0 + contract_6_0 >= 0
 R46: 10000 node_4_1 + contract_6_1 <= 10000
 R47: 10000 node_4_1 + contract_6_1 >= 0
 R48: 10000 node_4_2 + contract_6_2 <= 10000
 R49: 10000 node_4_2 + contract_6_2 >= 0
 R50: 10000 node_4_3 + contract_6_3 <= 10000
 R51: 10000 node_4_3 + contract_6_3 >= 0
 R52: 10000 node_4_4 + contract_6_4 <= 10000
 R53: 10000 node_4_4 + contract_6_4 >= 0
 R54: 10000 node_4_5 + contract_6_5 <= 10000
 R55: 10000 node_4_5 + contract_6_5 >= 0
 R56: 10000 node_4_6 + contract_6_6 <= 10000
 R57: 10000 node_4_6 + contract_6_6 >= 0
 R58: 10000 node_4_7 + contract_6_7 <= 10000
 R59: 10000 node_4_7 + contract_6_7 >= 0
 R60: 10000 node_4_8 + contract_6_8 <= 10000
 R61: 10000 node_4_8 + contract_6_8 >= 0
 R62: 10000 node_4_9 + contract_6_9 <= 10000
 R63: 10000 node_4_9 + contract_6_9 >= 0
 R64: 10000 node_4_10 + contract_6_10 <= 10000
 R65: 10000 node_4_10 + contract_6_10 >= 0
 R66: - contract_6_0 + 10000 node_5_0 <= 10000
 R67: - contract_6_0 + 10000 node_5_0 >= 0
 R68: - contract_6_1 + 10000 node_5_1 <= 10000
 R69: - contract_6_1 + 10000 node_5_1 >= 0
 R70: - contract_6_2 + 10000 node_5_2 <= 10000
 R71: - contract_6_2 + 10000 node_5_2 >= 0
 R72: - contract_6_3 + 10000 node_5_3 <= 10000
 R73: - contract_6_3 + 10000 node_5_3 >= 0
 R74: - contract_6_4 + 10000 node_5_4 <= 10000
 R75: - contract_6_4 + 10000 node_5_4 >= 0
 R76: - contract_6_5 + 10000 node_5_5 <= 10000
 R77: - contract_6_5 + 10000 node_5_5 >= 0
 R78: - contract_6_6 + 10000 node_5_6 <= 10000
 R79: - contract_6_6 + 10000 node_5_6 >= 0
 R80: - contract_6_7 + 10000 node_5_7 <= 10000
 R81: - contract_6_7 + 10000 node_5_7 >= 0
 R82: - contract_6_8 + 10000 node_5_8 <= 10000
 R83: - contract_6_8 + 10000 node_5_8 >= 0
 R84: - contract_6_9 + 10000 node_5_9 <= 10000
 R85: - contract_6_9 + 10000 node_5_9 >= 0
 R86: - contract_6_10 + 10000 node_5_10 <= 10000
 R87: - contract_6_10 + 10000 node_5_10 >= 0
 R88: 10000 node_6_0 + contract_11_0 <= 10000
 R89: 10000 node_6_0 + contract_11_0 >= 0
 R90: 10000 node_6_1 + contract_11_1 <= 10000
 R91: 10000 node_6_1 + contract_11_1 >= 0
 R92: 10000 node_6_2 + contract_11_2 <= 10000
 R93: 10000 node_6_2 + contract_11_2 >= 0
 R94: 10000 node_6_3 + contract_11_3 <= 10000
 R95: 10000 node_6_3 + contract_11_3 >= 0
 R96: 10000 node_6_4 + contract_11_4 <= 10000
 R97: 10000 node_6_4 + contract_11_4 >= 0
 R98: 10000 node_6_5 + contract_11_5 <= 10000
 R99: 10000 node_6_5 + contract_11_5 >= 0
 R100: 10000 node_6_6 + contract_11_6 <= 10000
 R101: 10000 node_6_6 + contract_11_6 >= 0
 R102: 10000 node_6_7 + contract_11_7 <= 10000
 R103: 10000 node_6_7 + contract_11_7 >= 0
 R104: 10000 node_6_8 + contract_11_8 <= 10000
 R105: 10000 node_6_8 + contract_11_8 >= 0
 R106: 10000 node_6_9 + contract_11_9 <= 10000
 R107: 10000 node_6_9 + contract_11_9 >= 0
 R108: 10000 node_6_10 + contract_11_10 <= 10000
 R109: 10000 node_6_10 + contract_11_10 >= 0
 R110: - contract_11_0 + 10000 node_7_0 <= 10000
 R111: - contract_11_0 + 10000 node_7_0 >= 0
 R112: - contract_11_1 + 10000 node_7_1 <= 10000
 R113: - contract_11_1 + 10000 node_7_1 >= 0
 R114: - contract_11_2 + 10000 node_7_2 <= 10000
 R115: - contract_11_2 + 10000 node_7_2 >= 0
 R116: - contract_11_3 + 10000 node_7_3 <= 10000
 R117: - contract_11_3 + 10000 node_7_3 >= 0
 R118: - contract_11_4 + 10000 node_7_4 <= 10000
 R119: - contract_11_4 + 10000 node_7_4 >= 0
 R120: - contract_11_5 + 10000 node_7_5 <= 10000
 R121: - contract_11_5 + 10000 node_7_5 >= 0
 R122: - contract_11_6 + 10000 node_7_6 <= 10000
 R123: - contract_11_6 + 10000 node_7_6 >= 0
 R124: - contract_11_7 + 10000 node_7_7 <= 10000
 R125: - contract_11_7 + 10000 node_7_7 >= 0
 R126: - contract_11_8 + 10000 node_7_8 <= 10000
 R127: - contract_11_8 + 10000 node_7_8 >= 0
 R128: - contract_11_9 + 10000 node_7_9 <= 10000
 R129: - contract_11_9 + 10000 node_7_9 >= 0
 R130: - contract_11_10 + 10000 node_7_10 <= 10000
 R131: - contract_11_10 + 10000 node_7_10 >= 0
 R132: 10000 node_8_0 + contract_12_0 <= 10000
 R133: 10000 node_8_0 + contract_12_0 >= 0
 R134: 10000 node_8_1 + contract_12_1 <= 10000
 R135: 10000 node_8_1 + contract_12_1 >= 0
 R136: 10000 node_8_2 + contract_12_2 <= 10000
 R137: 10000 node_8_2 + contract_12_2 >= 0
 R138: 10000 node_8_3 + contract_12_3 <= 10000
 R139: 10000 node_8_3 + contract_12_3 >= 0
 R140: 10000 node_8_4 + contract_12_4 <= 10000
 R141: 10000 node_8_4 + contract_12_4 >= 0
 R142: 10000 node_8_5 + contract_12_5 <= 10000
 R143: 10000 node_8_5 + contract_12_5 >= 0
 R144: 10000 node_8_6 + contract_12_6 <= 10000
 R145: 10000 node_8_6 + contract_12_6 >= 0
 R146: 10000 node_8_7 + contract_12_7 <= 10000
 R147: 10000 node_8_7 + contract_12_7 >= 0
 R148: 10000 node_8_8 + contract_12_8 <= 10000
 R149: 10000 node_8_8 + contract_12_8 >= 0
 R150: 10000 node_8_9 + contract_12_9 <= 10000
 R151: 10000 node_8_9 + contract_12_9 >= 0
 R152: 10000 node_8_10 + contract_12_10 <= 10000
 R153: 10000 node_8_10 + contract_12_10 >= 0
 R154: - contract_12_0 + 10000 node_9_0 <= 10000
 R155: - contract_12_0 + 10000 node_9_0 >= 0
 R156: - contract_12_1 + 10000 node_9_1 <= 10000
 R157: - contract_12_1 + 10000 node_9_1 >= 0
 R158: - contract_12_2 + 10000 node_9_2 <= 10000
 R159: - contract_12_2 + 10000 node_9_2 >= 0
 R160: - contract_12_3 + 10000 node_9_3 <= 10000
 R161: - contract_12_3 + 10000 node_9_3 >= 0
 R162: - contract_12_4 + 10000 node_9_4 <= 10000
 R163: - contract_12_4 + 10000 node_9_4 >= 0
 R164: - contract_12_5 + 10000 node_9_5 <= 10000
 R165: - contract_12_5 + 10000 node_9_5 >= 0
 R166: - contract_12_6 + 10000 node_9_6 <= 10000
 R167: - contract_12_6 + 10000 node_9_6 >= 0
 R168: - contract_12_7 + 10000 node_9_7 <= 10000
 R169: - contract_12_7 + 10000 node_9_7 >= 0
 R170: - contract_12_8 + 10000 node_9_8 <= 10000
 R171: - contract_12_8 + 10000 node_9_8 >= 0
 R172: - contract_12_9 + 10000 node_9_9 <= 10000
 R173: - contract_12_9 + 10000 node_9_9 >= 0
 R174: - contract_12_10 + 10000 node_9_10 <= 10000
 R175: - contract_12_10 + 10000 node_9_10 >= 0
 R176: node_0_0 = 1
 R177: 10000 node_13_0 + contract_1_0 - contract_13_0 <= 9997
 R178: 10000 node_13_0 + contract_1_0 - contract_13_0 >= -3
 R179: 10000 node_13_1 + contract_1_1 - contract_13_1 <= 9997
 R180: 10000 node_13_1 + contract_1_1 - contract_13_1 >= -3
 R181: 10000 node_13_2 + contract_1_2 - contract_13_2 <= 9997
 R182: 10000 node_13_2 + contract_1_2 - contract_13_2 >= -3
 R183: 10000 node_13_3 + contract_1_3 - contract_13_3 <= 9997
 R184: 10000 node_13_3 + contract_1_3 - contract_13_3 >= -3
 R185: 10000 node_13_4 + contract_1_4 - contract_13_4 <= 9997
 R186: 10000 node_13_4 + contract_1_4 - contract_13_4 >= -3
 R187: 10000 node_13_5 + contract_1_5 - contract_13_5 <= 9997
 R188: 10000 node_13_5 + contract_1_5 - contract_13_5 >= -3
 R189: 10000 node_13_6 + contract_1_6 - contract_13_6 <= 9997
 R190: 10000 node_13_6 + contract_1_6 - contract_13_6 >= -3
 R191: 10000 node_13_7 + contract_1_7 - contract_13_7 <= 9997
 R192: 10000 node_13_7 + contract_1_7 - contract_13_7 >= -3
 R193: 10000 node_13_8 + contract_1_8 - contract_13_8 <= 9997
 R194: 10000 node_13_8 + contract_1_8 - contract_13_8 >= -3
 R195: 10000 node_13_9 + contract_1_9 - contract_13_9 <= 9997
 R196: 10000 node_13_9 + contract_1_9 - contract_13_9 >= -3
 R197: 10000 node_13_10 + contract_1_10 - contract_13_10 <= 9997
 R198: 10000 node_13_10 + contract_1_10 - contract_13_10 >= -3
 R199: - contract_1_0 + contract_13_0 + 10000 node_14_0 <= 9997
 R200: - contract_1_0 + contract_13_0 + 10000 node_14_0 >= -3
 R201: - contract_1_1 + contract_13_1 + 10000 node_14_1 <= 9997
 R202: - contract_1_1 + contract_13_1 + 10000 node_14_1 >= -3
 R203: - contract_1_2 + contract_13_2 + 10000 node_14_2 <= 9997
 R204: - contract_1_2 + contract_13_2 + 10000 node_14_2 >= -3
 R205: - contract_1_3 + contract_13_3 + 10000 node_14_3 <= 9997
 R206: - contract_1_3 + contract_13_3 + 10000 node_14_3 >= -3
 R207: - contract_1_4 + contract_13_4 + 10000 node_14_4 <= 9997
 R208: - contract_1_4 + contract_13_4 + 10000 node_14_4 >= -3
 R209: - contract_1_5 + contract_13_5 + 10000 node_14_5 <= 9997
 R210: - contract_1_5 + contract_13_5 + 10000 node_14_5 >= -3
 R211: - contract_1_6 + contract_13_6 + 10000 node_14_6 <= 9997
 R212: - contract_1_6 + contract_13_6 + 10000 node_14_6 >= -3
 R213: - contract_1_7 + contract_13_7 + 10000 node_14_7 <= 9997
 R214: - contract_1_7 + contract_13_7 + 10000 node_14_7 >= -3
 R215: - contract_1_8 + contract_13_8 + 10000 node_14_8 <= 9997
 R216: - contract_1_8 + contract_13_8 + 10000 node_14_8 >= -3
 R217: - contract_1_9 + contract_13_9 + 10000 node_14_9 <= 9997
 R218: - contract_1_9 + contract_13_9 + 10000 node_14_9 >= -3
 R219: - contract_1_10 + contract_13_10 + 10000 node_14_10 <= 9997
 R220: - contract_1_10 + contract_13_10 + 10000 node_14_10 >= -3
 R221: 10000 node_15_0 + contract_2_0 - contract_14_0 <= 9997
 R222: 10000 node_15_0 + contract_2_0 - contract_14_0 >= -3
 R223: 10000 node_15_1 + contract_2_1 - contract_14_1 <= 9997
 R224: 10000 node_15_1 + contract_2_1 - contract_14_1 >= -3
 R225: 10000 node_15_2 + contract_2_2 - contract_14_2 <= 9997
 R226: 10000 node_15_2 + contract_2_2 - contract_14_2 >= -3
 R227: 10000 node_15_3 + contract_2_3 - contract_14_3 <= 9997
 R228: 10000 node_15_3 + contract_2_3 - contract_14_3 >= -3
 R229: 10000 node_15_4 + contract_2_4 - contract_14_4 <= 9997
 R230: 10000 node_15_4 + contract_2_4 - contract_14_4 >= -3
 R231: 10000 node_15_5 + contract_2_5 - contract_14_5 <= 9997
 R232: 10000 node_15_5 + contract_2_5 - contract_14_5 >= -3
 R233: 10000 node_15_6 + contract_2_6 - contract_14_6 <= 9997
 R234: 10000 node_15_6 + contract_2_6 - contract_14_6 >= -3
 R235: 10000 node_15_7 + contract_2_7 - contract_14_7 <= 9997
 R236: 10000 node_15_7 + contract_2_7 - contract_14_7 >= -3
 R237: 10000 node_15_8 + contract_2_8 - contract_14_8 <= 9997
 R238: 10000 node_15_8 + contract_2_8 - contract_14_8 >= -3
 R239: 10000 node_15_9 + contract_2_9 - contract_14_9 <= 9997
 R240: 10000 node_15_9 + contract_2_9 - contract_14_9 >= -3
 R241: 10000 node_15_10 + contract_2_10 - contract_14_10 <= 9997
 R242: 10000 node_15_10 + contract_2_10 - contract_14_10 >= -3
 R243: - contract_2_0 + contract_14_0 + 10000 node_16_0 <= 9997
 R244: - contract_2_0 + contract_14_0 + 10000 node_16_0 >= -3
 R245: - contract_2_1 + contract_14_1 + 10000 node_16_1 <= 9997
 R246: - contract_2_1 + contract_14_1 + 10000 node_16_1 >= -3
 R247: - contract_2_2 + contract_14_2 + 10000 node_16_2 <= 9997
 R248: - contract_2_2 + contract_14_2 + 10000 node_16_2 >= -3
 R249: - contract_2_3 + contract_14_3 + 10000 node_16_3 <= 9997
 R250: - contract_2_3 + contract_14_3 + 10000 node_16_3 >= -3
 R251: - contract_2_4 + contract_14_4 + 10000 node_16_4 <= 9997
 R252: - contract_2_4 + contract_14_4 + 10000 node_16_4 >= -3
 R253: - contract_2_5 + contract_14_5 + 10000 node_16_5 <= 9997
 R254: - contract_2_5 + contract_14_5 + 10000 node_16_5 >= -3
 R255: - contract_2_6 + contract_14_6 + 10000 node_16_6 <= 9997
 R256: - contract_2_6 + contract_14_6 + 10000 node_16_6 >= -3
 R257: - contract_2_7 + contract_14_7 + 10000 node_16_7 <= 9997
 R258: - contract_2_7 + contract_14_7 + 10000 node_16_7 >= -3
 R259: - contract_2_8 + contract_14_8 + 10000 node_16_8 <= 9997
 R260: - contract_2_8 + contract_14_8 + 10000 node_16_8 >= -3
 R261: - contract_2_9 + contract_14_9 + 10000 node_16_9 <= 9997
 R262: - contract_2_9 + contract_14_9 + 10000 node_16_9 >= -3
 R263: - contract_2_10 + contract_14_10 + 10000 node_16_10 <= 9997
 R264: - contract_2_10 + contract_14_10 + 10000 node_16_10 >= -3
 R265: - contract_13_0 + 10000 node_19_0 + contract_7_0 <= 9997
 R266: - contract_13_0 + 10000 node_19_0 + contract_7_0 >= -3
 R267: - contract_13_1 + 10000 node_19_1 + contract_7_1 <= 9997
 R268: - contract_13_1 + 10000 node_19_1 + contract_7_1 >= -3
 R269: - contract_13_2 + 10000 node_19_2 + contract_7_2 <= 9997
 R270: - contract_13_2 + 10000 node_19_2 + contract_7_2 >= -3
 R271: - contract_13_3 + 10000 node_19_3 + contract_7_3 <= 9997
 R272: - contract_13_3 + 10000 node_19_3 + contract_7_3 >= -3
 R273: - contract_13_4 + 10000 node_19_4 + contract_7_4 <= 9997
 R274: - contract_13_4 + 10000 node_19_4 + contract_7_4 >= -3
 R275: - contract_13_5 + 10000 node_19_5 + contract_7_5 <= 9997
 R276: - contract_13_5 + 10000 node_19_5 + contract_7_5 >= -3
 R277: - contract_13_6 + 10000 node_19_6 + contract_7_6 <= 9997
 R278: - contract_13_6 + 10000 node_19_6 + contract_7_6 >= -3
 R279: - contract_13_7 + 10000 node_19_7 + contract_7_7 <= 9997
 R280: - contract_13_7 + 10000 node_19_7 + contract_7_7 >= -3
 R281: - contract_13_8 + 10000 node_19_8 + contract_7_8 <= 9997
 R282: - contract_13_8 + 10000 node_19_8 + contract_7_8 >= -3
 R283: - contract_13_9 + 10000 node_19_9 + contract_7_9 <= 9997
 R284: - contract_13_9 + 10000 node_19_9 + contract_7_9 >= -3
 R285: - contract_13_10 + 10000 node_19_10 + contract_7_10 <= 9997
 R286: - contract_13_10 + 10000 node_19_10 + contract_7_10 >= -3
 R287: contract_13_0 - contract_7_0 + 10000 node_20_0 <= 9997
 R288: contract_13_0 - contract_7_0 + 10000 node_20_0 >= -3
 R289: contract_13_1 - contract_7_1 + 10000 node_20_1 <= 9997
 R290: contract_13_1 - contract_7_1 + 10000 node_20_1 >= -3
 R291: contract_13_2 - contract_7_2 + 10000 node_20_2 <= 9997
 R292: contract_13_2 - contract_7_2 + 10000 node_20_2 >= -3
 R293: contract_13_3 - contract_7_3 + 10000 node_20_3 <= 9997
 R294: contract_13_3 - contract_7_3 + 10000 node_20_3 >= -3
 R295: contract_13_4 - contract_7_4 + 10000 node_20_4 <= 9997
 R296: contract_13_4 - contract_7_4 + 10000 node_20_4 >= -3
 R297: contract_13_5 - contract_7_5 + 10000 node_20_5 <= 9997
 R298: contract_13_5 - contract_7_5 + 10000 node_20_5 >= -3
 R299: contract_13_6 - contract_7_6 + 10000 node_20_6 <= 9997
 R300: contract_13_6 - contract_7_6 + 10000 node_20_6 >= -3
 R301: contract_13_7 - contract_7_7 + 10000 node_20_7 <= 9997
 R302: contract_13_7 - contract_7_7 + 10000 node_20_7 >= -3
 R303: contract_13_8 - contract_7_8 + 10000 node_20_8 <= 9997
 R304: contract_13_8 - contract_7_8 + 10000 node_20_8 >= -3
 R305: contract_13_9 - contract_7_9 + 10000 node_20_9 <= 9997
 R306: contract_13_9 - contract_7_9 + 10000 node_20_9 >= -3
 R307: contract_13_10 - contract_7_10 + 10000 node_20_10 <= 9997
 R308: contract_13_10 - contract_7_10 + 10000 node_20_10 >= -3
 R309: - contract_14_0 + 10000 node_21_0 + contract_8_0 <= 9997
 R310: - contract_14_0 + 10000 node_21_0 + contract_8_0 >= -3
 R311: - contract_14_1 + 10000 node_21_1 + contract_8_1 <= 9997
 R312: - contract_14_1 + 10000 node_21_1 + contract_8_1 >= -3
 R313: - contract_14_2 + 10000 node_21_2 + contract_8_2 <= 9997
 R314: - contract_14_2 + 10000 node_21_2 + contract_8_2 >= -3
 R315: - contract_14_3 + 10000 node_21_3 + contract_8_3 <= 9997
 R316: - contract_14_3 + 10000 node_21_3 + contract_8_3 >= -3
 R317: - contract_14_4 + 10000 node_21_4 + contract_8_4 <= 9997
 R318: - contract_14_4 + 10000 node_21_4 + contract_8_4 >= -3
 R319: - contract_14_5 + 10000 node_21_5 + contract_8_5 <= 9997
 R320: - contract_14_5 + 10000 node_21_5 + contract_8_5 >= -3
 R321: - contract_14_6 + 10000 node_21_6 + contract_8_6 <= 9997
 R322: - contract_14_6 + 10000 node_21_6 + contract_8_6 >= -3
 R323: - contract_14_7 + 10000 node_21_7 + contract_8_7 <= 9997
 R324: - contract_14_7 + 10000 node_21_7 + contract_8_7 >= -3
 R325: - contract_14_8 + 10000 node_21_8 + contract_8_8 <= 9997
 R326: - contract_14_8 + 10000 node_21_8 + contract_8_8 >= -3
 R327: - contract_14_9 + 10000 node_21_9 + contract_8_9 <= 9997
 R328: - contract_14_9 + 10000 node_21_9 + contract_8_9 >= -3
 R329: - contract_14_10 + 10000 node_21_10 + contract_8_10 <= 9997
 R330: - contract_14_10 + 10000 node_21_10 + contract_8_10 >= -3
 R331: contract_14_0 - contract_8_0 + 10000 node_22_0 <= 9997
 R332: contract_14_0 - contract_8_0 + 10000 node_22_0 >= -3
 R333: contract_14_1 - contract_8_1 + 10000 node_22_1 <= 9997
 R334: contract_14_1 - contract_8_1 + 10000 node_22_1 >= -3
 R335: contract_14_2 - contract_8_2 + 10000 node_22_2 <= 9997
 R336: contract_14_2 - contract_8_2 + 10000 node_22_2 >= -3
 R337: contract_14_3 - contract_8_3 + 10000 node_22_3 <= 9997
 R338: contract_14_3 - contract_8_3 + 10000 node_22_3 >= -3
 R339: contract_14_4 - contract_8_4 + 10000 node_22_4 <= 9997
 R340: contract_14_4 - contract_8_4 + 10000 node_22_4 >= -3
 R341: contract_14_5 - contract_8_5 + 10000 node_22_5 <= 9997
 R342: contract_14_5 - contract_8_5 + 10000 node_22_5 >= -3
 R343: contract_14_6 - contract_8_6 + 10000 node_22_6 <= 9997
 R344: contract_14_6 - contract_8_6 + 10000 node_22_6 >= -3
 R345: contract_14_7 - contract_8_7 + 10000 node_22_7 <= 9997
 R346: contract_14_7 - contract_8_7 + 10000 node_22_7 >= -3
 R347: contract_14_8 - contract_8_8 + 10000 node_22_8 <= 9997
 R348: contract_14_8 - contract_8_8 + 10000 node_22_8 >= -3
 R349: contract_14_9 - contract_8_9 + 10000 node_22_9 <= 9997
 R350: contract_14_9 - contract_8_9 + 10000 node_22_9 >= -3
 R351: contract_14_10 - contract_8_10 + 10000 node_22_10 <= 9997
 R352: contract_14_10 - contract_8_10 + 10000 node_22_10 >= -3
 R353: node_10_0 = 1
 R354: - contract_13_0 + 10000 node_25_0 <= 9999
 R355: - contract_13_0 + 10000 node_25_0 >= -1
 R356: contract_13_0 + 10000 node_26_0 <= 10003
 R357: contract_13_0 + 10000 node_26_0 >= 3
 R358: - contract_14_0 + 10000 node_27_0 <= 10064
 R359: - contract_14_0 + 10000 node_27_0 >= 64
 R360: contract_14_0 + 10000 node_28_0 <= 9986
 R361: contract_14_0 + 10000 node_28_0 >= -14
 R362: - contract_13_1 + 10000 node_31_1 <= 9999
 R363: - contract_13_1 + 10000 node_31_1 >= -1
 R364: contract_13_1 + 10000 node_32_1 <= 10003
 R365: contract_13_1 + 10000 node_32_1 >= 3
 R366: - contract_14_1 + 10000 node_33_1 <= 10064
 R367: - contract_14_1 + 10000 node_33_1 >= 64
 R368: contract_14_1 + 10000 node_34_1 <= 9986
 R369: contract_14_1 + 10000 node_34_1 >= -14
 R370: - contract_13_2 + 10000 node_37_2 <= 9999
 R371: - contract_13_2 + 10000 node_37_2 >= -1
 R372: contract_13_2 + 10000 node_38_2 <= 10003
 R373: contract_13_2 + 10000 node_38_2 >= 3
 R374: - contract_14_2 + 10000 node_39_2 <= 10064
 R375: - contract_14_2 + 10000 node_39_2 >= 64
 R376: contract_14_2 + 10000 node_40_2 <= 9986
 R377: contract_14_2 + 10000 node_40_2 >= -14
 R378: - contract_13_3 + 10000 node_43_3 <= 9999
 R379: - contract_13_3 + 10000 node_43_3 >= -1
 R380: contract_13_3 + 10000 node_44_3 <= 10003
 R381: contract_13_3 + 10000 node_44_3 >= 3
 R382: - contract_14_3 + 10000 node_45_3 <= 10064
 R383: - contract_14_3 + 10000 node_45_3 >= 64
 R384: contract_14_3 + 10000 node_46_3 <= 9986
 R385: contract_14_3 + 10000 node_46_3 >= -14
 R386: - contract_13_4 + 10000 node_49_4 <= 9999
 R387: - contract_13_4 + 10000 node_49_4 >= -1
 R388: contract_13_4 + 10000 node_50_4 <= 10003
 R389: contract_13_4 + 10000 node_50_4 >= 3
 R390: - contract_14_4 + 10000 node_51_4 <= 10064
 R391: - contract_14_4 + 10000 node_51_4 >= 64
 R392: contract_14_4 + 10000 node_52_4 <= 9986
 R393: contract_14_4 + 10000 node_52_4 >= -14
 R394: - contract_13_5 + 10000 node_55_5 <= 9999
 R395: - contract_13_5 + 10000 node_55_5 >= -1
 R396: contract_13_5 + 10000 node_56_5 <= 10003
 R397: contract_13_5 + 10000 node_56_5 >= 3
 R398: - contract_14_5 + 10000 node_57_5 <= 10064
 R399: - contract_14_5 + 10000 node_57_5 >= 64
 R400: contract_14_5 + 10000 node_58_5 <= 9986
 R401: contract_14_5 + 10000 node_58_5 >= -14
 R402: - contract_13_6 + 10000 node_61_6 <= 9999
 R403: - contract_13_6 + 10000 node_61_6 >= -1
 R404: contract_13_6 + 10000 node_62_6 <= 10003
 R405: contract_13_6 + 10000 node_62_6 >= 3
 R406: - contract_14_6 + 10000 node_63_6 <= 10064
 R407: - contract_14_6 + 10000 node_63_6 >= 64
 R408: contract_14_6 + 10000 node_64_6 <= 9986
 R409: contract_14_6 + 10000 node_64_6 >= -14
 R410: - contract_13_7 + 10000 node_67_7 <= 9999
 R411: - contract_13_7 + 10000 node_67_7 >= -1
 R412: contract_13_7 + 10000 node_68_7 <= 10003
 R413: contract_13_7 + 10000 node_68_7 >= 3
 R414: - contract_14_7 + 10000 node_69_7 <= 10064
 R415: - contract_14_7 + 10000 node_69_7 >= 64
 R416: contract_14_7 + 10000 node_70_7 <= 9986
 R417: contract_14_7 + 10000 node_70_7 >= -14
 R418: - contract_13_8 + 10000 node_73_8 <= 9999
 R419: - contract_13_8 + 10000 node_73_8 >= -1
 R420: contract_13_8 + 10000 node_74_8 <= 10003
 R421: contract_13_8 + 10000 node_74_8 >= 3
 R422: - contract_14_8 + 10000 node_75_8 <= 10064
 R423: - contract_14_8 + 10000 node_75_8 >= 64
 R424: contract_14_8 + 10000 node_76_8 <= 9986
 R425: contract_14_8 + 10000 node_76_8 >= -14
 R426: - contract_13_9 + 10000 node_79_9 <= 9999
 R427: - contract_13_9 + 10000 node_79_9 >= -1
 R428: contract_13_9 + 10000 node_80_9 <= 10003
 R429: contract_13_9 + 10000 node_80_9 >= 3
 R430: - contract_14_9 + 10000 node_81_9 <= 10064
 R431: - contract_14_9 + 10000 node_81_9 >= 64
 R432: contract_14_9 + 10000 node_82_9 <= 9986
 R433: contract_14_9 + 10000 node_82_9 >= -14
 R434: - contract_13_0 + 10000 node_85_0 <= 9999
 R435: - contract_13_0 + 10000 node_85_0 >= -1
 R436: contract_13_0 + 10000 node_86_0 <= 10003
 R437: contract_13_0 + 10000 node_86_0 >= 3
 R438: - contract_14_0 + 10000 node_87_0 <= 10014
 R439: - contract_14_0 + 10000 node_87_0 >= 14
 R440: contract_14_0 + 10000 node_88_0 <= 10014
 R441: contract_14_0 + 10000 node_88_0 >= 14
 R442: - contract_13_1 + 10000 node_91_1 <= 9999
 R443: - contract_13_1 + 10000 node_91_1 >= -1
 R444: contract_13_1 + 10000 node_92_1 <= 10003
 R445: contract_13_1 + 10000 node_92_1 >= 3
 R446: - contract_14_1 + 10000 node_93_1 <= 10014
 R447: - contract_14_1 + 10000 node_93_1 >= 14
 R448: contract_14_1 + 10000 node_94_1 <= 10014
 R449: contract_14_1 + 10000 node_94_1 >= 14
 R450: - contract_13_2 + 10000 node_97_2 <= 9999
 R451: - contract_13_2 + 10000 node_97_2 >= -1
 R452: contract_13_2 + 10000 node_98_2 <= 10003
 R453: contract_13_2 + 10000 node_98_2 >= 3
 R454: - contract_14_2 + 10000 node_99_2 <= 10014
 R455: - contract_14_2 + 10000 node_99_2 >= 14
 R456: contract_14_2 + 10000 node_100_2 <= 10014
 R457: contract_14_2 + 10000 node_100_2 >= 14
 R458: - contract_13_3 + 10000 node_103_3 <= 9999
 R459: - contract_13_3 + 10000 node_103_3 >= -1
 R460: contract_13_3 + 10000 node_104_3 <= 10003
 R461: contract_13_3 + 10000 node_104_3 >= 3
 R462: - contract_14_3 + 10000 node_105_3 <= 10014
 R463: - contract_14_3 + 10000 node_105_3 >= 14
 R464: contract_14_3 + 10000 node_106_3 <= 10014
 R465: contract_14_3 + 10000 node_106_3 >= 14
 R466: - contract_13_4 + 10000 node_109_4 <= 9999
 R467: - contract_13_4 + 10000 node_109_4 >= -1
 R468: contract_13_4 + 10000 node_110_4 <= 10003
 R469: contract_13_4 + 10000 node_110_4 >= 3
 R470: - contract_14_4 + 10000 node_111_4 <= 10014
 R471: - contract_14_4 + 10000 node_111_4 >= 14
 R472: contract_14_4 + 10000 node_112_4 <= 10014
 R473: contract_14_4 + 10000 node_112_4 >= 14
 R474: - contract_13_5 + 10000 node_115_5 <= 9999
 R475: - contract_13_5 + 10000 node_115_5 >= -1
 R476: contract_13_5 + 10000 node_116_5 <= 10003
 R477: contract_13_5 + 10000 node_116_5 >= 3
 R478: - contract_14_5 + 10000 node_117_5 <= 10014
 R479: - contract_14_5 + 10000 node_117_5 >= 14
 R480: contract_14_5 + 10000 node_118_5 <= 10014
 R481: contract_14_5 + 10000 node_118_5 >= 14
 R482: - contract_13_6 + 10000 node_121_6 <= 9999
 R483: - contract_13_6 + 10000 node_121_6 >= -1
 R484: contract_13_6 + 10000 node_122_6 <= 10003
 R485: contract_13_6 + 10000 node_122_6 >= 3
 R486: - contract_14_6 + 10000 node_123_6 <= 10014
 R487: - contract_14_6 + 10000 node_123_6 >= 14
 R488: contract_14_6 + 10000 node_124_6 <= 10014
 R489: contract_14_6 + 10000 node_124_6 >= 14
 R490: - contract_13_7 + 10000 node_127_7 <= 9999
 R491: - contract_13_7 + 10000 node_127_7 >= -1
 R492: contract_13_7 + 10000 node_128_7 <= 10003
 R493: contract_13_7 + 10000 node_128_7 >= 3
 R494: - contract_14_7 + 10000 node_129_7 <= 10014
 R495: - contract_14_7 + 10000 node_129_7 >= 14
 R496: contract_14_7 + 10000 node_130_7 <= 10014
 R497: contract_14_7 + 10000 node_130_7 >= 14
 R498: - contract_13_8 + 10000 node_133_8 <= 9999
 R499: - contract_13_8 + 10000 node_133_8 >= -1
 R500: contract_13_8 + 10000 node_134_8 <= 10003
 R501: contract_13_8 + 10000 node_134_8 >= 3
 R502: - contract_14_8 + 10000 node_135_8 <= 10014
 R503: - contract_14_8 + 10000 node_135_8 >= 14
 R504: contract_14_8 + 10000 node_136_8 <= 10014
 R505: contract_14_8 + 10000 node_136_8 >= 14
 R506: - contract_13_9 + 10000 node_139_9 <= 9999
 R507: - contract_13_9 + 10000 node_139_9 >= -1
 R508: contract_13_9 + 10000 node_140_9 <= 10003
 R509: contract_13_9 + 10000 node_140_9 >= 3
 R510: - contract_14_9 + 10000 node_141_9 <= 10014
 R511: - contract_14_9 + 10000 node_141_9 >= 14
 R512: contract_14_9 + 10000 node_142_9 <= 10014
 R513: contract_14_9 + 10000 node_142_9 >= 14
 R514: - contract_13_0 + 10000 node_145_0 <= 9999
 R515: - contract_13_0 + 10000 node_145_0 >= -1
 R516: contract_13_0 + 10000 node_146_0 <= 10003
 R517: contract_13_0 + 10000 node_146_0 >= 3
 R518: - contract_14_0 + 10000 node_147_0 <= 9986
 R519: - contract_14_0 + 10000 node_147_0 >= -14
 R520: contract_14_0 + 10000 node_148_0 <= 10064
 R521: contract_14_0 + 10000 node_148_0 >= 64
 R522: - contract_13_1 + 10000 node_151_1 <= 9999
 R523: - contract_13_1 + 10000 node_151_1 >= -1
 R524: contract_13_1 + 10000 node_152_1 <= 10003
 R525: contract_13_1 + 10000 node_152_1 >= 3
 R526: - contract_14_1 + 10000 node_153_1 <= 9986
 R527: - contract_14_1 + 10000 node_153_1 >= -14
 R528: contract_14_1 + 10000 node_154_1 <= 10064
 R529: contract_14_1 + 10000 node_154_1 >= 64
 R530: - contract_13_2 + 10000 node_157_2 <= 9999
 R531: - contract_13_2 + 10000 node_157_2 >= -1
 R532: contract_13_2 + 10000 node_158_2 <= 10003
 R533: contract_13_2 + 10000 node_158_2 >= 3
 R534: - contract_14_2 + 10000 node_159_2 <= 9986
 R535: - contract_14_2 + 10000 node_159_2 >= -14
 R536: contract_14_2 + 10000 node_160_2 <= 10064
 R537: contract_14_2 + 10000 node_160_2 >= 64
 R538: - contract_13_3 + 10000 node_163_3 <= 9999
 R539: - contract_13_3 + 10000 node_163_3 >= -1
 R540: contract_13_3 + 10000 node_164_3 <= 10003
 R541: contract_13_3 + 10000 node_164_3 >= 3
 R542: - contract_14_3 + 10000 node_165_3 <= 9986
 R543: - contract_14_3 + 10000 node_165_3 >= -14
 R544: contract_14_3 + 10000 node_166_3 <= 10064
 R545: contract_14_3 + 10000 node_166_3 >= 64
 R546: - contract_13_4 + 10000 node_169_4 <= 9999
 R547: - contract_13_4 + 10000 node_169_4 >= -1
 R548: contract_13_4 + 10000 node_170_4 <= 10003
 R549: contract_13_4 + 10000 node_170_4 >= 3
 R550: - contract_14_4 + 10000 node_171_4 <= 9986
 R551: - contract_14_4 + 10000 node_171_4 >= -14
 R552: contract_14_4 + 10000 node_172_4 <= 10064
 R553: contract_14_4 + 10000 node_172_4 >= 64
 R554: - contract_13_5 + 10000 node_175_5 <= 9999
 R555: - contract_13_5 + 10000 node_175_5 >= -1
 R556: contract_13_5 + 10000 node_176_5 <= 10003
 R557: contract_13_5 + 10000 node_176_5 >= 3
 R558: - contract_14_5 + 10000 node_177_5 <= 9986
 R559: - contract_14_5 + 10000 node_177_5 >= -14
 R560: contract_14_5 + 10000 node_178_5 <= 10064
 R561: contract_14_5 + 10000 node_178_5 >= 64
 R562: - contract_13_6 + 10000 node_181_6 <= 9999
 R563: - contract_13_6 + 10000 node_181_6 >= -1
 R564: contract_13_6 + 10000 node_182_6 <= 10003
 R565: contract_13_6 + 10000 node_182_6 >= 3
 R566: - contract_14_6 + 10000 node_183_6 <= 9986
 R567: - contract_14_6 + 10000 node_183_6 >= -14
 R568: contract_14_6 + 10000 node_184_6 <= 10064
 R569: contract_14_6 + 10000 node_184_6 >= 64
 R570: - contract_13_7 + 10000 node_187_7 <= 9999
 R571: - contract_13_7 + 10000 node_187_7 >= -1
 R572: contract_13_7 + 10000 node_188_7 <= 10003
 R573: contract_13_7 + 10000 node_188_7 >= 3
 R574: - contract_14_7 + 10000 node_189_7 <= 9986
 R575: - contract_14_7 + 10000 node_189_7 >= -14
 R576: contract_14_7 + 10000 node_190_7 <= 10064
 R577: contract_14_7 + 10000 node_190_7 >= 64
 R578: - contract_13_8 + 10000 node_193_8 <= 9999
 R579: - contract_13_8 + 10000 node_193_8 >= -1
 R580: contract_13_8 + 10000 node_194_8 <= 10003
 R581: contract_13_8 + 10000 node_194_8 >= 3
 R582: - contract_14_8 + 10000 node_195_8 <= 9986
 R583: - contract_14_8 + 10000 node_195_8 >= -14
 R584: contract_14_8 + 10000 node_196_8 <= 10064
 R585: contract_14_8 + 10000 node_196_8 >= 64
 R586: - contract_13_9 + 10000 node_199_9 <= 9999
 R587: - contract_13_9 + 10000 node_199_9 >= -1
 R588: contract_13_9 + 10000 node_200_9 <= 10003
 R589: contract_13_9 + 10000 node_200_9 >= 3
 R590: - contract_14_9 + 10000 node_201_9 <= 9986
 R591: - contract_14_9 + 10000 node_201_9 >= -14
 R592: contract_14_9 + 10000 node_202_9 <= 10064
 R593: contract_14_9 + 10000 node_202_9 >= 64
 R594: - contract_1_0 + contract_1_1 - 0.5 contract_3_0 = 0
 R595: - contract_1_1 + contract_1_2 - 0.5 contract_3_1 = 0
 R596: - contract_1_2 + contract_1_3 - 0.5 contract_3_2 = 0
 R597: - contract_1_3 + contract_1_4 - 0.5 contract_3_3 = 0
 R598: - contract_1_4 + contract_1_5 - 0.5 contract_3_4 = 0
 R599: - contract_1_5 + contract_1_6 - 0.5 contract_3_5 = 0
 R600: - contract_1_6 + contract_1_7 - 0.5 contract_3_6 = 0
 R601: - contract_1_7 + contract_1_8 - 0.5 contract_3_7 = 0
 R602: - contract_1_8 + contract_1_9 - 0.5 contract_3_8 = 0
 R603: - contract_1_9 + contract_1_10 - 0.5 contract_3_9 = 0
 R604: - contract_2_0 + contract_2_1 - 0.5 contract_4_0 = 0
 R605: - contract_2_1 + contract_2_2 - 0.5 contract_4_1 = 0
 R606: - contract_2_2 + contract_2_3 - 0.5 contract_4_2 = 0
 R607: - contract_2_3 + contract_2_4 - 0.5 contract_4_3 = 0
 R608: - contract_2_4 + contract_2_5 - 0.5 contract_4_4 = 0
 R609: - contract_2_5 + contract_2_6 - 0.5 contract_4_5 = 0
 R610: - contract_2_6 + contract_2_7 - 0.5 contract_4_6 = 0
 R611: - contract_2_7 + contract_2_8 - 0.5 contract_4_7 = 0
 R612: - contract_2_8 + contract_2_9 - 0.5 contract_4_8 = 0
 R613: - contract_2_9 + contract_2_10 - 0.5 contract_4_9 = 0
 R614: - 0.5 contract_5_0 - contract_3_0 + contract_3_1 = 0
 R615: - 0.5 contract_5_1 - contract_3_1 + contract_3_2 = 0
 R616: - 0.5 contract_5_2 - contract_3_2 + contract_3_3 = 0
 R617: - 0.5 contract_5_3 - contract_3_3 + contract_3_4 = 0
 R618: - 0.5 contract_5_4 - contract_3_4 + contract_3_5 = 0
 R619: - 0.5 contract_5_5 - contract_3_5 + contract_3_6 = 0
 R620: - 0.5 contract_5_6 - contract_3_6 + contract_3_7 = 0
 R621: - 0.5 contract_5_7 - contract_3_7 + contract_3_8 = 0
 R622: - 0.5 contract_5_8 - contract_3_8 + contract_3_9 = 0
 R623: - 0.5 contract_5_9 - contract_3_9 + contract_3_10 = 0
 R624: - 0.5 contract_6_0 - contract_4_0 + contract_4_1 = 0
 R625: - 0.5 contract_6_1 - contract_4_1 + contract_4_2 = 0
 R626: - 0.5 contract_6_2 - contract_4_2 + contract_4_3 = 0
 R627: - 0.5 contract_6_3 - contract_4_3 + contract_4_4 = 0
 R628: - 0.5 contract_6_4 - contract_4_4 + contract_4_5 = 0
 R629: - 0.5 contract_6_5 - contract_4_5 + contract_4_6 = 0
 R630: - 0.5 contract_6_6 - contract_4_6 + contract_4_7 = 0
 R631: - 0.5 contract_6_7 - contract_4_7 + contract_4_8 = 0
 R632: - 0.5 contract_6_8 - contract_4_8 + contract_4_9 = 0
 R633: - 0.5 contract_6_9 - contract_4_9 + contract_4_10 = 0
 R634: - contract_7_0 + contract_7_1 - 0.5 contract_9_0 = 0
 R635: - contract_7_1 + contract_7_2 - 0.5 contract_9_1 = 0
 R636: - contract_7_2 + contract_7_3 - 0.5 contract_9_2 = 0
 R637: - contract_7_3 + contract_7_4 - 0.5 contract_9_3 = 0
 R638: - contract_7_4 + contract_7_5 - 0.5 contract_9_4 = 0
 R639: - contract_7_5 + contract_7_6 - 0.5 contract_9_5 = 0
 R640: - contract_7_6 + contract_7_7 - 0.5 contract_9_6 = 0
 R641: - contract_7_7 + contract_7_8 - 0.5 contract_9_7 = 0
 R642: - contract_7_8 + contract_7_9 - 0.5 contract_9_8 = 0
 R643: - contract_7_9 + contract_7_10 - 0.5 contract_9_9 = 0
 R644: - contract_8_0 + contract_8_1 - 0.5 contract_10_0 = 0
 R645: - contract_8_1 + contract_8_2 - 0.5 contract_10_1 = 0
 R646: - contract_8_2 + contract_8_3 - 0.5 contract_10_2 = 0
 R647: - contract_8_3 + contract_8_4 - 0.5 contract_10_3 = 0
 R648: - contract_8_4 + contract_8_5 - 0.5 contract_10_4 = 0
 R649: - contract_8_5 + contract_8_6 - 0.5 contract_10_5 = 0
 R650: - contract_8_6 + contract_8_7 - 0.5 contract_10_6 = 0
 R651: - contract_8_7 + contract_8_8 - 0.5 contract_10_7 = 0
 R652: - contract_8_8 + contract_8_9 - 0.5 contract_10_8 = 0
 R653: - contract_8_9 + contract_8_10 - 0.5 contract_10_9 = 0
 R654: - 0.5 contract_11_0 - contract_9_0 + contract_9_1 = 0
 R655: - 0.5 contract_11_1 - contract_9_1 + contract_9_2 = 0
 R656: - 0.5 contract_11_2 - contract_9_2 + contract_9_3 = 0
 R657: - 0.5 contract_11_3 - contract_9_3 + contract_9_4 = 0
 R658: - 0.5 contract_11_4 - contract_9_4 + contract_9_5 = 0
 R659: - 0.5 contract_11_5 - contract_9_5 + contract_9_6 = 0
 R660: - 0.5 contract_11_6 - contract_9_6 + contract_9_7 = 0
 R661: - 0.5 contract_11_7 - contract_9_7 + contract_9_8 = 0
 R662: - 0.5 contract_11_8 - contract_9_8 + contract_9_9 = 0
 R663: - 0.5 contract_11_9 - contract_9_9 + contract_9_10 = 0
 R664: - 0.5 contract_12_0 - contract_10_0 + contract_10_1 = 0
 R665: - 0.5 contract_12_1 - contract_10_1 + contract_10_2 = 0
 R666: - 0.5 contract_12_2 - contract_10_2 + contract_10_3 = 0
 R667: - 0.5 contract_12_3 - contract_10_3 + contract_10_4 = 0
 R668: - 0.5 contract_12_4 - contract_10_4 + contract_10_5 = 0
 R669: - 0.5 contract_12_5 - contract_10_5 + contract_10_6 = 0
 R670: - 0.5 contract_12_6 - contract_10_6 + contract_10_7 = 0
 R671: - 0.5 contract_12_7 - contract_10_7 + contract_10_8 = 0
 R672: - 0.5 contract_12_8 - contract_10_8 + contract_10_9 = 0
 R673: - 0.5 contract_12_9 - contract_10_9 + contract_10_10 = 0
 R674: - contract_13_0 + contract_13_1 - 0.5 contract_15_0 = 0
 R675: - contract_13_1 + contract_13_2 - 0.5 contract_15_1 = 0
 R676: - contract_13_2 + contract_13_3 - 0.5 contract_15_2 = 0
 R677: - contract_13_3 + contract_13_4 - 0.5 contract_15_3 = 0
 R678: - contract_13_4 + contract_13_5 - 0.5 contract_15_4 = 0
 R679: - contract_13_5 + contract_13_6 - 0.5 contract_15_5 = 0
 R680: - contract_13_6 + contract_13_7 - 0.5 contract_15_6 = 0
 R681: - contract_13_7 + contract_13_8 - 0.5 contract_15_7 = 0
 R682: - contract_13_8 + contract_13_9 - 0.5 contract_15_8 = 0
 R683: - contract_13_9 + contract_13_10 - 0.5 contract_15_9 = 0
 R684: - contract_14_0 + contract_14_1 - 0.5 contract_16_0 = 0
 R685: - contract_14_1 + contract_14_2 - 0.5 contract_16_1 = 0
 R686: - contract_14_2 + contract_14_3 - 0.5 contract_16_2 = 0
 R687: - contract_14_3 + contract_14_4 - 0.5 contract_16_3 = 0
 R688: - contract_14_4 + contract_14_5 - 0.5 contract_16_4 = 0
 R689: - contract_14_5 + contract_14_6 - 0.5 contract_16_5 = 0
 R690: - contract_14_6 + contract_14_7 - 0.5 contract_16_6 = 0
 R691: - contract_14_7 + contract_14_8 - 0.5 contract_16_7 = 0
 R692: - contract_14_8 + contract_14_9 - 0.5 contract_16_8 = 0
 R693: - contract_14_9 + contract_14_10 - 0.5 contract_16_9 = 0
 R694: - contract_15_0 + contract_15_1 - 0.5 contract_17_0 = 0
 R695: - contract_15_1 + contract_15_2 - 0.5 contract_17_1 = 0
 R696: - contract_15_2 + contract_15_3 - 0.5 contract_17_2 = 0
 R697: - contract_15_3 + contract_15_4 - 0.5 contract_17_3 = 0
 R698: - contract_15_4 + contract_15_5 - 0.5 contract_17_4 = 0
 R699: - contract_15_5 + contract_15_6 - 0.5 contract_17_5 = 0
 R700: - contract_15_6 + contract_15_7 - 0.5 contract_17_6 = 0
 R701: - contract_15_7 + contract_15_8 - 0.5 contract_17_7 = 0
 R702: - contract_15_8 + contract_15_9 - 0.5 contract_17_8 = 0
 R703: - contract_15_9 + contract_15_10 - 0.5 contract_17_9 = 0
 R704: - contract_16_0 + contract_16_1 - 0.5 contract_18_0 = 0
 R705: - contract_16_1 + contract_16_2 - 0.5 contract_18_1 = 0
 R706: - contract_16_2 + contract_16_3 - 0.5 contract_18_2 = 0
 R707: - contract_16_3 + contract_16_4 - 0.5 contract_18_3 = 0
 R708: - contract_16_4 + contract_16_5 - 0.5 contract_18_4 = 0
 R709: - contract_16_5 + contract_16_6 - 0.5 contract_18_5 = 0
 R710: - contract_16_6 + contract_16_7 - 0.5 contract_18_6 = 0
 R711: - contract_16_7 + contract_16_8 - 0.5 contract_18_7 = 0
 R712: - contract_16_8 + contract_16_9 - 0.5 contract_18_8 = 0
 R713: - contract_16_9 + contract_16_10 - 0.5 contract_18_9 = 0
 R714: regions_0 = 1
 R715: regions_1 = 1
 R716: regions_2 = 1
 R717: regions_3 = 1
 R718: regions_4 = 1
 R719: regions_5 = 1
 R720: regions_6 = 1
 R721: regions_7 = 1
 R722: regions_8 = 1
 R723: regions_9 = 1
 R724: contract_13_0 - goal_x_0 <= 2
 R725: - contract_13_0 - goal_x_0 <= -2
 R726: contract_14_0 - goal_y_0 <= 64
 R727: - contract_14_0 - goal_y_0 <= -64
 R728: contract_13_1 - goal_x_1 <= 2
 R729: - contract_13_1 - goal_x_1 <= -2
 R730: contract_14_1 - goal_y_1 <= 64
 R731: - contract_14_1 - goal_y_1 <= -64
 R732: contract_13_2 - goal_x_2 <= 2
 R733: - contract_13_2 - goal_x_2 <= -2
 R734: contract_14_2 - goal_y_2 <= 64
 R735: - contract_14_2 - goal_y_2 <= -64
 R736: contract_13_3 - goal_x_3 <= 2
 R737: - contract_13_3 - goal_x_3 <= -2
 R738: contract_14_3 - goal_y_3 <= 64
 R739: - contract_14_3 - goal_y_3 <= -64
 R740: contract_13_4 - goal_x_4 <= 2
 R741: - contract_13_4 - goal_x_4 <= -2
 R742: contract_14_4 - goal_y_4 <= 64
 R743: - contract_14_4 - goal_y_4 <= -64
 R744: contract_13_5 - goal_x_5 <= 2
 R745: - contract_13_5 - goal_x_5 <= -2
 R746: contract_14_5 - goal_y_5 <= 64
 R747: - contract_14_5 - goal_y_5 <= -64
 R748: contract_13_6 - goal_x_6 <= 2
 R749: - contract_13_6 - goal_x_6 <= -2
 R750: contract_14_6 - goal_y_6 <= 64
 R751: - contract_14_6 - goal_y_6 <= -64
 R752: contract_13_7 - goal_x_7 <= 2
 R753: - contract_13_7 - goal_x_7 <= -2
 R754: contract_14_7 - goal_y_7 <= 64
 R755: - contract_14_7 - goal_y_7 <= -64
 R756: contract_13_8 - goal_x_8 <= 2
 R757: - contract_13_8 - goal_x_8 <= -2
 R758: contract_14_8 - goal_y_8 <= 64
 R759: - contract_14_8 - goal_y_8 <= -64
 R760: contract_13_9 - goal_x_9 <= 2
 R761: - contract_13_9 - goal_x_9 <= -2
 R762: contract_14_9 - goal_y_9 <= 64
 R763: - contract_14_9 - goal_y_9 <= -64
 R764: contract_17_0 - fuel_x_0 <= 0
 R765: - contract_17_0 - fuel_x_0 <= 0
 R766: contract_18_0 - fuel_y_0 <= 0
 R767: - contract_18_0 - fuel_y_0 <= 0
 R768: contract_17_1 - fuel_x_1 <= 0
 R769: - contract_17_1 - fuel_x_1 <= 0
 R770: contract_18_1 - fuel_y_1 <= 0
 R771: - contract_18_1 - fuel_y_1 <= 0
 R772: contract_17_2 - fuel_x_2 <= 0
 R773: - contract_17_2 - fuel_x_2 <= 0
 R774: contract_18_2 - fuel_y_2 <= 0
 R775: - contract_18_2 - fuel_y_2 <= 0
 R776: contract_17_3 - fuel_x_3 <= 0
 R777: - contract_17_3 - fuel_x_3 <= 0
 R778: contract_18_3 - fuel_y_3 <= 0
 R779: - contract_18_3 - fuel_y_3 <= 0
 R780: contract_17_4 - fuel_x_4 <= 0
 R781: - contract_17_4 - fuel_x_4 <= 0
 R782: contract_18_4 - fuel_y_4 <= 0
 R783: - contract_18_4 - fuel_y_4 <= 0
 R784: contract_17_5 - fuel_x_5 <= 0
 R785: - contract_17_5 - fuel_x_5 <= 0
 R786: contract_18_5 - fuel_y_5 <= 0
 R787: - contract_18_5 - fuel_y_5 <= 0
 R788: contract_17_6 - fuel_x_6 <= 0
 R789: - contract_17_6 - fuel_x_6 <= 0
 R790: contract_18_6 - fuel_y_6 <= 0
 R791: - contract_18_6 - fuel_y_6 <= 0
 R792: contract_17_7 - fuel_x_7 <= 0
 R793: - contract_17_7 - fuel_x_7 <= 0
 R794: contract_18_7 - fuel_y_7 <= 0
 R795: - contract_18_7 - fuel_y_7 <= 0
 R796: contract_17_8 - fuel_x_8 <= 0
 R797: - contract_17_8 - fuel_x_8 <= 0
 R798: contract_18_8 - fuel_y_8 <= 0
 R799: - contract_18_8 - fuel_y_8 <= 0
 R800: contract_17_9 - fuel_x_9 <= 0
 R801: - contract_17_9 - fuel_x_9 <= 0
 R802: contract_18_9 - fuel_y_9 <= 0
 R803: - contract_18_9 - fuel_y_9 <= 0
 state_0_0: contract_1_0 = -46.86377563584074
 state_0_1: contract_2_0 = 2
 state_0_2: contract_3_0 = -3.000000000154903
 state_0_3: contract_4_0 = -1.18939e-13
 state_1_0: contract_7_0 = 4.785552359168568
 state_1_1: contract_8_0 = -1.294889377109271
 state_1_2: contract_9_0 = 7.15256e-07
 state_1_3: contract_10_0 = 1.90582e-21
 state_2_0: contract_13_0 = 1.785553099988374
 state_2_1: contract_14_0 = -5.404105182876958
 state_2_2: contract_15_0 = -2.55641e-08
 state_2_3: contract_16_0 = 1.19209e-07
Bounds
 -5 <= contract_5_0 <= 5
 -5 <= contract_5_1 <= 5
 -5 <= contract_5_2 <= 5
 -5 <= contract_5_3 <= 5
 -5 <= contract_5_4 <= 5
 -5 <= contract_5_5 <= 5
 -5 <= contract_5_6 <= 5
 -5 <= contract_5_7 <= 5
 -5 <= contract_5_8 <= 5
 -5 <= contract_5_9 <= 5
 -5 <= contract_5_10 <= 5
 -5 <= contract_6_0 <= 5
 -5 <= contract_6_1 <= 5
 -5 <= contract_6_2 <= 5
 -5 <= contract_6_3 <= 5
 -5 <= contract_6_4 <= 5
 -5 <= contract_6_5 <= 5
 -5 <= contract_6_6 <= 5
 -5 <= contract_6_7 <= 5
 -5 <= contract_6_8 <= 5
 -5 <= contract_6_9 <= 5
 -5 <= contract_6_10 <= 5
 -5 <= contract_11_0 <= 5
 -5 <= contract_11_1 <= 5
 -5 <= contract_11_2 <= 5
 -5 <= contract_11_3 <= 5
 -5 <= contract_11_4 <= 5
 -5 <= contract_11_5 <= 5
 -5 <= contract_11_6 <= 5
 -5 <= contract_11_7 <= 5
 -5 <= contract_11_8 <= 5
 -5 <= contract_11_9 <= 5
 -5 <= contract_11_10 <= 5
 -5 <= contract_12_0 <= 5
 -5 <= contract_12_1 <= 5
 -5 <= contract_12_2 <= 5
 -5 <= contract_12_3 <= 5
 -5 <= contract_12_4 <= 5
 -5 <= contract_12_5 <= 5
 -5 <= contract_12_6 <= 5
 -5 <= contract_12_7 <= 5
 -5 <= contract_12_8 <= 5
 -5 <= contract_12_9 <= 5
 -5 <= contract_12_10 <= 5
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
 -10000 <= contract_1_7 <= 10000
 -10000 <= contract_13_7 <= 10000
 -10000 <= contract_1_8 <= 10000
 -10000 <= contract_13_8 <= 10000
 -10000 <= contract_1_9 <= 10000
 -10000 <= contract_13_9 <= 10000
 -10000 <= contract_1_10 <= 10000
 -10000 <= contract_13_10 <= 10000
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
 -10000 <= contract_2_7 <= 10000
 -10000 <= contract_14_7 <= 10000
 -10000 <= contract_2_8 <= 10000
 -10000 <= contract_14_8 <= 10000
 -10000 <= contract_2_9 <= 10000
 -10000 <= contract_14_9 <= 10000
 -10000 <= contract_2_10 <= 10000
 -10000 <= contract_14_10 <= 10000
 -10000 <= contract_7_0 <= 10000
 -10000 <= contract_7_1 <= 10000
 -10000 <= contract_7_2 <= 10000
 -10000 <= contract_7_3 <= 10000
 -10000 <= contract_7_4 <= 10000
 -10000 <= contract_7_5 <= 10000
 -10000 <= contract_7_6 <= 10000
 -10000 <= contract_7_7 <= 10000
 -10000 <= contract_7_8 <= 10000
 -10000 <= contract_7_9 <= 10000
 -10000 <= contract_7_10 <= 10000
 -10000 <= contract_8_0 <= 10000
 -10000 <= contract_8_1 <= 10000
 -10000 <= contract_8_2 <= 10000
 -10000 <= contract_8_3 <= 10000
 -10000 <= contract_8_4 <= 10000
 -10000 <= contract_8_5 <= 10000
 -10000 <= contract_8_6 <= 10000
 -10000 <= contract_8_7 <= 10000
 -10000 <= contract_8_8 <= 10000
 -10000 <= contract_8_9 <= 10000
 -10000 <= contract_8_10 <= 10000
 -3 <= contract_3_0 <= 3
 -3 <= contract_3_1 <= 3
 -3 <= contract_3_2 <= 3
 -3 <= contract_3_3 <= 3
 -3 <= contract_3_4 <= 3
 -3 <= contract_3_5 <= 3
 -3 <= contract_3_6 <= 3
 -3 <= contract_3_7 <= 3
 -3 <= contract_3_8 <= 3
 -3 <= contract_3_9 <= 3
 -3 <= contract_4_0 <= 3
 -3 <= contract_4_1 <= 3
 -3 <= contract_4_2 <= 3
 -3 <= contract_4_3 <= 3
 -3 <= contract_4_4 <= 3
 -3 <= contract_4_5 <= 3
 -3 <= contract_4_6 <= 3
 -3 <= contract_4_7 <= 3
 -3 <= contract_4_8 <= 3
 -3 <= contract_4_9 <= 3
 -3 <= contract_3_10 <= 3
 -3 <= contract_4_10 <= 3
 -3 <= contract_9_0 <= 3
 -3 <= contract_9_1 <= 3
 -3 <= contract_9_2 <= 3
 -3 <= contract_9_3 <= 3
 -3 <= contract_9_4 <= 3
 -3 <= contract_9_5 <= 3
 -3 <= contract_9_6 <= 3
 -3 <= contract_9_7 <= 3
 -3 <= contract_9_8 <= 3
 -3 <= contract_9_9 <= 3
 -3 <= contract_10_0 <= 3
 -3 <= contract_10_1 <= 3
 -3 <= contract_10_2 <= 3
 -3 <= contract_10_3 <= 3
 -3 <= contract_10_4 <= 3
 -3 <= contract_10_5 <= 3
 -3 <= contract_10_6 <= 3
 -3 <= contract_10_7 <= 3
 -3 <= contract_10_8 <= 3
 -3 <= contract_10_9 <= 3
 -3 <= contract_9_10 <= 3
 -3 <= contract_10_10 <= 3
 -3 <= contract_15_0 <= 3
 -3 <= contract_15_1 <= 3
 -3 <= contract_15_2 <= 3
 -3 <= contract_15_3 <= 3
 -3 <= contract_15_4 <= 3
 -3 <= contract_15_5 <= 3
 -3 <= contract_15_6 <= 3
 -3 <= contract_15_7 <= 3
 -3 <= contract_15_8 <= 3
 -3 <= contract_15_9 <= 3
 -3 <= contract_16_0 <= 3
 -3 <= contract_16_1 <= 3
 -3 <= contract_16_2 <= 3
 -3 <= contract_16_3 <= 3
 -3 <= contract_16_4 <= 3
 -3 <= contract_16_5 <= 3
 -3 <= contract_16_6 <= 3
 -3 <= contract_16_7 <= 3
 -3 <= contract_16_8 <= 3
 -3 <= contract_16_9 <= 3
 -5 <= contract_17_0 <= 5
 -5 <= contract_17_1 <= 5
 -5 <= contract_17_2 <= 5
 -5 <= contract_17_3 <= 5
 -5 <= contract_17_4 <= 5
 -5 <= contract_17_5 <= 5
 -5 <= contract_17_6 <= 5
 -5 <= contract_17_7 <= 5
 -5 <= contract_17_8 <= 5
 -3 <= contract_15_10 <= 3
 -5 <= contract_17_9 <= 5
 -5 <= contract_18_0 <= 5
 -5 <= contract_18_1 <= 5
 -5 <= contract_18_2 <= 5
 -5 <= contract_18_3 <= 5
 -5 <= contract_18_4 <= 5
 -5 <= contract_18_5 <= 5
 -5 <= contract_18_6 <= 5
 -5 <= contract_18_7 <= 5
 -5 <= contract_18_8 <= 5
 -3 <= contract_16_10 <= 3
 -5 <= contract_18_9 <= 5
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
 -10000 <= goal_x_6 <= 10000
 -10000 <= goal_y_6 <= 10000
 -10000 <= goal_x_7 <= 10000
 -10000 <= goal_y_7 <= 10000
 -10000 <= goal_x_8 <= 10000
 -10000 <= goal_y_8 <= 10000
 -10000 <= goal_x_9 <= 10000
 -10000 <= goal_y_9 <= 10000
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
 fuel_x_6 <= 5
 fuel_y_6 <= 5
 fuel_x_7 <= 5
 fuel_y_7 <= 5
 fuel_x_8 <= 5
 fuel_y_8 <= 5
 fuel_x_9 <= 5
 fuel_y_9 <= 5
Binaries
 node_0_0 node_1_0 node_1_1 node_1_2 node_1_3 node_1_4 node_1_5 node_1_6
 node_1_7 node_1_8 node_1_9 node_1_10 node_2_0 node_2_1 node_2_2 node_2_3
 node_2_4 node_2_5 node_2_6 node_2_7 node_2_8 node_2_9 node_2_10 node_3_0
 node_3_1 node_3_2 node_3_3 node_3_4 node_3_5 node_3_6 node_3_7 node_3_8
 node_3_9 node_3_10 node_4_0 node_4_1 node_4_2 node_4_3 node_4_4 node_4_5
 node_4_6 node_4_7 node_4_8 node_4_9 node_4_10 node_5_0 node_5_1 node_5_2
 node_5_3 node_5_4 node_5_5 node_5_6 node_5_7 node_5_8 node_5_9 node_5_10
 node_6_0 node_6_1 node_6_2 node_6_3 node_6_4 node_6_5 node_6_6 node_6_7
 node_6_8 node_6_9 node_6_10 node_7_0 node_7_1 node_7_2 node_7_3 node_7_4
 node_7_5 node_7_6 node_7_7 node_7_8 node_7_9 node_7_10 node_8_0 node_8_1
 node_8_2 node_8_3 node_8_4 node_8_5 node_8_6 node_8_7 node_8_8 node_8_9
 node_8_10 node_9_0 node_9_1 node_9_2 node_9_3 node_9_4 node_9_5 node_9_6
 node_9_7 node_9_8 node_9_9 node_9_10 node_10_0 node_11_0 node_12_0
 node_12_1 node_12_2 node_12_3 node_12_4 node_12_5 node_12_6 node_12_7
 node_12_8 node_12_9 node_12_10 node_13_0 node_13_1 node_13_2 node_13_3
 node_13_4 node_13_5 node_13_6 node_13_7 node_13_8 node_13_9 node_13_10
 node_14_0 node_14_1 node_14_2 node_14_3 node_14_4 node_14_5 node_14_6
 node_14_7 node_14_8 node_14_9 node_14_10 node_15_0 node_15_1 node_15_2
 node_15_3 node_15_4 node_15_5 node_15_6 node_15_7 node_15_8 node_15_9
 node_15_10 node_16_0 node_16_1 node_16_2 node_16_3 node_16_4 node_16_5
 node_16_6 node_16_7 node_16_8 node_16_9 node_16_10 node_17_0 node_18_0
 node_18_1 node_18_2 node_18_3 node_18_4 node_18_5 node_18_6 node_18_7
 node_18_8 node_18_9 node_18_10 node_19_0 node_19_1 node_19_2 node_19_3
 node_19_4 node_19_5 node_19_6 node_19_7 node_19_8 node_19_9 node_19_10
 node_20_0 node_20_1 node_20_2 node_20_3 node_20_4 node_20_5 node_20_6
 node_20_7 node_20_8 node_20_9 node_20_10 node_21_0 node_21_1 node_21_2
 node_21_3 node_21_4 node_21_5 node_21_6 node_21_7 node_21_8 node_21_9
 node_21_10 node_22_0 node_22_1 node_22_2 node_22_3 node_22_4 node_22_5
 node_22_6 node_22_7 node_22_8 node_22_9 node_22_10 node_23_0 node_24_0
 node_25_0 node_26_0 node_27_0 node_28_0 node_29_0 node_30_1 node_31_1
 node_32_1 node_33_1 node_34_1 node_35_0 node_36_2 node_37_2 node_38_2
 node_39_2 node_40_2 node_41_0 node_42_3 node_43_3 node_44_3 node_45_3
 node_46_3 node_47_0 node_48_4 node_49_4 node_50_4 node_51_4 node_52_4
 node_53_0 node_54_5 node_55_5 node_56_5 node_57_5 node_58_5 node_59_0
 node_60_6 node_61_6 node_62_6 node_63_6 node_64_6 node_65_0 node_66_7
 node_67_7 node_68_7 node_69_7 node_70_7 node_71_0 node_72_8 node_73_8
 node_74_8 node_75_8 node_76_8 node_77_0 node_78_9 node_79_9 node_80_9
 node_81_9 node_82_9 node_83_0 node_84_0 node_85_0 node_86_0 node_87_0
 node_88_0 node_89_0 node_90_1 node_91_1 node_92_1 node_93_1 node_94_1
 node_95_0 node_96_2 node_97_2 node_98_2 node_99_2 node_100_2 node_101_0
 node_102_3 node_103_3 node_104_3 node_105_3 node_106_3 node_107_0
 node_108_4 node_109_4 node_110_4 node_111_4 node_112_4 node_113_0
 node_114_5 node_115_5 node_116_5 node_117_5 node_118_5 node_119_0
 node_120_6 node_121_6 node_122_6 node_123_6 node_124_6 node_125_0
 node_126_7 node_127_7 node_128_7 node_129_7 node_130_7 node_131_0
 node_132_8 node_133_8 node_134_8 node_135_8 node_136_8 node_137_0
 node_138_9 node_139_9 node_140_9 node_141_9 node_142_9 node_143_0
 node_144_0 node_145_0 node_146_0 node_147_0 node_148_0 node_149_0
 node_150_1 node_151_1 node_152_1 node_153_1 node_154_1 node_155_0
 node_156_2 node_157_2 node_158_2 node_159_2 node_160_2 node_161_0
 node_162_3 node_163_3 node_164_3 node_165_3 node_166_3 node_167_0
 node_168_4 node_169_4 node_170_4 node_171_4 node_172_4 node_173_0
 node_174_5 node_175_5 node_176_5 node_177_5 node_178_5 node_179_0
 node_180_6 node_181_6 node_182_6 node_183_6 node_184_6 node_185_0
 node_186_7 node_187_7 node_188_7 node_189_7 node_190_7 node_191_0
 node_192_8 node_193_8 node_194_8 node_195_8 node_196_8 node_197_0
 node_198_9 node_199_9 node_200_9 node_201_9 node_202_9 regions_0 regions_1
 regions_2 regions_3 regions_4 regions_5 regions_6 regions_7 regions_8
 regions_9
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
 GC7: node_1_7 = AND ( node_2_7 , node_3_7 , node_4_7 , node_5_7 ,
   node_6_7 , node_7_7 , node_8_7 , node_9_7 )
 GC8: node_1_8 = AND ( node_2_8 , node_3_8 , node_4_8 , node_5_8 ,
   node_6_8 , node_7_8 , node_8_8 , node_9_8 )
 GC9: node_1_9 = AND ( node_2_9 , node_3_9 , node_4_9 , node_5_9 ,
   node_6_9 , node_7_9 , node_8_9 , node_9_9 )
 GC10: node_1_10 = AND ( node_2_10 , node_3_10 , node_4_10 , node_5_10 ,
   node_6_10 , node_7_10 , node_8_10 , node_9_10 )
 GC11: node_0_0 = AND ( node_1_0 , node_1_1 , node_1_2 , node_1_3 ,
   node_1_4 , node_1_5 , node_1_6 , node_1_7 , node_1_8 , node_1_9 ,
   node_1_10 )
 GC12: node_12_0 = OR ( node_13_0 , node_14_0 , node_15_0 , node_16_0 )
 GC13: node_12_1 = OR ( node_13_1 , node_14_1 , node_15_1 , node_16_1 )
 GC14: node_12_2 = OR ( node_13_2 , node_14_2 , node_15_2 , node_16_2 )
 GC15: node_12_3 = OR ( node_13_3 , node_14_3 , node_15_3 , node_16_3 )
 GC16: node_12_4 = OR ( node_13_4 , node_14_4 , node_15_4 , node_16_4 )
 GC17: node_12_5 = OR ( node_13_5 , node_14_5 , node_15_5 , node_16_5 )
 GC18: node_12_6 = OR ( node_13_6 , node_14_6 , node_15_6 , node_16_6 )
 GC19: node_12_7 = OR ( node_13_7 , node_14_7 , node_15_7 , node_16_7 )
 GC20: node_12_8 = OR ( node_13_8 , node_14_8 , node_15_8 , node_16_8 )
 GC21: node_12_9 = OR ( node_13_9 , node_14_9 , node_15_9 , node_16_9 )
 GC22: node_12_10 = OR ( node_13_10 , node_14_10 , node_15_10 , node_16_10 )
 GC23: node_11_0 = AND ( node_12_0 , node_12_1 , node_12_2 , node_12_3 ,
   node_12_4 , node_12_5 , node_12_6 , node_12_7 , node_12_8 , node_12_9 ,
   node_12_10 )
 GC24: node_18_0 = OR ( node_19_0 , node_20_0 , node_21_0 , node_22_0 )
 GC25: node_18_1 = OR ( node_19_1 , node_20_1 , node_21_1 , node_22_1 )
 GC26: node_18_2 = OR ( node_19_2 , node_20_2 , node_21_2 , node_22_2 )
 GC27: node_18_3 = OR ( node_19_3 , node_20_3 , node_21_3 , node_22_3 )
 GC28: node_18_4 = OR ( node_19_4 , node_20_4 , node_21_4 , node_22_4 )
 GC29: node_18_5 = OR ( node_19_5 , node_20_5 , node_21_5 , node_22_5 )
 GC30: node_18_6 = OR ( node_19_6 , node_20_6 , node_21_6 , node_22_6 )
 GC31: node_18_7 = OR ( node_19_7 , node_20_7 , node_21_7 , node_22_7 )
 GC32: node_18_8 = OR ( node_19_8 , node_20_8 , node_21_8 , node_22_8 )
 GC33: node_18_9 = OR ( node_19_9 , node_20_9 , node_21_9 , node_22_9 )
 GC34: node_18_10 = OR ( node_19_10 , node_20_10 , node_21_10 , node_22_10 )
 GC35: node_17_0 = AND ( node_18_0 , node_18_1 , node_18_2 , node_18_3 ,
   node_18_4 , node_18_5 , node_18_6 , node_18_7 , node_18_8 , node_18_9 ,
   node_18_10 )
 GC36: node_10_0 = AND ( node_11_0 , node_17_0 )
 GC37: node_24_0 = AND ( node_25_0 , node_26_0 , node_27_0 , node_28_0 )
 GC38: node_23_0 = AND ( node_24_0 )
 GC39: node_30_1 = AND ( node_31_1 , node_32_1 , node_33_1 , node_34_1 )
 GC40: node_29_0 = AND ( node_30_1 )
 GC41: node_36_2 = AND ( node_37_2 , node_38_2 , node_39_2 , node_40_2 )
 GC42: node_35_0 = AND ( node_36_2 )
 GC43: node_42_3 = AND ( node_43_3 , node_44_3 , node_45_3 , node_46_3 )
 GC44: node_41_0 = AND ( node_42_3 )
 GC45: node_48_4 = AND ( node_49_4 , node_50_4 , node_51_4 , node_52_4 )
 GC46: node_47_0 = AND ( node_48_4 )
 GC47: node_54_5 = AND ( node_55_5 , node_56_5 , node_57_5 , node_58_5 )
 GC48: node_53_0 = AND ( node_54_5 )
 GC49: node_60_6 = AND ( node_61_6 , node_62_6 , node_63_6 , node_64_6 )
 GC50: node_59_0 = AND ( node_60_6 )
 GC51: node_66_7 = AND ( node_67_7 , node_68_7 , node_69_7 , node_70_7 )
 GC52: node_65_0 = AND ( node_66_7 )
 GC53: node_72_8 = AND ( node_73_8 , node_74_8 , node_75_8 , node_76_8 )
 GC54: node_71_0 = AND ( node_72_8 )
 GC55: node_78_9 = AND ( node_79_9 , node_80_9 , node_81_9 , node_82_9 )
 GC56: node_77_0 = AND ( node_78_9 )
 GC57: node_84_0 = AND ( node_85_0 , node_86_0 , node_87_0 , node_88_0 )
 GC58: node_83_0 = AND ( node_84_0 )
 GC59: node_90_1 = AND ( node_91_1 , node_92_1 , node_93_1 , node_94_1 )
 GC60: node_89_0 = AND ( node_90_1 )
 GC61: node_96_2 = AND ( node_97_2 , node_98_2 , node_99_2 , node_100_2 )
 GC62: node_95_0 = AND ( node_96_2 )
 GC63: node_102_3 = AND ( node_103_3 , node_104_3 , node_105_3 , node_106_3 )
 GC64: node_101_0 = AND ( node_102_3 )
 GC65: node_108_4 = AND ( node_109_4 , node_110_4 , node_111_4 , node_112_4 )
 GC66: node_107_0 = AND ( node_108_4 )
 GC67: node_114_5 = AND ( node_115_5 , node_116_5 , node_117_5 , node_118_5 )
 GC68: node_113_0 = AND ( node_114_5 )
 GC69: node_120_6 = AND ( node_121_6 , node_122_6 , node_123_6 , node_124_6 )
 GC70: node_119_0 = AND ( node_120_6 )
 GC71: node_126_7 = AND ( node_127_7 , node_128_7 , node_129_7 , node_130_7 )
 GC72: node_125_0 = AND ( node_126_7 )
 GC73: node_132_8 = AND ( node_133_8 , node_134_8 , node_135_8 , node_136_8 )
 GC74: node_131_0 = AND ( node_132_8 )
 GC75: node_138_9 = AND ( node_139_9 , node_140_9 , node_141_9 , node_142_9 )
 GC76: node_137_0 = AND ( node_138_9 )
 GC77: node_144_0 = AND ( node_145_0 , node_146_0 , node_147_0 , node_148_0 )
 GC78: node_143_0 = AND ( node_144_0 )
 GC79: node_150_1 = AND ( node_151_1 , node_152_1 , node_153_1 , node_154_1 )
 GC80: node_149_0 = AND ( node_150_1 )
 GC81: node_156_2 = AND ( node_157_2 , node_158_2 , node_159_2 , node_160_2 )
 GC82: node_155_0 = AND ( node_156_2 )
 GC83: node_162_3 = AND ( node_163_3 , node_164_3 , node_165_3 , node_166_3 )
 GC84: node_161_0 = AND ( node_162_3 )
 GC85: node_168_4 = AND ( node_169_4 , node_170_4 , node_171_4 , node_172_4 )
 GC86: node_167_0 = AND ( node_168_4 )
 GC87: node_174_5 = AND ( node_175_5 , node_176_5 , node_177_5 , node_178_5 )
 GC88: node_173_0 = AND ( node_174_5 )
 GC89: node_180_6 = AND ( node_181_6 , node_182_6 , node_183_6 , node_184_6 )
 GC90: node_179_0 = AND ( node_180_6 )
 GC91: node_186_7 = AND ( node_187_7 , node_188_7 , node_189_7 , node_190_7 )
 GC92: node_185_0 = AND ( node_186_7 )
 GC93: node_192_8 = AND ( node_193_8 , node_194_8 , node_195_8 , node_196_8 )
 GC94: node_191_0 = AND ( node_192_8 )
 GC95: node_198_9 = AND ( node_199_9 , node_200_9 , node_201_9 , node_202_9 )
 GC96: node_197_0 = AND ( node_198_9 )
 GC97: regions_0 = OR ( node_23_0 , node_83_0 , node_143_0 )
 GC98: regions_1 = OR ( node_29_0 , node_89_0 , node_149_0 )
 GC99: regions_2 = OR ( node_35_0 , node_95_0 , node_155_0 )
 GC100: regions_3 = OR ( node_41_0 , node_101_0 , node_161_0 )
 GC101: regions_4 = OR ( node_47_0 , node_107_0 , node_167_0 )
 GC102: regions_5 = OR ( node_53_0 , node_113_0 , node_173_0 )
 GC103: regions_6 = OR ( node_59_0 , node_119_0 , node_179_0 )
 GC104: regions_7 = OR ( node_65_0 , node_125_0 , node_185_0 )
 GC105: regions_8 = OR ( node_71_0 , node_131_0 , node_191_0 )
 GC106: regions_9 = OR ( node_77_0 , node_137_0 , node_197_0 )
End
