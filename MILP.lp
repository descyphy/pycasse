\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
  0 node_1_0 + 0 node_1_1 + 0 node_1_2 + 0 node_1_3 + 0 node_1_4
   + 0 node_1_5 + 0 node_1_6 + 0 node_11_0 + 0 node_12_0 + 0 node_12_1
   + 0 node_12_2 + 0 node_12_3 + 0 node_12_4 + 0 node_12_5 + 0 node_12_6
   + 0 node_17_0 + 0 node_18_0 + 0 node_18_1 + 0 node_18_2 + 0 node_18_3
   + 0 node_18_4 + 0 node_18_5 + 0 node_18_6 + 0 node_59_0 + 0 node_60_0
   + 0 node_65_0 + 0 node_66_1 + 0 node_71_0 + 0 node_72_2 + 0 node_77_0
   + 0 node_78_3 + 0 node_83_0 + 0 node_84_4 + 0 node_89_0 + 0 node_90_5
   + 0 node_95_0 + 0 node_96_0 + 0 node_101_0 + 0 node_102_1 + 0 node_107_0
   + 0 node_108_2 + 0 node_113_0 + 0 node_114_3 + 0 node_119_0
   + 0 node_120_4 + 0 node_125_0 + 0 node_126_5 + 0 node_131_0
   + 0 node_132_0 + 0 node_137_0 + 0 node_138_1 + 0 node_143_0
   + 0 node_144_2 + 0 node_149_0 + 0 node_150_3 + 0 node_155_0
   + 0 node_156_4 + 0 node_161_0 + 0 node_162_5 + 0 node_167_0
   + 0 node_168_0 + 0 node_173_0 + 0 node_174_1 + 0 node_179_0
   + 0 node_180_2 + 0 node_185_0 + 0 node_186_3 + 0 node_191_0
   + 0 node_192_4 + 0 node_197_0 + 0 node_198_5 + 0 node_203_0
   + 0 node_204_0 + 0 node_209_0 + 0 node_210_1 + 0 node_215_0
   + 0 node_216_2 + 0 node_221_0 + 0 node_222_3 + 0 node_227_0
   + 0 node_228_4 + 0 node_233_0 + 0 node_234_5 + 0 node_239_0
   + 0 node_240_0 + 0 node_245_0 + 0 node_246_1 + 0 node_251_0
   + 0 node_252_2 + 0 node_257_0 + 0 node_258_3 + 0 node_263_0
   + 0 node_264_4 + 0 node_269_0 + 0 node_270_5 + 0 node_275_0
   + 0 node_276_0 + 0 node_281_0 + 0 node_282_1 + 0 node_287_0
   + 0 node_288_2 + 0 node_293_0 + 0 node_294_3 + 0 node_299_0
   + 0 node_300_4 + 0 node_305_0 + 0 node_306_5 + 10000 goal_x_0
   + 10000 goal_y_0 + 10000 goal_x_1 + 10000 goal_y_1 + 10000 goal_x_2
   + 10000 goal_y_2 + 10000 goal_x_3 + 10000 goal_y_3 + 10000 goal_x_4
   + 10000 goal_y_4 + 10000 goal_x_5 + 10000 goal_y_5 + fuel_x_0 + fuel_y_0
   + fuel_x_1 + fuel_y_1 + fuel_x_2 + fuel_y_2 + fuel_x_3 + fuel_y_3
   + fuel_x_4 + fuel_y_4 + fuel_x_5 + fuel_y_5
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
 R113: 10000 node_13_0 + contract_1_0 - contract_13_0 <= 9994
 R114: 10000 node_13_0 + contract_1_0 - contract_13_0 >= -6
 R115: 10000 node_13_1 + contract_1_1 - contract_13_1 <= 9994
 R116: 10000 node_13_1 + contract_1_1 - contract_13_1 >= -6
 R117: 10000 node_13_2 + contract_1_2 - contract_13_2 <= 9994
 R118: 10000 node_13_2 + contract_1_2 - contract_13_2 >= -6
 R119: 10000 node_13_3 + contract_1_3 - contract_13_3 <= 9994
 R120: 10000 node_13_3 + contract_1_3 - contract_13_3 >= -6
 R121: 10000 node_13_4 + contract_1_4 - contract_13_4 <= 9994
 R122: 10000 node_13_4 + contract_1_4 - contract_13_4 >= -6
 R123: 10000 node_13_5 + contract_1_5 - contract_13_5 <= 9994
 R124: 10000 node_13_5 + contract_1_5 - contract_13_5 >= -6
 R125: 10000 node_13_6 + contract_1_6 - contract_13_6 <= 9994
 R126: 10000 node_13_6 + contract_1_6 - contract_13_6 >= -6
 R127: - contract_1_0 + contract_13_0 + 10000 node_14_0 <= 9994
 R128: - contract_1_0 + contract_13_0 + 10000 node_14_0 >= -6
 R129: - contract_1_1 + contract_13_1 + 10000 node_14_1 <= 9994
 R130: - contract_1_1 + contract_13_1 + 10000 node_14_1 >= -6
 R131: - contract_1_2 + contract_13_2 + 10000 node_14_2 <= 9994
 R132: - contract_1_2 + contract_13_2 + 10000 node_14_2 >= -6
 R133: - contract_1_3 + contract_13_3 + 10000 node_14_3 <= 9994
 R134: - contract_1_3 + contract_13_3 + 10000 node_14_3 >= -6
 R135: - contract_1_4 + contract_13_4 + 10000 node_14_4 <= 9994
 R136: - contract_1_4 + contract_13_4 + 10000 node_14_4 >= -6
 R137: - contract_1_5 + contract_13_5 + 10000 node_14_5 <= 9994
 R138: - contract_1_5 + contract_13_5 + 10000 node_14_5 >= -6
 R139: - contract_1_6 + contract_13_6 + 10000 node_14_6 <= 9994
 R140: - contract_1_6 + contract_13_6 + 10000 node_14_6 >= -6
 R141: 10000 node_15_0 + contract_2_0 - contract_14_0 <= 9997
 R142: 10000 node_15_0 + contract_2_0 - contract_14_0 >= -3
 R143: 10000 node_15_1 + contract_2_1 - contract_14_1 <= 9997
 R144: 10000 node_15_1 + contract_2_1 - contract_14_1 >= -3
 R145: 10000 node_15_2 + contract_2_2 - contract_14_2 <= 9997
 R146: 10000 node_15_2 + contract_2_2 - contract_14_2 >= -3
 R147: 10000 node_15_3 + contract_2_3 - contract_14_3 <= 9997
 R148: 10000 node_15_3 + contract_2_3 - contract_14_3 >= -3
 R149: 10000 node_15_4 + contract_2_4 - contract_14_4 <= 9997
 R150: 10000 node_15_4 + contract_2_4 - contract_14_4 >= -3
 R151: 10000 node_15_5 + contract_2_5 - contract_14_5 <= 9997
 R152: 10000 node_15_5 + contract_2_5 - contract_14_5 >= -3
 R153: 10000 node_15_6 + contract_2_6 - contract_14_6 <= 9997
 R154: 10000 node_15_6 + contract_2_6 - contract_14_6 >= -3
 R155: - contract_2_0 + contract_14_0 + 10000 node_16_0 <= 9997
 R156: - contract_2_0 + contract_14_0 + 10000 node_16_0 >= -3
 R157: - contract_2_1 + contract_14_1 + 10000 node_16_1 <= 9997
 R158: - contract_2_1 + contract_14_1 + 10000 node_16_1 >= -3
 R159: - contract_2_2 + contract_14_2 + 10000 node_16_2 <= 9997
 R160: - contract_2_2 + contract_14_2 + 10000 node_16_2 >= -3
 R161: - contract_2_3 + contract_14_3 + 10000 node_16_3 <= 9997
 R162: - contract_2_3 + contract_14_3 + 10000 node_16_3 >= -3
 R163: - contract_2_4 + contract_14_4 + 10000 node_16_4 <= 9997
 R164: - contract_2_4 + contract_14_4 + 10000 node_16_4 >= -3
 R165: - contract_2_5 + contract_14_5 + 10000 node_16_5 <= 9997
 R166: - contract_2_5 + contract_14_5 + 10000 node_16_5 >= -3
 R167: - contract_2_6 + contract_14_6 + 10000 node_16_6 <= 9997
 R168: - contract_2_6 + contract_14_6 + 10000 node_16_6 >= -3
 R169: - contract_13_0 + 10000 node_19_0 + contract_7_0 <= 9994
 R170: - contract_13_0 + 10000 node_19_0 + contract_7_0 >= -6
 R171: - contract_13_1 + 10000 node_19_1 + contract_7_1 <= 9994
 R172: - contract_13_1 + 10000 node_19_1 + contract_7_1 >= -6
 R173: - contract_13_2 + 10000 node_19_2 + contract_7_2 <= 9994
 R174: - contract_13_2 + 10000 node_19_2 + contract_7_2 >= -6
 R175: - contract_13_3 + 10000 node_19_3 + contract_7_3 <= 9994
 R176: - contract_13_3 + 10000 node_19_3 + contract_7_3 >= -6
 R177: - contract_13_4 + 10000 node_19_4 + contract_7_4 <= 9994
 R178: - contract_13_4 + 10000 node_19_4 + contract_7_4 >= -6
 R179: - contract_13_5 + 10000 node_19_5 + contract_7_5 <= 9994
 R180: - contract_13_5 + 10000 node_19_5 + contract_7_5 >= -6
 R181: - contract_13_6 + 10000 node_19_6 + contract_7_6 <= 9994
 R182: - contract_13_6 + 10000 node_19_6 + contract_7_6 >= -6
 R183: contract_13_0 - contract_7_0 + 10000 node_20_0 <= 9994
 R184: contract_13_0 - contract_7_0 + 10000 node_20_0 >= -6
 R185: contract_13_1 - contract_7_1 + 10000 node_20_1 <= 9994
 R186: contract_13_1 - contract_7_1 + 10000 node_20_1 >= -6
 R187: contract_13_2 - contract_7_2 + 10000 node_20_2 <= 9994
 R188: contract_13_2 - contract_7_2 + 10000 node_20_2 >= -6
 R189: contract_13_3 - contract_7_3 + 10000 node_20_3 <= 9994
 R190: contract_13_3 - contract_7_3 + 10000 node_20_3 >= -6
 R191: contract_13_4 - contract_7_4 + 10000 node_20_4 <= 9994
 R192: contract_13_4 - contract_7_4 + 10000 node_20_4 >= -6
 R193: contract_13_5 - contract_7_5 + 10000 node_20_5 <= 9994
 R194: contract_13_5 - contract_7_5 + 10000 node_20_5 >= -6
 R195: contract_13_6 - contract_7_6 + 10000 node_20_6 <= 9994
 R196: contract_13_6 - contract_7_6 + 10000 node_20_6 >= -6
 R197: - contract_14_0 + 10000 node_21_0 + contract_8_0 <= 9997
 R198: - contract_14_0 + 10000 node_21_0 + contract_8_0 >= -3
 R199: - contract_14_1 + 10000 node_21_1 + contract_8_1 <= 9997
 R200: - contract_14_1 + 10000 node_21_1 + contract_8_1 >= -3
 R201: - contract_14_2 + 10000 node_21_2 + contract_8_2 <= 9997
 R202: - contract_14_2 + 10000 node_21_2 + contract_8_2 >= -3
 R203: - contract_14_3 + 10000 node_21_3 + contract_8_3 <= 9997
 R204: - contract_14_3 + 10000 node_21_3 + contract_8_3 >= -3
 R205: - contract_14_4 + 10000 node_21_4 + contract_8_4 <= 9997
 R206: - contract_14_4 + 10000 node_21_4 + contract_8_4 >= -3
 R207: - contract_14_5 + 10000 node_21_5 + contract_8_5 <= 9997
 R208: - contract_14_5 + 10000 node_21_5 + contract_8_5 >= -3
 R209: - contract_14_6 + 10000 node_21_6 + contract_8_6 <= 9997
 R210: - contract_14_6 + 10000 node_21_6 + contract_8_6 >= -3
 R211: contract_14_0 - contract_8_0 + 10000 node_22_0 <= 9997
 R212: contract_14_0 - contract_8_0 + 10000 node_22_0 >= -3
 R213: contract_14_1 - contract_8_1 + 10000 node_22_1 <= 9997
 R214: contract_14_1 - contract_8_1 + 10000 node_22_1 >= -3
 R215: contract_14_2 - contract_8_2 + 10000 node_22_2 <= 9997
 R216: contract_14_2 - contract_8_2 + 10000 node_22_2 >= -3
 R217: contract_14_3 - contract_8_3 + 10000 node_22_3 <= 9997
 R218: contract_14_3 - contract_8_3 + 10000 node_22_3 >= -3
 R219: contract_14_4 - contract_8_4 + 10000 node_22_4 <= 9997
 R220: contract_14_4 - contract_8_4 + 10000 node_22_4 >= -3
 R221: contract_14_5 - contract_8_5 + 10000 node_22_5 <= 9997
 R222: contract_14_5 - contract_8_5 + 10000 node_22_5 >= -3
 R223: contract_14_6 - contract_8_6 + 10000 node_22_6 <= 9997
 R224: contract_14_6 - contract_8_6 + 10000 node_22_6 >= -3
 R225: node_10_0 = 1
 R226: contract_1_0 + 10000 node_24_0 <= 10530.73958566199
 R227: contract_1_0 + 10000 node_24_0 >= 530.7395856619936
 R228: - contract_1_0 + 10000 node_25_0 <= 9469.260414338007
 R229: - contract_1_0 + 10000 node_25_0 >= -530.7395856619936
 R230: node_23_0 = 1
 R231: contract_2_0 + 10000 node_27_0 <= 9989.5
 R232: contract_2_0 + 10000 node_27_0 >= -10.5
 R233: - contract_2_0 + 10000 node_28_0 <= 10010.5
 R234: - contract_2_0 + 10000 node_28_0 >= 10.5
 R235: node_26_0 = 1
 R236: 10000 node_30_0 + contract_3_0 <= 10015
 R237: 10000 node_30_0 + contract_3_0 >= 15
 R238: - contract_3_0 + 10000 node_31_0 <= 9985
 R239: - contract_3_0 + 10000 node_31_0 >= -15
 R240: node_29_0 = 1
 R241: 10000 node_33_0 + contract_4_0 <= 10000
 R242: 10000 node_33_0 + contract_4_0 >= -1.43063e-12
 R243: - contract_4_0 + 10000 node_34_0 <= 10000
 R244: - contract_4_0 + 10000 node_34_0 >= 1.43063e-12
 R245: node_32_0 = 1
 R246: contract_7_0 + 10000 node_36_0 <= 10538.87495092954
 R247: contract_7_0 + 10000 node_36_0 >= 538.874950929538
 R248: - contract_7_0 + 10000 node_37_0 <= 9461.125049070462
 R249: - contract_7_0 + 10000 node_37_0 >= -538.874950929538
 R250: node_35_0 = 1
 R251: contract_8_0 + 10000 node_39_0 <= 10004
 R252: contract_8_0 + 10000 node_39_0 >= 4
 R253: - contract_8_0 + 10000 node_40_0 <= 9996
 R254: - contract_8_0 + 10000 node_40_0 >= -4
 R255: node_38_0 = 1
 R256: 10000 node_42_0 + contract_9_0 <= 10015
 R257: 10000 node_42_0 + contract_9_0 >= 15
 R258: - contract_9_0 + 10000 node_43_0 <= 9985
 R259: - contract_9_0 + 10000 node_43_0 >= -15
 R260: node_41_0 = 1
 R261: 10000 node_45_0 + contract_10_0 <= 10000
 R262: 10000 node_45_0 + contract_10_0 >= -1.59817e-12
 R263: - contract_10_0 + 10000 node_46_0 <= 10000
 R264: - contract_10_0 + 10000 node_46_0 >= 1.59817e-12
 R265: node_44_0 = 1
 R266: contract_13_0 + 10000 node_48_0 <= 10527.92454755736
 R267: contract_13_0 + 10000 node_48_0 >= 527.924547557355
 R268: - contract_13_0 + 10000 node_49_0 <= 9472.075452442645
 R269: - contract_13_0 + 10000 node_49_0 >= -527.924547557355
 R270: node_47_0 = 1
 R271: contract_14_0 + 10000 node_51_0 <= 10000
 R272: contract_14_0 + 10000 node_51_0 >= -7.54952e-15
 R273: - contract_14_0 + 10000 node_52_0 <= 10000
 R274: - contract_14_0 + 10000 node_52_0 >= 7.54952e-15
 R275: node_50_0 = 1
 R276: 10000 node_54_0 + contract_15_0 <= 10013.05030162843
 R277: 10000 node_54_0 + contract_15_0 >= 13.05030162843013
 R278: - contract_15_0 + 10000 node_55_0 <= 9986.949698371569
 R279: - contract_15_0 + 10000 node_55_0 >= -13.05030162843013
 R280: node_53_0 = 1
 R281: 10000 node_57_0 + contract_16_0 <= 10000
 R282: 10000 node_57_0 + contract_16_0 >= -2.22045e-16
 R283: - contract_16_0 + 10000 node_58_0 <= 10000
 R284: - contract_16_0 + 10000 node_58_0 >= 2.22045e-16
 R285: node_56_0 = 1
 R286: contract_14_0 + 10000 node_61_0 <= 10001
 R287: contract_14_0 + 10000 node_61_0 >= 1
 R288: - contract_14_0 + 10000 node_62_0 <= 10002
 R289: - contract_14_0 + 10000 node_62_0 >= 2
 R290: - contract_13_0 + 10000 node_63_0 <= 10000
 R291: - contract_13_0 + 10000 node_63_0 >= 0
 R292: contract_13_0 + 10000 node_64_0 <= 10230
 R293: contract_13_0 + 10000 node_64_0 >= 230
 R294: contract_14_1 + 10000 node_67_1 <= 10001
 R295: contract_14_1 + 10000 node_67_1 >= 1
 R296: - contract_14_1 + 10000 node_68_1 <= 10002
 R297: - contract_14_1 + 10000 node_68_1 >= 2
 R298: - contract_13_1 + 10000 node_69_1 <= 10000
 R299: - contract_13_1 + 10000 node_69_1 >= 0
 R300: contract_13_1 + 10000 node_70_1 <= 10230
 R301: contract_13_1 + 10000 node_70_1 >= 230
 R302: contract_14_2 + 10000 node_73_2 <= 10001
 R303: contract_14_2 + 10000 node_73_2 >= 1
 R304: - contract_14_2 + 10000 node_74_2 <= 10002
 R305: - contract_14_2 + 10000 node_74_2 >= 2
 R306: - contract_13_2 + 10000 node_75_2 <= 10000
 R307: - contract_13_2 + 10000 node_75_2 >= 0
 R308: contract_13_2 + 10000 node_76_2 <= 10230
 R309: contract_13_2 + 10000 node_76_2 >= 230
 R310: contract_14_3 + 10000 node_79_3 <= 10001
 R311: contract_14_3 + 10000 node_79_3 >= 1
 R312: - contract_14_3 + 10000 node_80_3 <= 10002
 R313: - contract_14_3 + 10000 node_80_3 >= 2
 R314: - contract_13_3 + 10000 node_81_3 <= 10000
 R315: - contract_13_3 + 10000 node_81_3 >= 0
 R316: contract_13_3 + 10000 node_82_3 <= 10230
 R317: contract_13_3 + 10000 node_82_3 >= 230
 R318: contract_14_4 + 10000 node_85_4 <= 10001
 R319: contract_14_4 + 10000 node_85_4 >= 1
 R320: - contract_14_4 + 10000 node_86_4 <= 10002
 R321: - contract_14_4 + 10000 node_86_4 >= 2
 R322: - contract_13_4 + 10000 node_87_4 <= 10000
 R323: - contract_13_4 + 10000 node_87_4 >= 0
 R324: contract_13_4 + 10000 node_88_4 <= 10230
 R325: contract_13_4 + 10000 node_88_4 >= 230
 R326: contract_14_5 + 10000 node_91_5 <= 10001
 R327: contract_14_5 + 10000 node_91_5 >= 1
 R328: - contract_14_5 + 10000 node_92_5 <= 10002
 R329: - contract_14_5 + 10000 node_92_5 >= 2
 R330: - contract_13_5 + 10000 node_93_5 <= 10000
 R331: - contract_13_5 + 10000 node_93_5 >= 0
 R332: contract_13_5 + 10000 node_94_5 <= 10230
 R333: contract_13_5 + 10000 node_94_5 >= 230
 R334: contract_14_0 + 10000 node_97_0 <= 10006
 R335: contract_14_0 + 10000 node_97_0 >= 6
 R336: - contract_14_0 + 10000 node_98_0 <= 9997
 R337: - contract_14_0 + 10000 node_98_0 >= -3
 R338: - contract_13_0 + 10000 node_99_0 <= 10000
 R339: - contract_13_0 + 10000 node_99_0 >= 0
 R340: contract_13_0 + 10000 node_100_0 <= 10230
 R341: contract_13_0 + 10000 node_100_0 >= 230
 R342: contract_14_1 + 10000 node_103_1 <= 10006
 R343: contract_14_1 + 10000 node_103_1 >= 6
 R344: - contract_14_1 + 10000 node_104_1 <= 9997
 R345: - contract_14_1 + 10000 node_104_1 >= -3
 R346: - contract_13_1 + 10000 node_105_1 <= 10000
 R347: - contract_13_1 + 10000 node_105_1 >= 0
 R348: contract_13_1 + 10000 node_106_1 <= 10230
 R349: contract_13_1 + 10000 node_106_1 >= 230
 R350: contract_14_2 + 10000 node_109_2 <= 10006
 R351: contract_14_2 + 10000 node_109_2 >= 6
 R352: - contract_14_2 + 10000 node_110_2 <= 9997
 R353: - contract_14_2 + 10000 node_110_2 >= -3
 R354: - contract_13_2 + 10000 node_111_2 <= 10000
 R355: - contract_13_2 + 10000 node_111_2 >= 0
 R356: contract_13_2 + 10000 node_112_2 <= 10230
 R357: contract_13_2 + 10000 node_112_2 >= 230
 R358: contract_14_3 + 10000 node_115_3 <= 10006
 R359: contract_14_3 + 10000 node_115_3 >= 6
 R360: - contract_14_3 + 10000 node_116_3 <= 9997
 R361: - contract_14_3 + 10000 node_116_3 >= -3
 R362: - contract_13_3 + 10000 node_117_3 <= 10000
 R363: - contract_13_3 + 10000 node_117_3 >= 0
 R364: contract_13_3 + 10000 node_118_3 <= 10230
 R365: contract_13_3 + 10000 node_118_3 >= 230
 R366: contract_14_4 + 10000 node_121_4 <= 10006
 R367: contract_14_4 + 10000 node_121_4 >= 6
 R368: - contract_14_4 + 10000 node_122_4 <= 9997
 R369: - contract_14_4 + 10000 node_122_4 >= -3
 R370: - contract_13_4 + 10000 node_123_4 <= 10000
 R371: - contract_13_4 + 10000 node_123_4 >= 0
 R372: contract_13_4 + 10000 node_124_4 <= 10230
 R373: contract_13_4 + 10000 node_124_4 >= 230
 R374: contract_14_5 + 10000 node_127_5 <= 10006
 R375: contract_14_5 + 10000 node_127_5 >= 6
 R376: - contract_14_5 + 10000 node_128_5 <= 9997
 R377: - contract_14_5 + 10000 node_128_5 >= -3
 R378: - contract_13_5 + 10000 node_129_5 <= 10000
 R379: - contract_13_5 + 10000 node_129_5 >= 0
 R380: contract_13_5 + 10000 node_130_5 <= 10230
 R381: contract_13_5 + 10000 node_130_5 >= 230
 R382: contract_14_0 + 10000 node_133_0 <= 10001
 R383: contract_14_0 + 10000 node_133_0 >= 1
 R384: - contract_14_0 + 10000 node_134_0 <= 10002
 R385: - contract_14_0 + 10000 node_134_0 >= 2
 R386: - contract_13_0 + 10000 node_135_0 <= 9770
 R387: - contract_13_0 + 10000 node_135_0 >= -230
 R388: contract_13_0 + 10000 node_136_0 <= 10310
 R389: contract_13_0 + 10000 node_136_0 >= 310
 R390: contract_14_1 + 10000 node_139_1 <= 10001
 R391: contract_14_1 + 10000 node_139_1 >= 1
 R392: - contract_14_1 + 10000 node_140_1 <= 10002
 R393: - contract_14_1 + 10000 node_140_1 >= 2
 R394: - contract_13_1 + 10000 node_141_1 <= 9770
 R395: - contract_13_1 + 10000 node_141_1 >= -230
 R396: contract_13_1 + 10000 node_142_1 <= 10310
 R397: contract_13_1 + 10000 node_142_1 >= 310
 R398: contract_14_2 + 10000 node_145_2 <= 10001
 R399: contract_14_2 + 10000 node_145_2 >= 1
 R400: - contract_14_2 + 10000 node_146_2 <= 10002
 R401: - contract_14_2 + 10000 node_146_2 >= 2
 R402: - contract_13_2 + 10000 node_147_2 <= 9770
 R403: - contract_13_2 + 10000 node_147_2 >= -230
 R404: contract_13_2 + 10000 node_148_2 <= 10310
 R405: contract_13_2 + 10000 node_148_2 >= 310
 R406: contract_14_3 + 10000 node_151_3 <= 10001
 R407: contract_14_3 + 10000 node_151_3 >= 1
 R408: - contract_14_3 + 10000 node_152_3 <= 10002
 R409: - contract_14_3 + 10000 node_152_3 >= 2
 R410: - contract_13_3 + 10000 node_153_3 <= 9770
 R411: - contract_13_3 + 10000 node_153_3 >= -230
 R412: contract_13_3 + 10000 node_154_3 <= 10310
 R413: contract_13_3 + 10000 node_154_3 >= 310
 R414: contract_14_4 + 10000 node_157_4 <= 10001
 R415: contract_14_4 + 10000 node_157_4 >= 1
 R416: - contract_14_4 + 10000 node_158_4 <= 10002
 R417: - contract_14_4 + 10000 node_158_4 >= 2
 R418: - contract_13_4 + 10000 node_159_4 <= 9770
 R419: - contract_13_4 + 10000 node_159_4 >= -230
 R420: contract_13_4 + 10000 node_160_4 <= 10310
 R421: contract_13_4 + 10000 node_160_4 >= 310
 R422: contract_14_5 + 10000 node_163_5 <= 10001
 R423: contract_14_5 + 10000 node_163_5 >= 1
 R424: - contract_14_5 + 10000 node_164_5 <= 10002
 R425: - contract_14_5 + 10000 node_164_5 >= 2
 R426: - contract_13_5 + 10000 node_165_5 <= 9770
 R427: - contract_13_5 + 10000 node_165_5 >= -230
 R428: contract_13_5 + 10000 node_166_5 <= 10310
 R429: contract_13_5 + 10000 node_166_5 >= 310
 R430: contract_14_0 + 10000 node_169_0 <= 10006
 R431: contract_14_0 + 10000 node_169_0 >= 6
 R432: - contract_14_0 + 10000 node_170_0 <= 9998
 R433: - contract_14_0 + 10000 node_170_0 >= -2
 R434: - contract_13_0 + 10000 node_171_0 <= 9770
 R435: - contract_13_0 + 10000 node_171_0 >= -230
 R436: contract_13_0 + 10000 node_172_0 <= 10310
 R437: contract_13_0 + 10000 node_172_0 >= 310
 R438: contract_14_1 + 10000 node_175_1 <= 10006
 R439: contract_14_1 + 10000 node_175_1 >= 6
 R440: - contract_14_1 + 10000 node_176_1 <= 9998
 R441: - contract_14_1 + 10000 node_176_1 >= -2
 R442: - contract_13_1 + 10000 node_177_1 <= 9770
 R443: - contract_13_1 + 10000 node_177_1 >= -230
 R444: contract_13_1 + 10000 node_178_1 <= 10310
 R445: contract_13_1 + 10000 node_178_1 >= 310
 R446: contract_14_2 + 10000 node_181_2 <= 10006
 R447: contract_14_2 + 10000 node_181_2 >= 6
 R448: - contract_14_2 + 10000 node_182_2 <= 9998
 R449: - contract_14_2 + 10000 node_182_2 >= -2
 R450: - contract_13_2 + 10000 node_183_2 <= 9770
 R451: - contract_13_2 + 10000 node_183_2 >= -230
 R452: contract_13_2 + 10000 node_184_2 <= 10310
 R453: contract_13_2 + 10000 node_184_2 >= 310
 R454: contract_14_3 + 10000 node_187_3 <= 10006
 R455: contract_14_3 + 10000 node_187_3 >= 6
 R456: - contract_14_3 + 10000 node_188_3 <= 9998
 R457: - contract_14_3 + 10000 node_188_3 >= -2
 R458: - contract_13_3 + 10000 node_189_3 <= 9770
 R459: - contract_13_3 + 10000 node_189_3 >= -230
 R460: contract_13_3 + 10000 node_190_3 <= 10310
 R461: contract_13_3 + 10000 node_190_3 >= 310
 R462: contract_14_4 + 10000 node_193_4 <= 10006
 R463: contract_14_4 + 10000 node_193_4 >= 6
 R464: - contract_14_4 + 10000 node_194_4 <= 9998
 R465: - contract_14_4 + 10000 node_194_4 >= -2
 R466: - contract_13_4 + 10000 node_195_4 <= 9770
 R467: - contract_13_4 + 10000 node_195_4 >= -230
 R468: contract_13_4 + 10000 node_196_4 <= 10310
 R469: contract_13_4 + 10000 node_196_4 >= 310
 R470: contract_14_5 + 10000 node_199_5 <= 10006
 R471: contract_14_5 + 10000 node_199_5 >= 6
 R472: - contract_14_5 + 10000 node_200_5 <= 9998
 R473: - contract_14_5 + 10000 node_200_5 >= -2
 R474: - contract_13_5 + 10000 node_201_5 <= 9770
 R475: - contract_13_5 + 10000 node_201_5 >= -230
 R476: contract_13_5 + 10000 node_202_5 <= 10310
 R477: contract_13_5 + 10000 node_202_5 >= 310
 R478: contract_14_0 + 10000 node_205_0 <= 9998
 R479: contract_14_0 + 10000 node_205_0 >= -2
 R480: - contract_14_0 + 10000 node_206_0 <= 10005
 R481: - contract_14_0 + 10000 node_206_0 >= 5
 R482: - contract_13_0 + 10000 node_207_0 <= 9770
 R483: - contract_13_0 + 10000 node_207_0 >= -230
 R484: contract_13_0 + 10000 node_208_0 <= 10310
 R485: contract_13_0 + 10000 node_208_0 >= 310
 R486: contract_14_1 + 10000 node_211_1 <= 9998
 R487: contract_14_1 + 10000 node_211_1 >= -2
 R488: - contract_14_1 + 10000 node_212_1 <= 10005
 R489: - contract_14_1 + 10000 node_212_1 >= 5
 R490: - contract_13_1 + 10000 node_213_1 <= 9770
 R491: - contract_13_1 + 10000 node_213_1 >= -230
 R492: contract_13_1 + 10000 node_214_1 <= 10310
 R493: contract_13_1 + 10000 node_214_1 >= 310
 R494: contract_14_2 + 10000 node_217_2 <= 9998
 R495: contract_14_2 + 10000 node_217_2 >= -2
 R496: - contract_14_2 + 10000 node_218_2 <= 10005
 R497: - contract_14_2 + 10000 node_218_2 >= 5
 R498: - contract_13_2 + 10000 node_219_2 <= 9770
 R499: - contract_13_2 + 10000 node_219_2 >= -230
 R500: contract_13_2 + 10000 node_220_2 <= 10310
 R501: contract_13_2 + 10000 node_220_2 >= 310
 R502: contract_14_3 + 10000 node_223_3 <= 9998
 R503: contract_14_3 + 10000 node_223_3 >= -2
 R504: - contract_14_3 + 10000 node_224_3 <= 10005
 R505: - contract_14_3 + 10000 node_224_3 >= 5
 R506: - contract_13_3 + 10000 node_225_3 <= 9770
 R507: - contract_13_3 + 10000 node_225_3 >= -230
 R508: contract_13_3 + 10000 node_226_3 <= 10310
 R509: contract_13_3 + 10000 node_226_3 >= 310
 R510: contract_14_4 + 10000 node_229_4 <= 9998
 R511: contract_14_4 + 10000 node_229_4 >= -2
 R512: - contract_14_4 + 10000 node_230_4 <= 10005
 R513: - contract_14_4 + 10000 node_230_4 >= 5
 R514: - contract_13_4 + 10000 node_231_4 <= 9770
 R515: - contract_13_4 + 10000 node_231_4 >= -230
 R516: contract_13_4 + 10000 node_232_4 <= 10310
 R517: contract_13_4 + 10000 node_232_4 >= 310
 R518: contract_14_5 + 10000 node_235_5 <= 9998
 R519: contract_14_5 + 10000 node_235_5 >= -2
 R520: - contract_14_5 + 10000 node_236_5 <= 10005
 R521: - contract_14_5 + 10000 node_236_5 >= 5
 R522: - contract_13_5 + 10000 node_237_5 <= 9770
 R523: - contract_13_5 + 10000 node_237_5 >= -230
 R524: contract_13_5 + 10000 node_238_5 <= 10310
 R525: contract_13_5 + 10000 node_238_5 >= 310
 R526: contract_14_0 + 10000 node_241_0 <= 10001
 R527: contract_14_0 + 10000 node_241_0 >= 1
 R528: - contract_14_0 + 10000 node_242_0 <= 10002
 R529: - contract_14_0 + 10000 node_242_0 >= 2
 R530: - contract_13_0 + 10000 node_243_0 <= 9690
 R531: - contract_13_0 + 10000 node_243_0 >= -310
 R532: contract_13_0 + 10000 node_244_0 <= 10540
 R533: contract_13_0 + 10000 node_244_0 >= 540
 R534: contract_14_1 + 10000 node_247_1 <= 10001
 R535: contract_14_1 + 10000 node_247_1 >= 1
 R536: - contract_14_1 + 10000 node_248_1 <= 10002
 R537: - contract_14_1 + 10000 node_248_1 >= 2
 R538: - contract_13_1 + 10000 node_249_1 <= 9690
 R539: - contract_13_1 + 10000 node_249_1 >= -310
 R540: contract_13_1 + 10000 node_250_1 <= 10540
 R541: contract_13_1 + 10000 node_250_1 >= 540
 R542: contract_14_2 + 10000 node_253_2 <= 10001
 R543: contract_14_2 + 10000 node_253_2 >= 1
 R544: - contract_14_2 + 10000 node_254_2 <= 10002
 R545: - contract_14_2 + 10000 node_254_2 >= 2
 R546: - contract_13_2 + 10000 node_255_2 <= 9690
 R547: - contract_13_2 + 10000 node_255_2 >= -310
 R548: contract_13_2 + 10000 node_256_2 <= 10540
 R549: contract_13_2 + 10000 node_256_2 >= 540
 R550: contract_14_3 + 10000 node_259_3 <= 10001
 R551: contract_14_3 + 10000 node_259_3 >= 1
 R552: - contract_14_3 + 10000 node_260_3 <= 10002
 R553: - contract_14_3 + 10000 node_260_3 >= 2
 R554: - contract_13_3 + 10000 node_261_3 <= 9690
 R555: - contract_13_3 + 10000 node_261_3 >= -310
 R556: contract_13_3 + 10000 node_262_3 <= 10540
 R557: contract_13_3 + 10000 node_262_3 >= 540
 R558: contract_14_4 + 10000 node_265_4 <= 10001
 R559: contract_14_4 + 10000 node_265_4 >= 1
 R560: - contract_14_4 + 10000 node_266_4 <= 10002
 R561: - contract_14_4 + 10000 node_266_4 >= 2
 R562: - contract_13_4 + 10000 node_267_4 <= 9690
 R563: - contract_13_4 + 10000 node_267_4 >= -310
 R564: contract_13_4 + 10000 node_268_4 <= 10540
 R565: contract_13_4 + 10000 node_268_4 >= 540
 R566: contract_14_5 + 10000 node_271_5 <= 10001
 R567: contract_14_5 + 10000 node_271_5 >= 1
 R568: - contract_14_5 + 10000 node_272_5 <= 10002
 R569: - contract_14_5 + 10000 node_272_5 >= 2
 R570: - contract_13_5 + 10000 node_273_5 <= 9690
 R571: - contract_13_5 + 10000 node_273_5 >= -310
 R572: contract_13_5 + 10000 node_274_5 <= 10540
 R573: contract_13_5 + 10000 node_274_5 >= 540
 R574: contract_14_0 + 10000 node_277_0 <= 10006
 R575: contract_14_0 + 10000 node_277_0 >= 6
 R576: - contract_14_0 + 10000 node_278_0 <= 9997
 R577: - contract_14_0 + 10000 node_278_0 >= -3
 R578: - contract_13_0 + 10000 node_279_0 <= 9690
 R579: - contract_13_0 + 10000 node_279_0 >= -310
 R580: contract_13_0 + 10000 node_280_0 <= 10540
 R581: contract_13_0 + 10000 node_280_0 >= 540
 R582: contract_14_1 + 10000 node_283_1 <= 10006
 R583: contract_14_1 + 10000 node_283_1 >= 6
 R584: - contract_14_1 + 10000 node_284_1 <= 9997
 R585: - contract_14_1 + 10000 node_284_1 >= -3
 R586: - contract_13_1 + 10000 node_285_1 <= 9690
 R587: - contract_13_1 + 10000 node_285_1 >= -310
 R588: contract_13_1 + 10000 node_286_1 <= 10540
 R589: contract_13_1 + 10000 node_286_1 >= 540
 R590: contract_14_2 + 10000 node_289_2 <= 10006
 R591: contract_14_2 + 10000 node_289_2 >= 6
 R592: - contract_14_2 + 10000 node_290_2 <= 9997
 R593: - contract_14_2 + 10000 node_290_2 >= -3
 R594: - contract_13_2 + 10000 node_291_2 <= 9690
 R595: - contract_13_2 + 10000 node_291_2 >= -310
 R596: contract_13_2 + 10000 node_292_2 <= 10540
 R597: contract_13_2 + 10000 node_292_2 >= 540
 R598: contract_14_3 + 10000 node_295_3 <= 10006
 R599: contract_14_3 + 10000 node_295_3 >= 6
 R600: - contract_14_3 + 10000 node_296_3 <= 9997
 R601: - contract_14_3 + 10000 node_296_3 >= -3
 R602: - contract_13_3 + 10000 node_297_3 <= 9690
 R603: - contract_13_3 + 10000 node_297_3 >= -310
 R604: contract_13_3 + 10000 node_298_3 <= 10540
 R605: contract_13_3 + 10000 node_298_3 >= 540
 R606: contract_14_4 + 10000 node_301_4 <= 10006
 R607: contract_14_4 + 10000 node_301_4 >= 6
 R608: - contract_14_4 + 10000 node_302_4 <= 9997
 R609: - contract_14_4 + 10000 node_302_4 >= -3
 R610: - contract_13_4 + 10000 node_303_4 <= 9690
 R611: - contract_13_4 + 10000 node_303_4 >= -310
 R612: contract_13_4 + 10000 node_304_4 <= 10540
 R613: contract_13_4 + 10000 node_304_4 >= 540
 R614: contract_14_5 + 10000 node_307_5 <= 10006
 R615: contract_14_5 + 10000 node_307_5 >= 6
 R616: - contract_14_5 + 10000 node_308_5 <= 9997
 R617: - contract_14_5 + 10000 node_308_5 >= -3
 R618: - contract_13_5 + 10000 node_309_5 <= 9690
 R619: - contract_13_5 + 10000 node_309_5 >= -310
 R620: contract_13_5 + 10000 node_310_5 <= 10540
 R621: contract_13_5 + 10000 node_310_5 >= 540
 R622: - contract_1_0 + contract_1_1 - 0.5 contract_3_0 = 0
 R623: - contract_1_1 + contract_1_2 - 0.5 contract_3_1 = 0
 R624: - contract_1_2 + contract_1_3 - 0.5 contract_3_2 = 0
 R625: - contract_1_3 + contract_1_4 - 0.5 contract_3_3 = 0
 R626: - contract_1_4 + contract_1_5 - 0.5 contract_3_4 = 0
 R627: - contract_1_5 + contract_1_6 - 0.5 contract_3_5 = 0
 R628: - contract_2_0 + contract_2_1 - 0.5 contract_4_0 = 0
 R629: - contract_2_1 + contract_2_2 - 0.5 contract_4_1 = 0
 R630: - contract_2_2 + contract_2_3 - 0.5 contract_4_2 = 0
 R631: - contract_2_3 + contract_2_4 - 0.5 contract_4_3 = 0
 R632: - contract_2_4 + contract_2_5 - 0.5 contract_4_4 = 0
 R633: - contract_2_5 + contract_2_6 - 0.5 contract_4_5 = 0
 R634: - 0.5 contract_5_0 - contract_3_0 + contract_3_1 = 0
 R635: - 0.5 contract_5_1 - contract_3_1 + contract_3_2 = 0
 R636: - 0.5 contract_5_2 - contract_3_2 + contract_3_3 = 0
 R637: - 0.5 contract_5_3 - contract_3_3 + contract_3_4 = 0
 R638: - 0.5 contract_5_4 - contract_3_4 + contract_3_5 = 0
 R639: - 0.5 contract_5_5 - contract_3_5 + contract_3_6 = 0
 R640: - 0.5 contract_6_0 - contract_4_0 + contract_4_1 = 0
 R641: - 0.5 contract_6_1 - contract_4_1 + contract_4_2 = 0
 R642: - 0.5 contract_6_2 - contract_4_2 + contract_4_3 = 0
 R643: - 0.5 contract_6_3 - contract_4_3 + contract_4_4 = 0
 R644: - 0.5 contract_6_4 - contract_4_4 + contract_4_5 = 0
 R645: - 0.5 contract_6_5 - contract_4_5 + contract_4_6 = 0
 R646: - contract_7_0 + contract_7_1 - 0.5 contract_9_0 = 0
 R647: - contract_7_1 + contract_7_2 - 0.5 contract_9_1 = 0
 R648: - contract_7_2 + contract_7_3 - 0.5 contract_9_2 = 0
 R649: - contract_7_3 + contract_7_4 - 0.5 contract_9_3 = 0
 R650: - contract_7_4 + contract_7_5 - 0.5 contract_9_4 = 0
 R651: - contract_7_5 + contract_7_6 - 0.5 contract_9_5 = 0
 R652: - contract_8_0 + contract_8_1 - 0.5 contract_10_0 = 0
 R653: - contract_8_1 + contract_8_2 - 0.5 contract_10_1 = 0
 R654: - contract_8_2 + contract_8_3 - 0.5 contract_10_2 = 0
 R655: - contract_8_3 + contract_8_4 - 0.5 contract_10_3 = 0
 R656: - contract_8_4 + contract_8_5 - 0.5 contract_10_4 = 0
 R657: - contract_8_5 + contract_8_6 - 0.5 contract_10_5 = 0
 R658: - 0.5 contract_11_0 - contract_9_0 + contract_9_1 = 0
 R659: - 0.5 contract_11_1 - contract_9_1 + contract_9_2 = 0
 R660: - 0.5 contract_11_2 - contract_9_2 + contract_9_3 = 0
 R661: - 0.5 contract_11_3 - contract_9_3 + contract_9_4 = 0
 R662: - 0.5 contract_11_4 - contract_9_4 + contract_9_5 = 0
 R663: - 0.5 contract_11_5 - contract_9_5 + contract_9_6 = 0
 R664: - 0.5 contract_12_0 - contract_10_0 + contract_10_1 = 0
 R665: - 0.5 contract_12_1 - contract_10_1 + contract_10_2 = 0
 R666: - 0.5 contract_12_2 - contract_10_2 + contract_10_3 = 0
 R667: - 0.5 contract_12_3 - contract_10_3 + contract_10_4 = 0
 R668: - 0.5 contract_12_4 - contract_10_4 + contract_10_5 = 0
 R669: - 0.5 contract_12_5 - contract_10_5 + contract_10_6 = 0
 R670: - contract_13_0 + contract_13_1 - 0.5 contract_15_0 = 0
 R671: - contract_13_1 + contract_13_2 - 0.5 contract_15_1 = 0
 R672: - contract_13_2 + contract_13_3 - 0.5 contract_15_2 = 0
 R673: - contract_13_3 + contract_13_4 - 0.5 contract_15_3 = 0
 R674: - contract_13_4 + contract_13_5 - 0.5 contract_15_4 = 0
 R675: - contract_13_5 + contract_13_6 - 0.5 contract_15_5 = 0
 R676: - contract_14_0 + contract_14_1 - 0.5 contract_16_0 = 0
 R677: - contract_14_1 + contract_14_2 - 0.5 contract_16_1 = 0
 R678: - contract_14_2 + contract_14_3 - 0.5 contract_16_2 = 0
 R679: - contract_14_3 + contract_14_4 - 0.5 contract_16_3 = 0
 R680: - contract_14_4 + contract_14_5 - 0.5 contract_16_4 = 0
 R681: - contract_14_5 + contract_14_6 - 0.5 contract_16_5 = 0
 R682: - contract_15_0 + contract_15_1 - 0.5 contract_17_0 = 0
 R683: - contract_15_1 + contract_15_2 - 0.5 contract_17_1 = 0
 R684: - contract_15_2 + contract_15_3 - 0.5 contract_17_2 = 0
 R685: - contract_15_3 + contract_15_4 - 0.5 contract_17_3 = 0
 R686: - contract_15_4 + contract_15_5 - 0.5 contract_17_4 = 0
 R687: - contract_15_5 + contract_15_6 - 0.5 contract_17_5 = 0
 R688: - contract_16_0 + contract_16_1 - 0.5 contract_18_0 = 0
 R689: - contract_16_1 + contract_16_2 - 0.5 contract_18_1 = 0
 R690: - contract_16_2 + contract_16_3 - 0.5 contract_18_2 = 0
 R691: - contract_16_3 + contract_16_4 - 0.5 contract_18_3 = 0
 R692: - contract_16_4 + contract_16_5 - 0.5 contract_18_4 = 0
 R693: - contract_16_5 + contract_16_6 - 0.5 contract_18_5 = 0
 R694: regions_0 = 1
 R695: regions_1 = 1
 R696: regions_2 = 1
 R697: regions_3 = 1
 R698: regions_4 = 1
 R699: regions_5 = 1
 R700: contract_13_0 - goal_x_0 <= 540
 R701: - contract_13_0 - goal_x_0 <= -540
 R702: contract_14_0 - goal_y_0 <= 0
 R703: - contract_14_0 - goal_y_0 <= 0
 R704: contract_13_1 - goal_x_1 <= 540
 R705: - contract_13_1 - goal_x_1 <= -540
 R706: contract_14_1 - goal_y_1 <= 0
 R707: - contract_14_1 - goal_y_1 <= 0
 R708: contract_13_2 - goal_x_2 <= 540
 R709: - contract_13_2 - goal_x_2 <= -540
 R710: contract_14_2 - goal_y_2 <= 0
 R711: - contract_14_2 - goal_y_2 <= 0
 R712: contract_13_3 - goal_x_3 <= 540
 R713: - contract_13_3 - goal_x_3 <= -540
 R714: contract_14_3 - goal_y_3 <= 0
 R715: - contract_14_3 - goal_y_3 <= 0
 R716: contract_13_4 - goal_x_4 <= 540
 R717: - contract_13_4 - goal_x_4 <= -540
 R718: contract_14_4 - goal_y_4 <= 0
 R719: - contract_14_4 - goal_y_4 <= 0
 R720: contract_13_5 - goal_x_5 <= 540
 R721: - contract_13_5 - goal_x_5 <= -540
 R722: contract_14_5 - goal_y_5 <= 0
 R723: - contract_14_5 - goal_y_5 <= 0
 R724: contract_17_0 - fuel_x_0 <= 0
 R725: - contract_17_0 - fuel_x_0 <= 0
 R726: contract_18_0 - fuel_y_0 <= 0
 R727: - contract_18_0 - fuel_y_0 <= 0
 R728: contract_17_1 - fuel_x_1 <= 0
 R729: - contract_17_1 - fuel_x_1 <= 0
 R730: contract_18_1 - fuel_y_1 <= 0
 R731: - contract_18_1 - fuel_y_1 <= 0
 R732: contract_17_2 - fuel_x_2 <= 0
 R733: - contract_17_2 - fuel_x_2 <= 0
 R734: contract_18_2 - fuel_y_2 <= 0
 R735: - contract_18_2 - fuel_y_2 <= 0
 R736: contract_17_3 - fuel_x_3 <= 0
 R737: - contract_17_3 - fuel_x_3 <= 0
 R738: contract_18_3 - fuel_y_3 <= 0
 R739: - contract_18_3 - fuel_y_3 <= 0
 R740: contract_17_4 - fuel_x_4 <= 0
 R741: - contract_17_4 - fuel_x_4 <= 0
 R742: contract_18_4 - fuel_y_4 <= 0
 R743: - contract_18_4 - fuel_y_4 <= 0
 R744: contract_17_5 - fuel_x_5 <= 0
 R745: - contract_17_5 - fuel_x_5 <= 0
 R746: contract_18_5 - fuel_y_5 <= 0
 R747: - contract_18_5 - fuel_y_5 <= 0
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
 -15 <= contract_4_0 <= 15
 -15 <= contract_9_0 <= 15
 -15 <= contract_10_0 <= 15
 -15 <= contract_15_0 <= 15
 -15 <= contract_16_0 <= 15
 -15 <= contract_3_1 <= 15
 -15 <= contract_3_2 <= 15
 -15 <= contract_3_3 <= 15
 -15 <= contract_3_4 <= 15
 -15 <= contract_3_5 <= 15
 -15 <= contract_4_1 <= 15
 -15 <= contract_4_2 <= 15
 -15 <= contract_4_3 <= 15
 -15 <= contract_4_4 <= 15
 -15 <= contract_4_5 <= 15
 -15 <= contract_3_6 <= 15
 -15 <= contract_4_6 <= 15
 -15 <= contract_9_1 <= 15
 -15 <= contract_9_2 <= 15
 -15 <= contract_9_3 <= 15
 -15 <= contract_9_4 <= 15
 -15 <= contract_9_5 <= 15
 -15 <= contract_10_1 <= 15
 -15 <= contract_10_2 <= 15
 -15 <= contract_10_3 <= 15
 -15 <= contract_10_4 <= 15
 -15 <= contract_10_5 <= 15
 -15 <= contract_9_6 <= 15
 -15 <= contract_10_6 <= 15
 -15 <= contract_15_1 <= 15
 -15 <= contract_15_2 <= 15
 -15 <= contract_15_3 <= 15
 -15 <= contract_15_4 <= 15
 -15 <= contract_15_5 <= 15
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
 node_27_0 node_28_0 node_29_0 node_30_0 node_31_0 node_32_0 node_33_0
 node_34_0 node_35_0 node_36_0 node_37_0 node_38_0 node_39_0 node_40_0
 node_41_0 node_42_0 node_43_0 node_44_0 node_45_0 node_46_0 node_47_0
 node_48_0 node_49_0 node_50_0 node_51_0 node_52_0 node_53_0 node_54_0
 node_55_0 node_56_0 node_57_0 node_58_0 node_59_0 node_60_0 node_61_0
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
 node_164_5 node_165_5 node_166_5 node_167_0 node_168_0 node_169_0
 node_170_0 node_171_0 node_172_0 node_173_0 node_174_1 node_175_1
 node_176_1 node_177_1 node_178_1 node_179_0 node_180_2 node_181_2
 node_182_2 node_183_2 node_184_2 node_185_0 node_186_3 node_187_3
 node_188_3 node_189_3 node_190_3 node_191_0 node_192_4 node_193_4
 node_194_4 node_195_4 node_196_4 node_197_0 node_198_5 node_199_5
 node_200_5 node_201_5 node_202_5 node_203_0 node_204_0 node_205_0
 node_206_0 node_207_0 node_208_0 node_209_0 node_210_1 node_211_1
 node_212_1 node_213_1 node_214_1 node_215_0 node_216_2 node_217_2
 node_218_2 node_219_2 node_220_2 node_221_0 node_222_3 node_223_3
 node_224_3 node_225_3 node_226_3 node_227_0 node_228_4 node_229_4
 node_230_4 node_231_4 node_232_4 node_233_0 node_234_5 node_235_5
 node_236_5 node_237_5 node_238_5 node_239_0 node_240_0 node_241_0
 node_242_0 node_243_0 node_244_0 node_245_0 node_246_1 node_247_1
 node_248_1 node_249_1 node_250_1 node_251_0 node_252_2 node_253_2
 node_254_2 node_255_2 node_256_2 node_257_0 node_258_3 node_259_3
 node_260_3 node_261_3 node_262_3 node_263_0 node_264_4 node_265_4
 node_266_4 node_267_4 node_268_4 node_269_0 node_270_5 node_271_5
 node_272_5 node_273_5 node_274_5 node_275_0 node_276_0 node_277_0
 node_278_0 node_279_0 node_280_0 node_281_0 node_282_1 node_283_1
 node_284_1 node_285_1 node_286_1 node_287_0 node_288_2 node_289_2
 node_290_2 node_291_2 node_292_2 node_293_0 node_294_3 node_295_3
 node_296_3 node_297_3 node_298_3 node_299_0 node_300_4 node_301_4
 node_302_4 node_303_4 node_304_4 node_305_0 node_306_5 node_307_5
 node_308_5 node_309_5 node_310_5 regions_0 regions_1 regions_2 regions_3
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
 GC25: node_23_0 = AND ( node_24_0 , node_25_0 )
 GC26: node_26_0 = AND ( node_27_0 , node_28_0 )
 GC27: node_29_0 = AND ( node_30_0 , node_31_0 )
 GC28: node_32_0 = AND ( node_33_0 , node_34_0 )
 GC29: node_35_0 = AND ( node_36_0 , node_37_0 )
 GC30: node_38_0 = AND ( node_39_0 , node_40_0 )
 GC31: node_41_0 = AND ( node_42_0 , node_43_0 )
 GC32: node_44_0 = AND ( node_45_0 , node_46_0 )
 GC33: node_47_0 = AND ( node_48_0 , node_49_0 )
 GC34: node_50_0 = AND ( node_51_0 , node_52_0 )
 GC35: node_53_0 = AND ( node_54_0 , node_55_0 )
 GC36: node_56_0 = AND ( node_57_0 , node_58_0 )
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
 GC73: node_168_0 = AND ( node_169_0 , node_170_0 , node_171_0 , node_172_0 )
 GC74: node_167_0 = AND ( node_168_0 )
 GC75: node_174_1 = AND ( node_175_1 , node_176_1 , node_177_1 , node_178_1 )
 GC76: node_173_0 = AND ( node_174_1 )
 GC77: node_180_2 = AND ( node_181_2 , node_182_2 , node_183_2 , node_184_2 )
 GC78: node_179_0 = AND ( node_180_2 )
 GC79: node_186_3 = AND ( node_187_3 , node_188_3 , node_189_3 , node_190_3 )
 GC80: node_185_0 = AND ( node_186_3 )
 GC81: node_192_4 = AND ( node_193_4 , node_194_4 , node_195_4 , node_196_4 )
 GC82: node_191_0 = AND ( node_192_4 )
 GC83: node_198_5 = AND ( node_199_5 , node_200_5 , node_201_5 , node_202_5 )
 GC84: node_197_0 = AND ( node_198_5 )
 GC85: node_204_0 = AND ( node_205_0 , node_206_0 , node_207_0 , node_208_0 )
 GC86: node_203_0 = AND ( node_204_0 )
 GC87: node_210_1 = AND ( node_211_1 , node_212_1 , node_213_1 , node_214_1 )
 GC88: node_209_0 = AND ( node_210_1 )
 GC89: node_216_2 = AND ( node_217_2 , node_218_2 , node_219_2 , node_220_2 )
 GC90: node_215_0 = AND ( node_216_2 )
 GC91: node_222_3 = AND ( node_223_3 , node_224_3 , node_225_3 , node_226_3 )
 GC92: node_221_0 = AND ( node_222_3 )
 GC93: node_228_4 = AND ( node_229_4 , node_230_4 , node_231_4 , node_232_4 )
 GC94: node_227_0 = AND ( node_228_4 )
 GC95: node_234_5 = AND ( node_235_5 , node_236_5 , node_237_5 , node_238_5 )
 GC96: node_233_0 = AND ( node_234_5 )
 GC97: node_240_0 = AND ( node_241_0 , node_242_0 , node_243_0 , node_244_0 )
 GC98: node_239_0 = AND ( node_240_0 )
 GC99: node_246_1 = AND ( node_247_1 , node_248_1 , node_249_1 , node_250_1 )
 GC100: node_245_0 = AND ( node_246_1 )
 GC101: node_252_2 = AND ( node_253_2 , node_254_2 , node_255_2 ,
   node_256_2 )
 GC102: node_251_0 = AND ( node_252_2 )
 GC103: node_258_3 = AND ( node_259_3 , node_260_3 , node_261_3 ,
   node_262_3 )
 GC104: node_257_0 = AND ( node_258_3 )
 GC105: node_264_4 = AND ( node_265_4 , node_266_4 , node_267_4 ,
   node_268_4 )
 GC106: node_263_0 = AND ( node_264_4 )
 GC107: node_270_5 = AND ( node_271_5 , node_272_5 , node_273_5 ,
   node_274_5 )
 GC108: node_269_0 = AND ( node_270_5 )
 GC109: node_276_0 = AND ( node_277_0 , node_278_0 , node_279_0 ,
   node_280_0 )
 GC110: node_275_0 = AND ( node_276_0 )
 GC111: node_282_1 = AND ( node_283_1 , node_284_1 , node_285_1 ,
   node_286_1 )
 GC112: node_281_0 = AND ( node_282_1 )
 GC113: node_288_2 = AND ( node_289_2 , node_290_2 , node_291_2 ,
   node_292_2 )
 GC114: node_287_0 = AND ( node_288_2 )
 GC115: node_294_3 = AND ( node_295_3 , node_296_3 , node_297_3 ,
   node_298_3 )
 GC116: node_293_0 = AND ( node_294_3 )
 GC117: node_300_4 = AND ( node_301_4 , node_302_4 , node_303_4 ,
   node_304_4 )
 GC118: node_299_0 = AND ( node_300_4 )
 GC119: node_306_5 = AND ( node_307_5 , node_308_5 , node_309_5 ,
   node_310_5 )
 GC120: node_305_0 = AND ( node_306_5 )
 GC121: regions_0 = OR ( node_59_0 , node_95_0 , node_131_0 , node_167_0 ,
   node_203_0 , node_239_0 , node_275_0 )
 GC122: regions_1 = OR ( node_65_0 , node_101_0 , node_137_0 , node_173_0 ,
   node_209_0 , node_245_0 , node_281_0 )
 GC123: regions_2 = OR ( node_71_0 , node_107_0 , node_143_0 , node_179_0 ,
   node_215_0 , node_251_0 , node_287_0 )
 GC124: regions_3 = OR ( node_77_0 , node_113_0 , node_149_0 , node_185_0 ,
   node_221_0 , node_257_0 , node_293_0 )
 GC125: regions_4 = OR ( node_83_0 , node_119_0 , node_155_0 , node_191_0 ,
   node_227_0 , node_263_0 , node_299_0 )
 GC126: regions_5 = OR ( node_89_0 , node_125_0 , node_161_0 , node_197_0 ,
   node_233_0 , node_269_0 , node_305_0 )
End
