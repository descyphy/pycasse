\ LP format - for model browsing. Use MPS format to capture full model detail.
Minimize
 
Subject To
 R0: rho_1 + 1.644853626951472 sigma_1 = 0.2
 R1: rho + 2.326347874040841 sigma = 0.2
 R2: anx_1 + anx_2 = 0
 R3: anx_3 <= -0.0001
 qc0: rho_2 - 61.07030970896656 sigma_2 - c + [
   69.27984586056796 sigma_2 * p ] = 0
 qc1: sigma_squared + [ - sigma_1 ^2 - sigma_2 ^2 ] = 0
Bounds
 -10000 <= rho_1 <= 10000
 -10000 <= rho_2 <= 10000
 -10000 <= rho <= 10000
 sigma_1 <= 1
 sigma_2 <= 1
 sigma <= 1
 sigma_squared <= 10000
 1 <= c <= 3
 0.95 <= p <= 1
 -10000 <= anx_1 <= 10000
 -10000 <= anx_2 <= 10000
 -10000 <= anx_3 <= 10000
General Constraints
 GC0: sigma = POW ( sigma_squared ^ 0.5 )
 GC1: anx_1 = MIN ( rho_1 , rho_2 )
 GC2: anx_3 = MAX ( anx_2 , rho )
End
