#PM_H = 500.4
#PM_M = 35.5
#PM_L = 0.0
 
AQI_H = 500.0
AQI_M = 101.0  
AQI_L = 0.0   

numberOfAntAttributes = 2
#relativeWeight = 1.0   
  
cbd_0 = 1.0
cbd_1 = 0.0  
cbd_2 = 0.0 
cbd_3 = 0.0  
cbd_4 = 1.0
cbd_5 = 0.0  
cbd_6 = 0.0  
cbd_7 = 0.0
cbd_8 = 1.0
 
aqi1 = 1.0
aqi2 = 1.0  
aqi3 = 1.0
aqi4 = 1.0 
aqi5 = 1.0      
 
def ruleBase(s,c,x):     
    global consequentBeliefDegree 
    #global relativeWeight1
    #global relativeWeight2
    temp_consequentBeliefDegree = [x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], x[8], x[9], x[10], x[11], x[12], x[13], x[14], x[15], x[16], x[17], x[18], x[19], x[20], x[21], x[22], x[23], x[24], x[25], x[26], x[27], x[28], x[29], x[30], x[31], x[32], x[33], x[34], x[35], x[36], x[37], x[38], x[39], x[40], x[41], x[42], x[43], x[44], x[45], x[46], x[47], x[48], x[49], x[50], x[51], x[52], x[53], x[54], x[55], x[56], x[57], x[58], x[59], x[60], x[61], x[62], x[63], x[64], x[65], x[66], x[67], x[68], x[69], x[70], x[71], x[72], x[73], x[74], x[75], x[76], x[77], x[78], x[79], x[80], x[81], x[82], x[83], x[84], x[85], x[86], x[87], x[88], x[89], x[90], x[91], x[92], x[93], x[94], x[95], x[96], x[97], x[98], x[99], x[100], x[101], x[102], x[103], x[104], x[105], x[106], x[107]] 
        
    de0 = x[0]/(x[0] + x[1] + x[2])
    de1 = x[1]/(x[0] + x[1] + x[2])
    de2 = x[2]/(x[0] + x[1] + x[2])
    de3 = x[3]/(x[3] + x[4] + x[5]) 
    de4 = x[4]/(x[3] + x[4] + x[5])
    de5 = x[5]/(x[3] + x[4] + x[5])
    de6 = x[6]/(x[6] + x[7] + x[8])
    de7 = x[7]/(x[6] + x[7] + x[8])
    de8 = x[8]/(x[6] + x[7] + x[8])
    de9 = x[9]/(x[9] + x[10] + x[11])
    de10 = x[10]/(x[9] + x[10] + x[11])
    de11 = x[11]/(x[9] + x[10] + x[11])   
    de12 = x[12]/(x[12] + x[13] + x[14])   
    de13 = x[13]/(x[12] + x[13] + x[14])
    de14 = x[14]/(x[12] + x[13] + x[14])   
    de15 = x[15]/(x[15] + x[16] + x[17])
    de16 = x[16]/(x[15] + x[16] + x[17])
    de17 = x[17]/(x[15] + x[16] + x[17])   
    de18 = x[18]/(x[18] + x[19] + x[20])   
    de19 = x[19]/(x[18] + x[19] + x[20])
    de20 = x[20]/(x[18] + x[19] + x[20])   
    de21 = x[21]/(x[21] + x[22] + x[23])   
    de22 = x[22]/(x[21] + x[22] + x[23])
    de23 = x[23]/(x[21] + x[22] + x[23])
    de24 = x[24]/(x[24] + x[25] + x[26])   
    de25 = x[25]/(x[24] + x[25] + x[26])
    de26 = x[26]/(x[24] + x[25] + x[26])
    de27 = x[27]/(x[27] + x[28] + x[29])
    de28 = x[28]/(x[27] + x[28] + x[29])
    de29 = x[29]/(x[27] + x[28] + x[29])
    de30 = x[30]/(x[30] + x[31] + x[32])   
    de31 = x[31]/(x[30] + x[31] + x[32])
    de32 = x[32]/(x[30] + x[31] + x[32])
    de33 = x[33]/(x[33] + x[34] + x[35])
    de34 = x[34]/(x[33] + x[34] + x[35])
    de35 = x[35]/(x[33] + x[34] + x[35])
    de36 = x[36]/(x[36] + x[37] + x[38])
    de37 = x[37]/(x[36] + x[37] + x[38])
    de38 = x[38]/(x[36] + x[37] + x[38])
    de39 = x[39]/(x[39] + x[40] + x[41])   
    de40 = x[40]/(x[39] + x[40] + x[41])
    de41 = x[41]/(x[39] + x[40] + x[41])
    de42 = x[42]/(x[42] + x[43] + x[44])   
    de43 = x[43]/(x[42] + x[43] + x[44])
    de44 = x[44]/(x[42] + x[43] + x[44])
    de45 = x[45]/(x[45] + x[46] + x[47])   
    de46 = x[46]/(x[45] + x[46] + x[47])
    de47 = x[47]/(x[45] + x[46] + x[47])
    de48 = x[48]/(x[48] + x[49] + x[50])
    de49 = x[49]/(x[48] + x[49] + x[50])
    de50 = x[50]/(x[48] + x[49] + x[50])
    de51 = x[51]/(x[51] + x[52] + x[53])
    de52 = x[52]/(x[51] + x[52] + x[53])
    de53 = x[53]/(x[51] + x[52] + x[53])
    de54 = x[54]/(x[54] + x[55] + x[56])
    de55 = x[55]/(x[54] + x[55] + x[56])
    de56 = x[56]/(x[54] + x[55] + x[56])
    de57 = x[57]/(x[57] + x[58] + x[59])
    de58 = x[58]/(x[57] + x[58] + x[59])
    de59 = x[59]/(x[57] + x[58] + x[59])
    de60 = x[60]/(x[60] + x[61] + x[62])
    de61 = x[61]/(x[60] + x[61] + x[62])
    de62 = x[62]/(x[60] + x[61] + x[62])
    de63 = x[63]/(x[63] + x[64] + x[65])
    de64 = x[64]/(x[63] + x[64] + x[65])
    de65 = x[65]/(x[63] + x[64] + x[65])
    de66 = x[66]/(x[66] + x[67] + x[68])
    de67 = x[67]/(x[66] + x[67] + x[68])
    de68 = x[68]/(x[66] + x[67] + x[68])
    de69 = x[69]/(x[69] + x[70] + x[71])
    de70 = x[70]/(x[69] + x[70] + x[71])
    de71 = x[71]/(x[69] + x[70] + x[71])
    de72 = x[72]/(x[72] + x[73] + x[74])
    de73 = x[73]/(x[72] + x[73] + x[74])
    de74 = x[74]/(x[72] + x[73] + x[74])
    de75 = x[75]/(x[75] + x[76] + x[77])
    de76 = x[76]/(x[75] + x[76] + x[77])
    de77 = x[77]/(x[75] + x[76] + x[77])
    de78 = x[78]/(x[78] + x[79] + x[80])
    de79 = x[79]/(x[78] + x[79] + x[80])
    de80 = x[80]/(x[78] + x[79] + x[80])
    de81 = x[81]/(x[81] + x[82] + x[83])
    de82 = x[82]/(x[81] + x[82] + x[83])
    de83 = x[83]/(x[81] + x[82] + x[83])
    de84 = x[84]/(x[84] + x[85] + x[86])
    de85 = x[85]/(x[84] + x[85] + x[86])
    de86 = x[86]/(x[84] + x[85] + x[86])
    de87 = x[87]/(x[87] + x[88] + x[89])
    de88 = x[88]/(x[87] + x[88] + x[89])
    de89 = x[89]/(x[87] + x[88] + x[89])
    de90 = x[90]/(x[90] + x[91] + x[92])
    de91 = x[91]/(x[90] + x[91] + x[92])
    de92 = x[92]/(x[90] + x[91] + x[92])
    de93 = x[93]/(x[93] + x[94] + x[95])
    de94 = x[94]/(x[93] + x[94] + x[95])
    de95 = x[95]/(x[93] + x[94] + x[95])
    de96 = x[96]/(x[96] + x[97] + x[98])
    de97 = x[97]/(x[96] + x[97] + x[98])
    de98 = x[98]/(x[96] + x[97] + x[98])
    de99 = x[99]/(x[99] + x[100] + x[101])
    de100 = x[100]/(x[99] + x[100] + x[101])
    de101 = x[101]/(x[99] + x[100] + x[101])
    de102 = x[102]/(x[102] + x[103] + x[104])
    de103 = x[103]/(x[102] + x[103] + x[104])
    de104 = x[104]/(x[102] + x[103] + x[104])
    de105 = x[105]/(x[105] + x[106] + x[107])
    de106 = x[106]/(x[105] + x[106] + x[107])
    de107 = x[107]/(x[105] + x[106] + x[107])
          
    consequentBeliefDegree = [de0, de1, de2, de3, de4, de5, de6, de7, de8, de9, de10, de11, de12, de13, de14, de15, de16, de17, de18, de19, de20, de21, de22, de23, de24, de25, de26, de27, de28, de29, de30, de31, de32, de33, de34, de35, de36, de37, de38, de39, de40, de41, de42, de43, de44, de45, de46, de47, de48, de49, de50, de51, de52, de53, de54, de55, de56, de57, de58, de59, de60, de61, de62, de63, de64, de65, de66, de67, de68, de69, de70, de71, de72, de73, de74, de75, de76, de77, de78, de79, de80, de81, de82, de83, de84, de85, de86, de87, de88, de89, de90, de91, de92, de93, de94, de95, de96, de97, de98, de99, de100, de101, de102, de103, de104, de105, de106, de107]  
    
    attrw1 = x[108]    
    attrw2 = x[109] 
    irulewt1 = x[110]      
    irulewt2 = x[111]  
    irulewt3 = x[112]
    irulewt4 = x[113] 
    irulewt5 = x[114] 
    irulewt6 = x[115]
    irulewt7 = x[116]
    irulewt8 = x[117]
    irulewt9 = x[118]
    irulewt10 = x[119]
    irulewt11 = x[120]
    irulewt12 = x[121]
    irulewt13 = x[122]
    irulewt14 = x[123]
    irulewt15 = x[124]
    irulewt16 = x[125]
    irulewt17 = x[126]
    irulewt18 = x[127]
    irulewt19 = x[128]
    irulewt20 = x[129]
    irulewt21 = x[130] 
    irulewt22 = x[131]
    irulewt23 = x[132]
    irulewt24 = x[133]
    irulewt25 = x[134]
    irulewt26 = x[135]
    irulewt27 = x[136]
    irulewt28 = x[137]
    irulewt29 = x[138]
    irulewt30 = x[139]
    irulewt31 = x[140]
    irulewt32 = x[141]
    irulewt33 = x[142]
    irulewt34 = x[143]
    irulewt35 = x[144]
    irulewt36 = x[145] 

    #transformInput1(384.5891688061617)  
    PMH = 0 + (x[146] * 499.4)
    PMUM = 0 + (x[147] * 499.4)  #500.4     
    PMM = 0 + (x[148] * 499.4)
    PMLM = 0 + (x[149] * 499.4)   #35.5
    PMSL = 0 + (x[150] * 499.4)
    PML = 0 + (x[151] * 499.4)  
    
    transformInput1(s,PMH,PMUM,PMM,PMLM,PMSL,PML)   
    transformInput2(c,PMH,PMUM,PMM,PMLM,PMSL,PML)    
    calculateMatchingDegreeBrbCnn(attrw1,attrw2,irulewt1,irulewt2,irulewt3,irulewt4,irulewt5,irulewt6,irulewt7,irulewt8,irulewt9,irulewt10,irulewt11,irulewt12,irulewt13,irulewt14,irulewt15,irulewt16,irulewt17,irulewt18,irulewt19,irulewt20,irulewt21,irulewt22,irulewt23,irulewt24,irulewt25,irulewt26,irulewt27,irulewt28,irulewt29,irulewt30,irulewt31,irulewt32,irulewt33,irulewt34,irulewt35,irulewt36)    
    showActivationWeight()
    updateBeliefDegree()  
    result = aggregateER_BrbCnn()
    return result       

def transformInput1(i,j,k,l,m,n,p):   
    global H1 
    global UM1  
    global M1   
    global LM1 
    global SL1
    global L1 
             
    PM_H = j
    PM_UM = k 
    PM_M = l
    PM_LM = m
    PM_SL = n 
    PM_L = p
          
    print("Inside transformInput1() Input is ",i,"PM_H ", PM_H,"PM_UM",PM_UM, "PM_M ",PM_M,"PM_LM", PM_LM,"PM_SL", PM_SL, "PM_L ",PM_L) 
        
    if (i >= PM_H): 
        H1 = 1 
        UM1 = 0
        M1 = 0
        LM1 = 0
        SL1 = 0
        L1 = 0

    elif (i == PM_UM):
        H1 = 0 
        UM1 = 1
        M1 = 0
        LM1 = 0
        SL1 = 0
        L1 = 0 

    elif (i == PM_M):
        H1 = 0
        UM1 = 0 
        M1 = 1
        LM1 = 0
        SL1 = 0
        L1 = 0
    
    elif (i == PM_LM):
        H1 = 0
        UM1 = 0 
        M1 = 0
        LM1 = 1
        SL1 = 0
        L1 = 0    

    elif (i == PM_SL):
        H1 = 0
        UM1 = 0 
        M1 = 0
        LM1 = 0
        SL1 = 1  
        L1 = 0        
    
    elif (i <= PM_L):
        H1 = 0
        UM1 = 0       
        M1 = 0
        LM1 = 0
        SL1 = 0
        L1 = 1    
    
    elif (i < PM_H) and (i > PM_UM): 
        UM1 = (PM_H-i)/(PM_H-PM_UM) 
        H1 = 1 - UM1
        M1 = 0.0
        L1 = 0.0         
        LM1 = 0.0
        SL1 = 0.0
   
    elif (i < PM_UM) and (i > PM_M): 
        M1 = (PM_UM-i)/(PM_UM-PM_M)   
        UM1 = 1 - M1
        H1 = 0.0
        L1 = 0.0 
        LM1 = 0.0
        SL1 = 0
 
    elif (i < PM_M) and (i > PM_LM): 
        LM1 = (PM_M-i)/(PM_M-PM_LM)   
        M1 = 1 - LM1 
        H1 = 0.0 
        UM1 = 0.0 
        L1 = 0.0
        SL1 = 0
        
    elif (i < PM_LM) and (i > PM_SL):
        SL1 = (PM_LM-i)/(PM_LM-PM_SL)   
        LM1 = 1 - SL1
        H1 = 0.0
        UM1 = 0.0
        M1 = 0.0 
        L1 = 0.0
      
    elif (i < PM_SL) and (i > PM_L): 
        L1 = (PM_SL-i)/(PM_SL-PM_L)   
        SL1 = 1 - L1
        H1 = 0.0
        UM1 = 0.0
        M1 = 0.0
        LM1 = 0.0
        
    print("Inside transformInput1(), H1", H1, "UM1 ", UM1, "M1 ",M1,"LM1 ", LM1, "SL1 ", SL1, "L1 ", L1)      
def transformInput2(i,j,k,l,m,n,p):
    global H2 
    global UM2  
    global M2   
    global LM2 
    global SL2
    global L2 
              
    PM_H = j 
    PM_UM = k 
    PM_M = l
    PM_LM = m
    PM_SL = n 
    PM_L = p
          
    print("Inside transformInput2() Input is ",i,"PM_H ", PM_H,"PM_UM",PM_UM, "PM_M ",PM_M,"PM_LM", PM_LM,"PM_SL", PM_SL, "PM_L ",PM_L) 
        
    if (i >= PM_H): 
        H2 = 1 
        UM2 = 0
        M2 = 0
        LM2 = 0
        SL2 = 0
        L2 = 0

    elif (i == PM_UM):
        H2 = 0 
        UM2 = 1
        M2 = 0
        LM2 = 0
        SL2 = 0
        L2 = 0 

    elif (i == PM_M):
        H2 = 0
        UM2 = 0 
        M2 = 1
        LM2 = 0
        SL2 = 0
        L2 = 0
    
    elif (i == PM_LM):
        H2 = 0
        UM2 = 0 
        M2 = 0
        LM2 = 1
        SL2 = 0
        L2 = 0    

    elif (i == PM_SL):
        H2 = 0
        UM2 = 0 
        M2 = 0
        LM2 = 0
        SL2 = 1  
        L2 = 0        
    
    elif (i <= PM_L):
        H2 = 0
        UM2 = 0       
        M2 = 0
        LM2 = 0
        SL2 = 0
        L2 = 1    
    
    elif (i < PM_H) and (i > PM_UM): 
        UM2 = (PM_H-i)/(PM_H-PM_UM) 
        H2 = 1 - UM2
        M2 = 0.0
        L2 = 0.0         
        LM2 = 0.0
        SL2 = 0.0
   
    elif (i < PM_UM) and (i > PM_M): 
        M2 = (PM_UM-i)/(PM_UM-PM_M)   
        UM2 = 1 - M2
        H2 = 0.0
        L2 = 0.0 
        LM2 = 0.0
        SL2 = 0
 
    elif (i < PM_M) and (i > PM_LM): 
        LM2 = (PM_M-i)/(PM_M-PM_LM)   
        M2 = 1 - LM2 
        H2 = 0.0 
        UM2 = 0.0 
        L2 = 0.0
        SL2 = 0
        
    elif (i < PM_LM) and (i > PM_SL):
        SL2 = (PM_LM-i)/(PM_LM-PM_SL)   
        LM2 = 1 - SL2
        H2 = 0.0
        UM2 = 0.0
        M2 = 0.0 
        L2 = 0.0
       
    elif (i < PM_SL) and (i > PM_L): 
        L2 = (PM_SL-i)/(PM_SL-PM_L)   
        SL2 = 1 - L2
        H2 = 0.0
        UM2 = 0.0
        M2 = 0.0
        LM2 = 0.0
        
    print("Inside transformInput1(), H2", H2, "UM2 ", UM2, "M2 ",M2,"LM2 ", LM2, "SL2 ", SL2, "L2 ", L2)      
def calculateMatchingDegreeBrbCnn(aw1,aw2,irw1,irw2,irw3,irw4,irw5,irw6,irw7,irw8,irw9,irw10,irw11,irw12,irw13,irw14,irw15,irw16,irw17,irw18,irw19,irw20,irw21,irw22,irw23,irw24,irw25,irw26,irw27,irw28,irw29,irw30,irw31,irw32,irw33,irw34,irw35,irw36):
    antattrw1 = aw1 
    antattrw2 = aw2
    global initialRuleWeight      
    initialRuleWeight = [irw1,irw2,irw3,irw4, irw5,irw6,irw7,irw8,irw9,irw10,irw11,irw12,irw13,irw14,irw15,irw16,irw17,irw18,irw19,irw20,irw21,irw22,irw23,irw24,irw25,irw26,irw27,irw28,irw29,irw30,irw31,irw32,irw33,irw34,irw35,irw36]
    increment = 0     
    global matchingDegree 
    matchingDegree = [1.51] * 36
  
    ti1 = [H1, UM1, M1, LM1, SL1, L1]  
    #print("ti1[0] is ")         
    #print(ti1[0])  
    #ti2 = array.array('f', [normalized_cnn_severe_degree, normalized_cnn_mild_degree, normalized_cnn_nominal_degree])
    ti2 = [H2, UM2, M2, LM2, SL2, L2] 
         
    for c in range(6):   
        for d in range(6):
            #print(ti1[c])  
            matchingDegree[increment] = initialRuleWeight[increment] * (ti1[c] ** antattrw1) * (ti2[d] ** antattrw2)     
            #trainedMatchingDegree[increment] = (ti1[c] ** relativeWeight) + (ti2[c] ** relativeWeight)
            increment +=1
    print("Inside calculateMatchingDegreeBrbCnn() relativeWeight1 ",antattrw1,"relativeWeight2 ",antattrw2)   
    #print("Inside calculateMatchingDegreeBrbCnn() best9 relativeWeight1 ",best[9]," best10 relativeWeight2 ",best[10])        
def showActivationWeight():   
    trace = 1            
    totalWeight = 0 
    totalActivationWeight = 0   
    global activationWeight  
    activationWeight = [1.51] * 36         
    for x in range(36):    
        totalWeight += matchingDegree[x]           
        
    ##Conj
    for counter in range(36):            
        inter = matchingDegree[counter]  
        activationWeight[counter] = inter/totalWeight      
    ##Conj          
def updateBeliefDegree():
    update = 0
    sumAntAttr1 = 1
    sumAntAttr2 = 1  
    
    if (H1 + UM1 + M1 + LM1 + SL1 + L1) < 1:
        sumAntAttr1 = H1 + UM1 + M1 + LM1 + SL1 + L1
        update = 1 
      
    if (H2 + UM2 + M2 + LM2 + SL2 + L2) < 1:
        sumAntAttr2 = H2 + UM2 + M2 + LM2 + SL2 + L2
        update = 1 
     
    if update == 1:
        beliefDegreeChangeLevel = (sumAntAttr1 + sumAntAttr2)/numberOfAntAttributes 

        for go in range(108):
            consequentBeliefDegree[go] = beliefDegreeChangeLevel * consequentBeliefDegree[go]
    else: 
        print ("No upgradation of belief degree required.")    
def aggregateER_BrbCnn():   
    parse = 0
    move1 = 0 
    move2 = 1  
    move3 = 2 
    action1 = 0
    action2 = 1
    action3 = 2 
    
    global ruleWiseBeliefDegreeSum 
    ruleWiseBeliefDegreeSum = [1.51] * 36
    
    part11 = 1.51
    part12 = 1.51
    part13 = 1.51
    
    part1 = 1.0
    part2 = 1.0
    value = 1.0
    meu = 1.0
    
    numeratorH1 = 1.0
    numeratorH2 = 1.0
    numeratorH = 1.0
    denominatorH1 = 1.0
    denominatorH = 1.0
    
    numeratorM1 = 1.0  
    numeratorM = 1.0
    
    numeratorL1 = 1.0
    numeratorL = 1.0
     
    utilityScoreH = 1.0
    utilityScoreM = 0.5
    utilityScoreL = 0.0
    crispValue = 1.0
    degreeOfIncompleteness = 1.0
    utilityMax = 1.0 
    utilityMin = 1.0
    utilityAvg = 1.0
    
    global aqi
    
    for s in range(108): 
        print("Inside aggregateER)BrbCNN() consequentBeliefDegree: ",consequentBeliefDegree[s])
     
    for t in range(36): 
        parse = t * 3   
        ruleWiseBeliefDegreeSum[t] = consequentBeliefDegree[parse] + consequentBeliefDegree[parse+1] + consequentBeliefDegree[parse+2]
 
    for rule in range(36):  
        part11 *= (activationWeight[rule] * consequentBeliefDegree[move1] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))         
        move1 += 3 
  
    for rule in range(36):
        part12 *= (activationWeight[rule] * consequentBeliefDegree[move2] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        move2 += 3 
 
    for rule in range(36):
        part13 *= (activationWeight[rule] * consequentBeliefDegree[move3] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        move3 += 3

    part1 = (part11 + part12 + part13)
    
    for rule in range(36):
        part2 *= (1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule])) 
    
    value = part1 - part2 
    
    meu = 1/value 
 
    for rule in range(36):
        numeratorH1 *= (activationWeight[rule] * consequentBeliefDegree[action1] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        action1 += 3

    for rule in range(36):
        numeratorH2 *= (1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))              
      
    numeratorH = meu * (numeratorH1 - numeratorH2) 
    
    for rule in range(36):  
        denominatorH1 *= (1 - activationWeight[rule])        
 
    denominatorH = 1 - (meu * denominatorH1)
    
    aggregatedBeliefDegreeH = (numeratorH/denominatorH)
    
    for rule in range(36):
        numeratorM1 *= (activationWeight[rule] * consequentBeliefDegree[action2] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        action2 += 3 

    numeratorM = meu * (numeratorM1 - numeratorH2) 
    aggregatedBeliefDegreeM = (numeratorM/denominatorH)  
    
    for rule in range(36):
        numeratorL1 *= (activationWeight[rule] * consequentBeliefDegree[action3] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        action3 += 3
     
    numeratorL = meu * (numeratorL1 - numeratorH2)
    aggregatedBeliefDegreeL = (numeratorL/denominatorH) 
    
    if (aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL) == 1:
        crispValue = (aggregatedBeliefDegreeH * utilityScoreH) + (aggregatedBeliefDegreeM * utilityScoreM) + (aggregatedBeliefDegreeL * utilityScoreL)
        brbH = aggregatedBeliefDegreeH
        brbM = aggregatedBeliefDegreeM
        brbL = aggregatedBeliefDegreeL       
        
        print ("\n BRB-CNN integrated Belief Degree for Hazardous AQI: ",aggregatedBeliefDegreeH,"\n")
        print ("\n BRB-CNN integrated Belief Degree for Unhealthy AQI: ",aggregatedBeliefDegreeM,"\n")
        print ("\n BRB-CNN integrated Belief Degree for Good AQI: ",aggregatedBeliefDegreeL,"\n")
        #cout << "brbH: " << brbH << " brbM: " << brbM << " brbL: " << brbL <<endl;    
 
    else:         
        degreeOfIncompleteness = 1 - (aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL)
        
        utilityMax = ((aggregatedBeliefDegreeH + degreeOfIncompleteness) * utilityScoreH + (aggregatedBeliefDegreeM*utilityScoreM) + (aggregatedBeliefDegreeL*utilityScoreL))
        
        utilityMin = (aggregatedBeliefDegreeH*utilityScoreH) + (aggregatedBeliefDegreeM*utilityScoreM) + (aggregatedBeliefDegreeL + degreeOfIncompleteness) * utilityScoreL
        
        utilityAvg = (utilityMax + utilityMin)/2  
         
        print ("BRB-CNN integrated Belief Degrees considering degree of Incompleteness: ")  
        
        finalAggregatedBeliefDegreeH = aggregatedBeliefDegreeH/(aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL)  
         
        finalAggregatedBeliefDegreeM = aggregatedBeliefDegreeM/(aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL)
        
        finalAggregatedBeliefDegreeL = aggregatedBeliefDegreeL/(aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL) 
          
        brbH = finalAggregatedBeliefDegreeH
        brbM = finalAggregatedBeliefDegreeM 
        brbL = finalAggregatedBeliefDegreeL       
            
        if (finalAggregatedBeliefDegreeH > finalAggregatedBeliefDegreeM) and (finalAggregatedBeliefDegreeH > finalAggregatedBeliefDegreeL):
            aqi = (201 + 299*finalAggregatedBeliefDegreeH) + ((200*finalAggregatedBeliefDegreeM)/2)
            print ("AQI predicted by BRB-CNN:",aqi)    
            
        elif (finalAggregatedBeliefDegreeL > finalAggregatedBeliefDegreeM) and (finalAggregatedBeliefDegreeL > finalAggregatedBeliefDegreeH): 
            aqi = (100*(1 - finalAggregatedBeliefDegreeL)) + ((200*finalAggregatedBeliefDegreeM)/2) 
            print ("AQI predicted by BRB-CNN:",aqi)
  
        elif (finalAggregatedBeliefDegreeM > finalAggregatedBeliefDegreeH) and (finalAggregatedBeliefDegreeM > finalAggregatedBeliefDegreeL):
            if finalAggregatedBeliefDegreeH > finalAggregatedBeliefDegreeL:
                aqi = (101 + 99*finalAggregatedBeliefDegreeM) + ((500*finalAggregatedBeliefDegreeH)/2)
                print ("AQI predicted by BRB-CNN: ",aqi)
      
            elif (finalAggregatedBeliefDegreeL > finalAggregatedBeliefDegreeH):   
                aqi = (101 + 99*finalAggregatedBeliefDegreeM) + ((100*finalAggregatedBeliefDegreeL)/2)
                print ("AQI predicted by BRB-CNN:",aqi)  
          
        print("aqi ",aqi)     
                
        if aqi >= 301: 
            aqi6 = (aqi- 301)/199.0  
 
        elif (aqi >= 201)and (aqi <= 300.9999999999): 
            aqi6 = (aqi- 201)/99.0     

        elif (aqi >= 151)and (aqi <= 200.9999999999):
            aqi6 = (aqi- 151)/49.0 

        elif((aqi >= 101)and (aqi <= 150.9999999999)): 
            aqi6 = (aqi- 101)/49.0   

        elif((aqi >= 51)and (aqi <= 100.9999999999)): 
            aqi6 = (aqi- 51)/49.0    
 
        elif(aqi <= 50.9999999999):    
            aqi6 = (aqi/49.0)    
              
        print("aqi6 ",aqi6) 
        print ("BRB-CNN integrated Belief Degree for Hazardous AQI:",finalAggregatedBeliefDegreeH*aqi6)   
        print ("BRB-CNN integrated Belief Degree for Very Unhealthy AQI:",finalAggregatedBeliefDegreeH*(1-aqi6)) 
        print ("BRB-CNN integrated Belief Degree for Unhealthy AQI: ",finalAggregatedBeliefDegreeM*aqi6)
        print ("BRB-CNN integrated Belief Degree for Unhealthy (Sensitive Groups) AQI:",finalAggregatedBeliefDegreeM*(1-aqi6)) 
        print ("BRB-CNN integrated Belief Degree for Moderate AQI:",finalAggregatedBeliefDegreeL*aqi6) 
        print ("BRB-CNN integrated Belief Degree for Good AQI:",finalAggregatedBeliefDegreeL*(1-aqi6))
        
        return aqi