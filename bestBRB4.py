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
    
def deRuleBase(s,c,best):  
    global consequentBeliefDegree
    temp_consequentBeliefDegree = [best[0], best[1], best[2], best[3], best[4], best[5], best[6], best[7], best[8], best[9], best[10], best[11], best[12], best[13], best[14], best[15], best[16], best[17], best[18], best[19], best[20], best[21], best[22], best[23], best[24], best[25], best[26], best[27], best[28], best[29], best[30], best[31], best[32], best[33], best[34], best[35], best[36], best[37], best[38], best[39], best[40], best[41], best[42], best[43], best[44], best[45], best[46], best[47]] 
    
    de0 = best[0]/(best[0] + best[1] + best[2])
    de1 = best[1]/(best[0] + best[1] + best[2])
    de2 = best[2]/(best[0] + best[1] + best[2])
    de3 = best[3]/(best[3] + best[4] + best[5]) 
    de4 = best[4]/(best[3] + best[4] + best[5])
    de5 = best[5]/(best[3] + best[4] + best[5])
    de6 = best[6]/(best[6] + best[7] + best[8])
    de7 = best[7]/(best[6] + best[7] + best[8])
    de8 = best[8]/(best[6] + best[7] + best[8])
    de9 = best[9]/(best[9] + best[10] + best[11])
    de10 = best[10]/(best[9] + best[10] + best[11])
    de11 = best[11]/(best[9] + best[10] + best[11])   
    de12 = best[12]/(best[12] + best[13] + best[14])   
    de13 = best[13]/(best[12] + best[13] + best[14])
    de14 = best[14]/(best[12] + best[13] + best[14])   
    de15 = best[15]/(best[15] + best[16] + best[17])
    de16 = best[16]/(best[15] + best[16] + best[17])
    de17 = best[17]/(best[15] + best[16] + best[17])   
    de18 = best[18]/(best[18] + best[19] + best[20])   
    de19 = best[19]/(best[18] + best[19] + best[20])
    de20 = best[20]/(best[18] + best[19] + best[20])   
    de21 = best[21]/(best[21] + best[22] + best[23])   
    de22 = best[22]/(best[21] + best[22] + best[23])
    de23 = best[23]/(best[21] + best[22] + best[23])
    de24 = best[24]/(best[24] + best[25] + best[26])   
    de25 = best[25]/(best[24] + best[25] + best[26])
    de26 = best[26]/(best[24] + best[25] + best[26])
    de27 = best[27]/(best[27] + best[28] + best[29])
    de28 = best[28]/(best[27] + best[28] + best[29])
    de29 = best[29]/(best[27] + best[28] + best[29])
    de30 = best[30]/(best[30] + best[31] + best[32])   
    de31 = best[31]/(best[30] + best[31] + best[32])
    de32 = best[32]/(best[30] + best[31] + best[32])
    de33 = best[33]/(best[33] + best[34] + best[35])
    de34 = best[34]/(best[33] + best[34] + best[35])
    de35 = best[35]/(best[33] + best[34] + best[35])
    de36 = best[36]/(best[36] + best[37] + best[38])
    de37 = best[37]/(best[36] + best[37] + best[38])
    de38 = best[38]/(best[36] + best[37] + best[38])
    de39 = best[39]/(best[39] + best[40] + best[41])   
    de40 = best[40]/(best[39] + best[40] + best[41])
    de41 = best[41]/(best[39] + best[40] + best[41])
    de42 = best[42]/(best[42] + best[43] + best[44])   
    de43 = best[43]/(best[42] + best[43] + best[44])
    de44 = best[44]/(best[42] + best[43] + best[44])
    de45 = best[45]/(best[45] + best[46] + best[47])   
    de46 = best[46]/(best[45] + best[46] + best[47])
    de47 = best[47]/(best[45] + best[46] + best[47])
        
    consequentBeliefDegree = [de0, de1, de2, de3, de4, de5, de6, de7, de8, de9, de10, de11, de12, de13, de14, de15, de16, de17, de18, de19, de20, de21, de22, de23, de24, de25, de26, de27, de28, de29, de30, de31, de32, de33, de34, de35, de36, de37, de38, de39, de40, de41, de42, de43, de44, de45, de46, de47]

    attrw1 = best[48]  
    attrw2 = best[49]
    irulewt1 = best[50]      
    irulewt2 = best[51]  
    irulewt3 = best[52]  
    irulewt4 = best[53]
    irulewt5 = best[54]
    irulewt6 = best[55]
    irulewt7 = best[56]
    irulewt8 = best[57]
    irulewt9 = best[58]
    irulewt10 = best[59]
    irulewt11 = best[60]
    irulewt12 = best[61]
    irulewt13 = best[62]
    irulewt14 = best[63]
    irulewt15 = best[64]
    irulewt16 = best[65]  
     
    H_PM = 0 + (best[66] * 499.4)     #500.4     
    UM_PM = 0 + (best[67] * 499.4)
    M_PM = 0 + (best[68] * 499.4)     #35.5 
    L_PM = 0 + (best[69] * 499.4)     #0  
    transformInput1(s,H_PM,UM_PM,M_PM,L_PM)      
    transformInput2(c,H_PM,UM_PM,M_PM,L_PM)     
    calculateMatchingDegreeBrbCnn(attrw1,attrw2, irulewt1, irulewt2, irulewt3, irulewt4, irulewt5, irulewt6, irulewt7, irulewt8, irulewt9, irulewt10, irulewt11, irulewt12, irulewt13, irulewt14, irulewt15, irulewt16)    
    showActivationWeight()  
    updateBeliefDegree()       
    deResult = aggregateER_BrbCnn()   
    return deResult

def transformInput1(i,j,k,l,m):   
    global H1
    global UM1  
    global M1 
    global L1 
            
    PM_H = j
    PM_UM = k
    PM_M = l 
    PM_L = m 
        
    print("Inside transformInput1() Input is ",i,"PM_H ", PM_H,"PM_UM",PM_UM, "PM_M ",PM_M," PM_L ",PM_L)
      
    if (i >= PM_H): 
        H1 = 1
        UM1 = 0
        M1 = 0
        L1 = 0

    elif (i == PM_UM):
        H1 = 0 
        UM1 = 1
        M1 = 0
        L1 = 0 

    elif (i == PM_M):
        H1 = 0
        UM1 = 0 
        M1 = 1
        L1 = 0
   
    elif (i <= PM_L):
        H1 = 0
        UM1 = 0       
        M1 = 0
        L1 = 1   
   
    elif (i < PM_H) and (i > PM_UM): 
        UM1 = (PM_H-i)/(PM_H-PM_UM) 
        H1 = 1 - UM1
        M1 = 0.0
        L1 = 0.0         
 
    elif (i < PM_UM) and (i > PM_M): 
        M1 = (PM_UM-i)/(PM_UM-PM_M)   
        UM1 = 1 - M1
        H1 = 0.0
        L1 = 0.0              
     
    elif (i < PM_M) and (i > PM_L): 
        L1 = (PM_M-i)/(PM_M-PM_L)   
        M1 = 1 - L1
        H1 = 0.0
        UM1 = 0.0              
    print("Inside transformInput1(), H1", H1, "UM1 ", UM1, "M1 ",M1,"L1 ", L1)

def transformInput2(i,j,k,l,m):
    global H2
    global UM2  
    global M2 
    global L2 
             
    PM_H = j
    PM_UM = k
    PM_M = l 
    PM_L = m      
        
    print("Inside transformInput1() Input is ",i,"PM_H ", PM_H,"PM_UM",PM_UM, "PM_M ",PM_M," PM_L ",PM_L)
      
    if (i >= PM_H): 
        H2 = 1
        UM2 = 0
        M2 = 0
        L2 = 0

    elif (i == PM_UM):
        H2 = 0 
        UM2 = 1
        M2 = 0
        L2 = 0 

    elif (i == PM_M):
        H2 = 0
        UM2 = 0 
        M2 = 1
        L2 = 0
   
    elif (i <= PM_L):
        H2 = 0
        UM2 = 0       
        M2 = 0
        L2 = 1   
   
    elif (i < PM_H) and (i > PM_UM): 
        UM2 = (PM_H-i)/(PM_H-PM_UM) 
        H2 = 1 - UM2
        M2 = 0.0
        L2 = 0.0         
 
    elif (i < PM_UM) and (i > PM_M): 
        M2 = (PM_UM-i)/(PM_UM-PM_M)   
        UM2 = 1 - M2
        H2 = 0.0
        L2 = 0.0              
     
    elif (i < PM_M) and (i > PM_L): 
        L2 = (PM_M-i)/(PM_M-PM_L)   
        M2 = 1 - L2     
        H2 = 0.0
        UM2 = 0.0              
    print("Inside transformInput1(), H2", H2, "UM2 ", UM2, "M2 ",M2,"L2 ", L2) 

def calculateMatchingDegreeBrbCnn(aw1,aw2,irw1,irw2,irw3,irw4,irw5,irw6,irw7,irw8,irw9,irw10,irw11,irw12,irw13,irw14,irw15,irw16):
    antattrw1 = aw1 
    antattrw2 = aw2
    global initialRuleWeight      
    initialRuleWeight = [irw1,irw2,irw3,irw4,irw5,irw6,irw7,irw8,irw9,irw10,irw11,irw12,irw13,irw14,irw15,irw16]     
    increment = 0     
    global matchingDegree 
    matchingDegree = [1.51] * 16
    
    ti1 = [H1, UM1, M1, L1]  
    #print("ti1[0] is ")         
    #print(ti1[0])  
    #ti2 = array.array('f', [normalized_cnn_severe_degree, normalized_cnn_mild_degree, normalized_cnn_nominal_degree])
    ti2 = [H2, UM2, M2, L2] 

    for c in range(4):   
        for d in range(4):
            #print(ti1[c])  
            matchingDegree[increment] = initialRuleWeight[increment] * (ti1[c] ** antattrw1) * (ti2[d] ** antattrw2)     
            #trainedMatchingDegree[increment] = (ti1[c] ** relativeWeight) + (ti2[c] ** relativeWeight)
            increment +=1    
    print("Inside calculateMatchingDegreeBrbCnn() relativeWeight1 ",antattrw1,"relativeWeight2 ",antattrw2)   
       
def showActivationWeight():   
    trace = 1           
    totalWeight = 0 
    totalActivationWeight = 0   
    global activationWeight  
    activationWeight = [1.51] * 16         
    for x in range(16):    
        totalWeight += matchingDegree[x]           
        
    ##Conj
    for counter in range(16):            
        inter = matchingDegree[counter]  
        activationWeight[counter] = inter/totalWeight      
    ##Conj         
    

def updateBeliefDegree():
    update = 0
    sumAntAttr1 = 1
    sumAntAttr2 = 1  
    
    if (H1 + UM1 + M1 + L1) < 1:
        sumAntAttr1 = H1 + UM1 + M1 + L1
        update = 1 
      
    if (H2 + UM2 + M2 + L2) < 1:
        sumAntAttr2 = H2 + UM2 + M2 + L2
        update = 1 
     
    if update == 1:
        beliefDegreeChangeLevel = (sumAntAttr1 + sumAntAttr2)/numberOfAntAttributes 

        for go in range(48):
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
    ruleWiseBeliefDegreeSum = [1.51] * 16
    
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
    
    for s in range(48): 
        print("Inside aggregateER)BrbCNN() consequentBeliefDegree: ",consequentBeliefDegree[s])
     
    for t in range(16): 
        parse = t * 3   
        ruleWiseBeliefDegreeSum[t] = consequentBeliefDegree[parse] + consequentBeliefDegree[parse+1] + consequentBeliefDegree[parse+2]
 
    for rule in range(16):  
        part11 *= (activationWeight[rule] * consequentBeliefDegree[move1] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))         
        move1 += 3 
  
    for rule in range(16):
        part12 *= (activationWeight[rule] * consequentBeliefDegree[move2] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        move2 += 3   
 
    for rule in range(16):
        part13 *= (activationWeight[rule] * consequentBeliefDegree[move3] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        move3 += 3

    part1 = (part11 + part12 + part13)
    
    for rule in range(16):
        part2 *= (1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule])) 
    
    value = part1 - part2 
     
    meu = 1/value 
 
    for rule in range(16):
        numeratorH1 *= (activationWeight[rule] * consequentBeliefDegree[action1] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        action1 += 3

    for rule in range(16):
        numeratorH2 *= (1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))              
      
    numeratorH = meu * (numeratorH1 - numeratorH2) 
    
    for rule in range(16):  
        denominatorH1 *= (1 - activationWeight[rule])        
 
    denominatorH = 1 - (meu * denominatorH1)
    
    aggregatedBeliefDegreeH = (numeratorH/denominatorH)
    
    for rule in range(16):
        numeratorM1 *= (activationWeight[rule] * consequentBeliefDegree[action2] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        action2 += 3 

    numeratorM = meu * (numeratorM1 - numeratorH2) 
    aggregatedBeliefDegreeM = (numeratorM/denominatorH)  
    
    for rule in range(16):
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