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
    temp_deConsequentBeliefDegree = [best[0],best[1],best[2],best[3],best[4],best[5],best[6],best[7],best[8],best[9],best[10],best[11]]  
    
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
        
    consequentBeliefDegree = [de0, de1, de2, de3, de4, de5, de6, de7, de8, de9, de10, de11] 
      
    H_PM = 0 + (best[18] * 499.4)     #500.4     
    L_PM = 0 + (best[19] * 499.4)     #0     
    transformInput1(s,H_PM,L_PM)      
    transformInput2(c,H_PM,L_PM)    
    calculateMatchingDegreeBrbCnn(best[12], best[13], best[14], best[15], best[16], best[17]) 
    showActivationWeight()  
    updateBeliefDegree()               
    deResult = aggregateER_BrbCnn()   
    return deResult

def transformInput1(i,j,k):   
    global H1 
    global L1  
             
    PM_H = j  
    PM_L = k
       
    print("Inside transformInput1() Input is ",i,"PM_H ", PM_H," PM_L ",PM_L)
       
    if (i >= PM_H):  
        H1 = 1 
        L1 = 0    
        
    elif (i <= PM_L):   
        H1 = 0
        L1 = 1
       
    elif (i < PM_H) and (i > PM_L):
        L1 = (PM_H-i)/(PM_H-PM_L)
        H1 = 1 - L1
 
    print("Inside transformInput1(), H1", H1, "L1 ", L1) 

def transformInput2(i,j,k):
    global H2 
    global L2  
              
    PM_H = j  
    PM_L = k
       
    print("Inside transformInput2() Input is ",i,"PM_H ", PM_H," PM_L ",PM_L)
       
    if (i >= PM_H):    
        H2 = 1 
        L2 = 0    
        
    elif (i <= PM_L):   
        H2 = 0
        L2 = 1 
       
    elif (i < PM_H) and (i > PM_L):
        L2 = (PM_H-i)/(PM_H-PM_L)
        H2 = 1 - L2 

    print("Inside transformInput1(), H2", H2, "L2 ", L2)  

def calculateMatchingDegreeBrbCnn(aw1,aw2,irw1,irw2,irw3, irw4):
    antattrw1 = aw1 
    antattrw2 = aw2
    global initialRuleWeight      
    initialRuleWeight = [irw1, irw2, irw3, irw4]     
    increment = 0     
    global matchingDegree 
    matchingDegree = [1.51, 1.51, 1.51, 1.51]
     
    global trainedMatchingDegree
    trainedMatchingDegree = [1.51, 1.51, 1.51, 1.51]  
  
    ti1 = [H1, L1]  
    ti2 = [H2, L2] 
    
    for c in range(2):   
        for d in range(2):
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
    activationWeight = [1.51] * 4       
    for x in range(4):    
        totalWeight += matchingDegree[x]           
    
    for counter in range(4):            
        inter = matchingDegree[counter]  
        activationWeight[counter] = inter/totalWeight      
        
def updateBeliefDegree():
    update = 0
    sumAntAttr1 = 1
    sumAntAttr2 = 1  
    
    if (H1 + L1) < 1:
        sumAntAttr1 = H1 + L1
        update = 1 
      
    if (H2 + L2) < 1:
        sumAntAttr2 = H2 + L2
        update = 1 
     
    if update == 1:
        beliefDegreeChangeLevel = (sumAntAttr1 + sumAntAttr2)/numberOfAntAttributes 

        for go in range(12):
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
    ruleWiseBeliefDegreeSum = [1.51, 1.51, 1.51, 1.51]
    
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
    
    for s in range(12): 
        print("Inside aggregateER)BrbCNN() consequentBeliefDegree: ",consequentBeliefDegree[s])
     
    for t in range(4): 
        parse = t * 3   
        ruleWiseBeliefDegreeSum[t] = consequentBeliefDegree[parse] + consequentBeliefDegree[parse+1] + consequentBeliefDegree[parse+2]
 
    for rule in range(4):  
        part11 *= (activationWeight[rule] * consequentBeliefDegree[move1] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))         
        move1 += 3 
  
    for rule in range(4):
        part12 *= (activationWeight[rule] * consequentBeliefDegree[move2] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        move2 += 3   
 
    for rule in range(4):
        part13 *= (activationWeight[rule] * consequentBeliefDegree[move3] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        move3 += 3

    part1 = (part11 + part12 + part13)
    
    for rule in range(4):
        part2 *= (1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule])) 
    
    value = part1 - part2 
     
    meu = 1/value 
 
    for rule in range(4):
        numeratorH1 *= (activationWeight[rule] * consequentBeliefDegree[action1] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        action1 += 3

    for rule in range(4):
        numeratorH2 *= (1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))              
      
    numeratorH = meu * (numeratorH1 - numeratorH2) 
    
    for rule in range(4):  
        denominatorH1 *= (1 - activationWeight[rule])        
 
    denominatorH = 1 - (meu * denominatorH1)
    
    aggregatedBeliefDegreeH = (numeratorH/denominatorH)
    
    for rule in range(4):
        numeratorM1 *= (activationWeight[rule] * consequentBeliefDegree[action2] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        action2 += 3 

    numeratorM = meu * (numeratorM1 - numeratorH2) 
    aggregatedBeliefDegreeM = (numeratorM/denominatorH)  
    
    for rule in range(4):
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