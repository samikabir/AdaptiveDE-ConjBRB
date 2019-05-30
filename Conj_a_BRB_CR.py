#PM_H = 500.4
#PM_M = 35.5
#PM_L = 0.0
 
AQI_H = 500.0
AQI_M = 101.0  
AQI_L = 0.0    

numberOfAntAttributes = 2
#relativeWeight = 1.0   
aqi1 = 1.0
aqi2 = 1.0  
aqi3 = 1.0
aqi4 = 1.0  
aqi5 = 1.0      

def ruleBase(d11, d12):
    global consequentBeliefDegree

    consequentBeliefDegree = [1, 0, 0, 0.5, 0.5, 0, 0, 1, 0, 0.5, 0.5, 0, 0, 1, 0, 0, 0.5, 0.5, 0, 1, 0, 0, 0.5, 0.5, 0, 0, 1]      
    
    #consequentBeliefDegree = [cbd_0, cbd_1, cbd_2, cbd_3, cbd_4, cbd_5, cbd_6, cbd_7, cbd_8]   
    #for u in range(27):    
    #    print("DE Trained Normalized Belief Degree ",consequentBeliefDegree[u])   
    #transformInput1(384.5891688061617)  
     
    big = 1.0
    medium = 0.5
    small = 0.0  
    
    transformInput1(d11,big,medium,small)           
    transformInput2(d12)    
    calculateMatchingDegreeBrbCnn(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1)  
    showActivationWeight()  
    updateBeliefDegree()        
    cross_factor = aggregateER_BrbCnn()   
    return cross_factor
  
def transformInput1(i,j,k,l):   
    global H1 
    global M1 
    global L1 
            
    PM_H = j
    PM_M = k 
    PM_L = l 
       
    print("Inside transformInput1() Input is ",i,"big ", PM_H,"medium ",PM_M," small ",PM_L)
      
    if (i >= PM_H): 
        H1 = 1 
        M1 = 0
        L1 = 0  

    elif (i == PM_M):
        H1 = 0 
        M1 = 1
        L1 = 0
 
    elif (i <= PM_L):
        H1 = 0
        M1 = 0
        L1 = 1
       
    elif (i <= PM_H) and (i >= PM_M):
        M1 = (PM_H-i)/(PM_H-PM_M)
        H1 = 1 - M1
        L1 = 0.0 

    elif (i <= PM_M) and (i >= PM_L):
        L1 = (PM_M-i)/(PM_M-PM_L)
        M1 = 1 - L1  
        H1 = 0.0
    print("Inside transformInput1(), H1", H1, "M1 ",M1,"L1 ", L1)

def transformInput2(i): 
    global H2   
    global M2 
    global L2
            
    PM_H = 1.0
    PM_M = 0.50
    PM_L = 0.0
       
    print("Inside transformInput2() Input is ",i)  
       
    if (i >= PM_H): 
        H2 = 1 
        M2 = 0 
        L2 = 0 

    elif (i == PM_M):
        H2 = 0 
        M2 = 1 
        L2 = 0
  
    elif (i <= PM_L):
        H2 = 0
        M2 = 0
        L2 = 1 
        
    elif (i <= PM_H) and (i >= PM_M):
        M2 = (PM_H-i)/(PM_H-PM_M)
        H2 = 1 - M2
        L2 = 0.0 

    elif (i <= PM_M) and (i >= PM_L):
        L2 = (PM_M-i)/(PM_M-PM_L)
        M2 = 1 - L2  
        H2 = 0.0   
    
    print("Inside transformInput2(), H2", H2, "M2 ",M2,"L2 ", L2)    

def calculateMatchingDegreeBrbCnn(aw1,aw2,irw1,irw2,irw3, irw4, irw5, irw6, irw7, irw8, irw9): 
    antattrw1 = aw1 
    antattrw2 = aw2 
    global initialRuleWeight       
    initialRuleWeight = [irw1, irw2, irw3, irw4, irw5, irw6, irw7, irw8, irw9]     
    increment = 0     
    global matchingDegree 
    matchingDegree = [1.51] * 9  

    ti1 = [H1, M1, L1]  
    #print("ti1[0] is ")          
    #print(ti1[0])  
    #ti2 = array.array('f', [normalized_cnn_severe_degree, normalized_cnn_mild_degree, normalized_cnn_nominal_degree])
    ti2 = [H2, M2, L2] 
    ## Conj 
    for c in range(3): 
        for d in range(3):
            #print(ti1[c])
            matchingDegree[increment] = initialRuleWeight[increment] * (ti1[c] ** antattrw1) * (ti2[c] ** antattrw2)    
            #trainedMatchingDegree[increment] = (ti1[c] ** relativeWeight) + (ti2[c] ** relativeWeight)
            increment +=1  
    ##Conj    
    print("Inside calculateMatchingDegreeBrbCnn() relativeWeight1 ",antattrw1,"relativeWeight2 ",antattrw2)   
    #print("Inside calculateMatchingDegreeBrbCnn() best9 relativeWeight1 ",best[9]," best10 relativeWeight2 ",best[10])     
def showActivationWeight():    
    trace = 1           
    totalWeight = 0 
    totalActivationWeight = 0   
    global activationWeight  
    activationWeight = [1.51] * 9        
    temp_activationWeight = [1.57, 1.81, 1.92]     
    for x in range(9):    
        totalWeight += matchingDegree[x]           
        
    ##Conj
    for counter in range(9):            
        inter = matchingDegree[counter]    
        activationWeight[counter] = inter/totalWeight          
    ##Conj 
        
def takeCnnOutput():
    global normalized_cnn_severe_degree 
    global normalized_cnn_mild_degree 
    global normalized_cnn_nominal_degree
    
    parser = 0
    #f = open("cnn_prediction.txt", "r") #cnn output
    f = open("cnn_prediction1.txt", "r") #severe 408       
    #f = open("cnn_prediction2.txt", "r") #nominal 36
    #f = open("cnn_prediction3.txt", "r") #mild 117
    if f.mode == 'r':
        #print("reading cnn_prediction.txt file \n") 
        f1 = f.readlines()
         
        for line in f1:  
            if parser == 0: 
                cnn_mild = line
            elif parser == 1:
                cnn_nominal = line
            else: 
                cnn_severe = line
                
            parser +=1    
        
        f.close()    
    else:
        print("Unable to open the file.");
            
    a = float(cnn_mild)
    b = float(cnn_nominal) 
    c = float(cnn_severe)     
    
    mild_degree = a/100    
    nominal_degree = b/100 
    severe_degree = c/100
    
    sum_degree = severe_degree + mild_degree + nominal_degree
  
    normalized_cnn_severe_degree = severe_degree/sum_degree
    normalized_cnn_mild_degree = mild_degree/sum_degree      
    normalized_cnn_nominal_degree = nominal_degree/sum_degree       
    
    if ((normalized_cnn_severe_degree > normalized_cnn_mild_degree) and (normalized_cnn_severe_degree > normalized_cnn_nominal_degree)):
        cnn_pm25 = (150.5 + 349.9*normalized_cnn_severe_degree) + ((150.4*normalized_cnn_mild_degree)/2)
        print ("PM2.5 computed by CNN: ",cnn_pm25," µg/m3")  

    elif ((normalized_cnn_nominal_degree > normalized_cnn_mild_degree) and (normalized_cnn_nominal_degree > normalized_cnn_severe_degree)):       
        cnn_pm25 = (35.4*(1 - normalized_cnn_nominal_degree)) + ((150.4*normalized_cnn_mild_degree)/2)            
        print ("PM2.5 computed by CNN: ",cnn_pm25," µg/m3")   

    elif ((normalized_cnn_mild_degree > normalized_cnn_severe_degree) and (normalized_cnn_mild_degree > normalized_cnn_nominal_degree)):    
        if normalized_cnn_severe_degree > normalized_cnn_nominal_degree: 
            cnn_pm25 = (35.5 + 114.9*normalized_cnn_mild_degree) + ((500.4*normalized_cnn_severe_degree)/2)
            print ("PM2.5 computed by CNN: ",cnn_pm25," µg/m3")  
            
        elif (normalized_cnn_nominal_degree > normalized_cnn_severe_degree): 
            cnn_pm25 = (35.5 + 114.9*normalized_cnn_mild_degree) + ((35.4*normalized_cnn_nominal_degree)/2)     
            print ("PM2.5 computed by CNN: ",cnn_pm25," µg/m3")

def updateBeliefDegree(): 
    update = 0
    sumAntAttr1 = 1
    sumAntAttr2 = 1  
    
    if (H1 + M1 + L1) < 1:
        sumAntAttr1 = H1 + M1 + L1
        update = 1 
      
    if (H2 + M2 + L2) < 1:
        sumAntAttr2 = H2 + M2 + L2
        update = 1 
     
    if update == 1:
        beliefDegreeChangeLevel = (sumAntAttr1 + sumAntAttr2)/numberOfAntAttributes 

        for go in range(27): 
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
    ruleWiseBeliefDegreeSum = [1.51] * 9 
    
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
    
    #for s in range(27): 
    #    print("Inside aggregateER)BrbCNN() consequentBeliefDegree: ",consequentBeliefDegree[s])
     
    for t in range(9): 
        parse = t * 3   
        ruleWiseBeliefDegreeSum[t] = consequentBeliefDegree[parse] + consequentBeliefDegree[parse+1] + consequentBeliefDegree[parse+2]
 
    for rule in range(9):  
        part11 *= (activationWeight[rule] * consequentBeliefDegree[move1] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))         
        move1 += 3 
  
    for rule in range(9):
        part12 *= (activationWeight[rule] * consequentBeliefDegree[move2] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        move2 += 3 
 
    for rule in range(9):
        part13 *= (activationWeight[rule] * consequentBeliefDegree[move3] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        move3 += 3

    part1 = (part11 + part12 + part13)
    
    for rule in range(9):
        part2 *= (1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule])) 
    
    value = part1 - part2 
    
    meu = 1/value 
 
    for rule in range(9):
        numeratorH1 *= (activationWeight[rule] * consequentBeliefDegree[action1] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        action1 += 3

    for rule in range(9):
        numeratorH2 *= (1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))              
      
    numeratorH = meu * (numeratorH1 - numeratorH2)   
    
    for rule in range(9):  
        denominatorH1 *= (1 - activationWeight[rule])        
 
    denominatorH = 1 - (meu * denominatorH1)
    
    aggregatedBeliefDegreeH = (numeratorH/denominatorH)
    
    for rule in range(9):
        numeratorM1 *= (activationWeight[rule] * consequentBeliefDegree[action2] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        action2 += 3 

    numeratorM = meu * (numeratorM1 - numeratorH2) 
    aggregatedBeliefDegreeM = (numeratorM/denominatorH)  
    
    for rule in range(9):
        numeratorL1 *= (activationWeight[rule] * consequentBeliefDegree[action3] + 1 - (activationWeight[rule] * ruleWiseBeliefDegreeSum[rule]))        
        action3 += 3
     
    numeratorL = meu * (numeratorL1 - numeratorH2)
    aggregatedBeliefDegreeL = (numeratorL/denominatorH) 
    
    if (aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL) == 1:
        crispValue = (aggregatedBeliefDegreeH * utilityScoreH) + (aggregatedBeliefDegreeM * utilityScoreM) + (aggregatedBeliefDegreeL * utilityScoreL)
        brbH = aggregatedBeliefDegreeH
        brbM = aggregatedBeliefDegreeM
        brbL = aggregatedBeliefDegreeL         
        
        print ("\n Aggregated Belief Degree for Big CR: ",aggregatedBeliefDegreeH,"\n")
        print ("\n Aggregated Belief Degree for Medium CR: ",aggregatedBeliefDegreeM,"\n")  
        print ("\n Aggregated Belief Degree for Small CR: ",aggregatedBeliefDegreeL,"\n") 
        #cout << "brbH: " << brbH << " brbM: " << brbM << " brbL: " << brbL <<endl;
        cr = (1 * aggregatedBeliefDegreeH) + (0.75 * aggregatedBeliefDegreeM) + (0.1 * aggregatedBeliefDegreeL)
        
        print("Final CR value under complete assessment is", cr) 
 
    else:         
        degreeOfIncompleteness = 1 - (aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL)
        
        utilityMax = ((aggregatedBeliefDegreeH + degreeOfIncompleteness) * utilityScoreH + (aggregatedBeliefDegreeM*utilityScoreM) + (aggregatedBeliefDegreeL*utilityScoreL))
        
        utilityMin = (aggregatedBeliefDegreeH*utilityScoreH) + (aggregatedBeliefDegreeM*utilityScoreM) + (aggregatedBeliefDegreeL + degreeOfIncompleteness) * utilityScoreL
        
        utilityAvg = (utilityMax + utilityMin)/2   
         
        print ("Aggregated CR Belief Degrees considering degree of Incompleteness: ")  
        
        finalAggregatedBeliefDegreeH = aggregatedBeliefDegreeH/(aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL)  
         
        finalAggregatedBeliefDegreeM = aggregatedBeliefDegreeM/(aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL) 
        
        finalAggregatedBeliefDegreeL = aggregatedBeliefDegreeL/(aggregatedBeliefDegreeH + aggregatedBeliefDegreeM + aggregatedBeliefDegreeL)  
           
        brbH = finalAggregatedBeliefDegreeH
        brbM = finalAggregatedBeliefDegreeM 
        brbL = finalAggregatedBeliefDegreeL        
        
        cr = (1 * finalAggregatedBeliefDegreeH) + (0.75 * finalAggregatedBeliefDegreeM) + (0.1 * finalAggregatedBeliefDegreeL)
         
        print("Final CR value under incomplete assessment is", cr) 
        
    return cr  
 
#def getAQI(x):     
#    cbd_de0 = x[0]
#    cbd_de1 = x[1] 
#    cbd_de2 = x[2]
#    ruleBase()
     
    #aqi = x[0] + x[1] + x[2] + x[3]   
    #print("Diff Evo BRB/CNN AQI is ",aqi) 
#   return cbd_de0 + cbd_de1 + cbd_de2   
   
#def main():
#    ruleBase()       
#    takeInput()  
    #showTransformedInput() unnecessary      
#    takeCnnOutput() 
#    calculateMatchingDegreeBrbCnn() 
    #showMatchingDegree() unnecessary
#    showActivationWeight()   
#    updateBeliefDegree()    
#    aggregateER_BrbCnn()
    #getAQI(x) unnecessary

#main()      