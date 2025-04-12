""" 
Circular Gas Station
 """
""" 
brute force
O(n2)/O(1)
 """
def startStationBruteForce(gas, cost):
    n=len(gas)

    for i in range(n):
        print(f'i: {i}')
        startIndex=i
        curr_gas=0
        flag=True
        for j in range(n):
            print(f'j: {j}')
            circularIndex= (startIndex+j)%n
            curr_gas= curr_gas+ gas[circularIndex]-cost[startIndex]
            if curr_gas<0:
                curr_gas=0
                flag=False
                break
            
        if flag:
            break

    return startIndex

""" 
Optimized
O(n)/O(1)
 """
def startStationOptimized(gas, cost):
    total_gas=0
    curr_gas=0
    n=len(gas)
    start_index=0
    for i in range(n):
        curr_gas+= gas[i]-cost[i]
        total_gas= gas[i]-cost[i]
        if curr_gas<0:
            start_index=i+1
            curr_gas=0
        
    return start_index if total_gas>=0 else -1