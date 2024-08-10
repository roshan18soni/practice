""" 
sum of integer array
O(n)/O(n)
 """
def sumOfIntArray(arr, n):
    assert n>0 and arr[n-1]>=0, 'enter positive integers'
    if n==1:
        return arr[0]
    return arr[n-1] + sumOfIntArray(arr, n-1)

""" 
sum of digits
O(n)/O(n)
 """
def sumOfDigits(num):
    assert num>=0 and int(num)==num, 'enter positive number'
    if num==0:
        return 0
    return (num%10) + sumOfDigits(num//10)

""" 
power
O(n)/O(n)
 """
def power(num, n):
    if n==0: #base case
        return 1
    elif n<0: #recursive case for negative powers
        return 1/num * power(num, n+1)
    else: #recursive case for positive powers
        return num * power(num, n-1)

""" 
gcd- greatest common diviser
log(min(a, b))
 """
def gcd(a, b):
    assert int(a)==a and int(b)==b, "numbers must be integers"
    if b==0:
        return a
    else:
        return gcd(b, a%b)

""" 
decimal to binary
 """
def decimalToBinary(num):
    assert int(num)== num and num>=0, 'must be positive integer'
    if num==1:
        return 1
    else:
        return decimalToBinary(num//2)*10 + (num%2)

""" 
Reverse string
 """
def reverse(strng)-> str:
    if len(strng)==1:
        return strng
    else:
        a= reverse(strng[1:])
        return a+strng[0]

""" 
Is palindrome
 """
def isPalindrome(strng):
    if len(strng)<=1:
        return True
    else:
        if strng[0]==strng[len(strng)-1]:
            return isPalindrome(strng[1:len(strng)-1])
        else:
            return False

""" 
someRecursive-
Write a recursive function called someRecursive which accepts an array and a callback. 
The function returns true if a single value in the array returns true when passed to 
the callback. Otherwise it returns false.
 """
def isOdd(num):
    if num%2==0:
        return False
    else:
        return True
        
def someRecursive(arr, cb):
    if len(arr)==0:
        return False
    elif cb(arr[0]):
        return True
    else:
        return someRecursive(arr[1:], cb)

""" 
Flatten-
Write a recursive function called flatten which accepts an array of arrays 
and returns a new array with all values flattened.
 """
def flatten(arr):
    resultArr = []
    for custItem in arr:
        if type(custItem) is list:
            resultArr.extend(flatten(custItem))
        else: 
            resultArr.append(custItem)
    return resultArr 

