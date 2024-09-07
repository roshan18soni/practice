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

""" 
Capitalize first: Given an array of strings, capitalize the first letter of 
each string in the array.
 """
def capitalizeFirst(arr):
    result=[]
    if not arr:
        return result
        
    result.append(arr[0].capitalize())
    return result+capitalizeFirst(arr[1:])


""" 
nestedEvenSum. Return the sum of all even numbers in an object which 
may contain nested objects.

obj2 = {
  "a": 2,
  "b": {"b": 2, "bb": {"b": 3, "bb": {"b": 2}}},
  "c": {"c": {"c": 2}, "cc": 'ball', "ccc": 5},
  "d": 1,
  "e": {"e": {"e": 2}, "ee": 'car'}
}
 """
def nestedEvenSum(obj, sum=0):
    for key in obj:
        val= obj[key]
        if type(val) is int:
            if val%2==0:
                sum+=val
        elif type(val) is dict:
            sum+=nestedEvenSum(val)
            
    return sum

""" 
capitalizeWords. Given an array of words, return a new array 
containing each word capitalized.
 """
def capitalizeWords(arr):
    result=[]
    if not arr:
        return result
        
    result.append(arr[0].upper())
    return result+capitalizeWords(arr[1:])

""" 
stringifyNumbers which takes in an object and finds all of the values 
which are numbers and converts them to strings
 """
def stringifyNumbers(obj):
    result=obj
    for key, val in obj.items():
        if type(val) is int:
            result[key]=str(val)
        elif type(val) is dict:
            result[key]=stringifyNumbers(val)
            
    return result


""" 
collectStrings which accepts an object and returns an array of all 
the values in the object that have a typeof string.
 """
def collectStrings(obj):
    result=[]
    for key, val in obj.items():
        if type(val) is str:
            result.append(val)
        elif type(val) is dict:
            result+=collectStrings(val)
    return result