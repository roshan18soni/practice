""" 
Remove duplicates from a given string
Time: O(n)
Space: O(n)
 """
def removeDuplicates(word: str):
    result=''
    seen=set()

    for char in word:
        if char not in seen:
            seen.add(char)
            result+=char

    return result

""" 
Remove characters from the first string which are present in the second string
Time: O(|S1|), where |S1| is the size of given string 1.
Space: O(1), as only an array of constant size (26) is used. 
 """
def removeChars(s1, n1, s2, n2):
    s3 = ""
 
    # an array of size 26 to count the frequency of
    # characters
    arr = [0] * 26
 
    for i in range(0, n2):
        # assigned all the index of characters
        # which are present
        arr[ord(s2[i]) - ord('a')] = -1  # in second string by -1
    # (just flagging)
    for i in range(0, n1):
        if (arr[ord(s1[i]) - ord('a')] != -1):
            # Checking if the index of
            # characters don't have -1
            s3 += s1[i]  # i.e, that character was not
            # present in second string and
            # then storing that character
            # in string
 
    s1 = s3
    return s1

""" 
Check if given strings are rotations of each other or not
Time: O(N*N)
Space: O(N)
 """
def areRotations(string1, string2):
    size1 = len(string1)
    size2 = len(string2)
    temp = ''

    # Check if sizes of two strings are same
    if size1 != size2:
        return 0

    # Create a temp string with value str1.str1
    temp = string1 + string1

    # Now check if str2 is a substring of temp
    # string.count returns the number of occurrences of
    # the second string in temp
    if (temp.count(string2) > 0):
        return 1
    else:
        return 0

""" 
Check if a string is substring of another
 """
""" 
Approch: Naive
Time: O(n*m)
Space: O(1)
 """
def isSubstring(s1, s2):
    M = len(s1)
    N = len(s2)
 
    # A loop to slide pat[] one by one
    for i in range(N - M + 1):
 
        # For current index i,
        # check for pattern match
        for j in range(M):
            if (s2[i + j] != s1[j]):
                break
 
        if j + 1 == M:
            return i
 
    return -1
""" 
Approach: Efficient
Time: O(n)
Space: O(1)
 """
def isSubstring(s1, s2):
    if s1 in s2:
        return s2.index(s1)
    return -1

""" 
Reverse words in a given string
Time: O(n)
Space: O(1)
 """
def reverseWords(string: str):
    n=len(string)
    temp=''
    answer=''
    start=0
    for ch in string:
        print(start)
        if ch!=' ':
            temp+=ch
        elif (ch==' ' and temp):
            print(f'inside {start}')
            answer= temp+' '+answer
            temp=''

        if start==(n-1) and temp:
            answer= temp+' '+answer

        start+=1

    return answer


""" 
Smallest window in a String containing all characters of other String
Approach: Two Pointers and Hashing
Time: O(n)
Space: O(1)
 """
import sys
def findSubString(string: str, pattern: str):
    num_of_chars= 256
    len_str=len(string)
    len_pat=len(pattern)
    hash_str=[0]* num_of_chars
    hash_pat=[0]* num_of_chars

    for p in pattern:
        hash_pat[ord(p)]+=1

    start=0
    start_index=-1
    min_len=sys.maxsize
    count=0
    for i in range(len_str):
        hash_str[ord(string[i])]+=1

        if hash_str[ord(string[i])]<= hash_pat[ord(string[i])]:
            count+=1

        if count==len_pat:

            while (hash_str[ord(string[start])]> hash_pat[ord(string[start])]
                    or hash_pat[ord(string[start])]==0):

                    hash_str[ord(string[start])]-=1
                    start+=1

            window_size= i-start+1
            if min_len> window_size:
                min_len=window_size
                start_index=start

    if start_index==-1:
        print('no window found')
        return ""
    else:   
        print(f'max window from {start_index} to {start_index+min_len-1}')
        return string[start_index:start_index+min_len]

""" 
Check whether two Strings are anagram of each other
Time: O(n)
Space: O(1)
 """
def isAnagram(a, b):
 
    if (len(a) != len(b)):
        return False
 
    map = {}
    for i in range(len(a)):
        if (a[i] in map):
            map[a[i]] += 1
 
        else:
            map[a[i]] = 1
 
    for i in range(len(b)):
        if (b[i] in map):
            map[b[i]] -= 1
        else:
            return False
 
    keys = map.keys()
    for key in keys:
        if (map[key] != 0):
            return False
 
    return True

""" 
Write your own atoi()
Time: O(n)
Space: O(1)
 """
def myAtoi(Str):
 
    sign, base, i = 1, 0, 0
 
    # If whitespaces then ignore.
    while (Str[i] == ' '):
        i += 1
 
    # Sign of number
    if (Str[i] == '-' or Str[i] == '+'):
        sign = 1 - 2 * (Str[i] == '-')
        i += 1
 
    # Checking for valid input
    while (i < len(Str) and
           Str[i] >= '0' and Str[i] <= '9'):
 
        # Handling overflow test case, maxsize=9223372036854775807
        if (base > (sys.maxsize // 10) or
            (base == (sys.maxsize // 10) and
                (ord(Str[i]) - ord('0')) > 7)):
            if (sign == 1):
                return sys.maxsize
            else:
                return -(sys.maxsize)
 
        base = 10 * base + (ord(Str[i]) - ord('0'))
        i += 1
 
    return base * sign


""" 
Rearrange a string so that all same characters become d distance away
Time: O(n+mlog(Max))
Space: O(n)
 """
def rearragne(string, d):
    n= len(string)
    freq_hash= {}

    for ch in string:
        freq_hash[ch]= freq_hash.get(ch, 0)+1
        
    sorted_freq_hash_keys= sorted(freq_hash, key=freq_hash.get, reverse=True)
    
    result= ['']*n

    start_index=0
    for ch in sorted_freq_hash_keys:
        freq= freq_hash[ch]

        for i in range(freq):
            index= start_index+i*d
            if index<n:
                result[index]=ch
            else:
                return 'Cannot be arranged'
        
        start_index+=1
        while start_index<n and result[start_index]!='':
            start_index+=1

    return ''.join(result)

""" 
Find Excel column name from a given column number
Time: O(log26n)
Space: O(log26n)
 """
def excel_column_name(column_number):
    result = ''
    while column_number > 0:
        remainder = (column_number - 1) % 26
        result = chr(65 + remainder) + result
        column_number = (column_number - 1) // 26

    return result

""" 
Using Recursion
Time: O(log26n)
Space: O(1)
 """
def excel_column_name(column_number):
    if column_number == 0:
        return ''
    else:
        remainder = (column_number - 1) % 26
        return excel_column_name((column_number - 1) // 26) + chr(65 + remainder)