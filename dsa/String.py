""" 
Remove duplicates from a given string
Approach: Using seen set
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
Patter matching problem
Using KMP algo
KMP:
    https://www.youtube.com/watch?v=sODA1BzFvsE&t=1656s&ab_channel=CoderArmy 
    Mimatch logic at 45:50 min
    Clear example starts at 49:18 min
String match:
    https://www.youtube.com/watch?v=6gQR8TaFXMw&ab_channel=CoderArmy

O(m+n)/O(m)
 """ 
def kmp_algo(arr, n):
    #longest prefix suffix
    lps=[0]*n
    pre= 0
    suf=1
    while suf<n:
        if arr[pre]==arr[suf]:
            lps[suf]=pre+1
            pre+=1
            suf+=1
        else:
            if pre==0: # reached till zero still no match
                lps[suf]=0
                suf+=1
            else:
                pre=lps[pre-1] # go to the index stored at previous index in lps     

    return lps

def areRotations_kmp(s1, s2):
    n1=len(s1)
    n2= len(s2)

    if n1!=n2:
        return False
    
    s_new= s1+s1
    n_new=2*n1
    lps_s2= kmp_algo(s2, n2)

    i=0
    j=0

    while i<n_new and j<n2:
        if s_new[i]==s2[j]:
            i+=1
            j+=1
        else:
            if j==0:
                i+=1
            else:
                j=lps_s2[j-1]

    if j==n2:
        return True
        #string match starts at index i-j
    
    return False

""" 
Print all permutations of given string
Approach: recursion
""" 
"""
Approach 1 with space O(n)
https://www.youtube.com/watch?v=YK78FU5Ffjw&ab_channel=takeUforward
O(n!*n)/O(n)
 """
def getPermutations1(word:str, ds:list, freq:dict):
    if len(ds)==len(word):
        print(''.join(ds))
        return
    for i in range(len(word)): #dont iterate chars directly as they get add to freq dict and duplicate chars dont pass if cond
        if i not in freq or freq[i]==False:
            ds.append(word[i])
            freq[i]=True
            getPermutations1(word, ds, freq)
            ds.pop()
            freq[i]=False

""" 
Approach 2 with space O(1)
https://www.youtube.com/watch?v=f2ic2Rsc9pU&ab_channel=takeUforward
O(n!*n)/O(1)
 """
def recurPermute(index, s, ans): #index starts with 0

    # Base Case
    if index == len(s):
        ans.append("".join(s))
        return

    # Swap the current index with all
    # possible indices and recur
    for i in range(index, len(s)):
        s[index], s[i] = s[i], s[index]
        recurPermute(index + 1, s, ans)
        s[index], s[i] = s[i], s[index] #swap back the indices to make the word back to previous shape as we are changing original string

""" 
Reverse words in a given string
Approach: Reverse the whole string then start word by word and keep reversing them
https://www.youtube.com/watch?v=RitppzIdMCo&ab_channel=ApnaCollege
Time: O(n)
Space: O(1)
 """
def reverseWords(s: str):
    n= len(s)
    #reverse the whole string
    s = s[::-1]
    answer=''
    i=0
    while i<n:
        word=''
        while s[i]!=' ' and i<n:
            word+=s[i]
            i+=1

        if word!='':
            orininalWord= word[::-1]
            answer= answer+orininalWord if answer=='' else answer+' '+orininalWord

        i+=1 #keep on looping if its a space
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
Approach: Hashmap
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
Approach: sorted hasmap
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
Approach: 26 based number system, %26 to get right most letter then /26 to go on searching for next letter.
Doing -1 to so that we can directly get the letter by adding it to 65(ascii for A). 
""" 

"""
Iterative
Time: O(log26n)
Space: O(log26n)
 """
def excel_column_name(column_number):
    result = ""
    while column_number > 0:
        column_number -= 1  # Adjust for 1-based index
        remainder = column_number % 26
        result = chr(65 + remainder) + result  # Convert to corresponding letter
        column_number = column_number // 26
    return result

""" 
Recursive
Time: O(log26n)
Space: O(1)
 """
def excel_column_name(column_number):
    if column_number == 0:
        return ''
    else:
        column_number -= 1
        remainder = column_number % 26
        letterAtLeft= excel_column_name(column_number // 26)
        return letterAtLeft + chr(65 + remainder)