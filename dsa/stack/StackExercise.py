""" 
implement queue using stack
Time: enqueue- O(n) dequeue- O(1)
Space: enqueue- O(n) dequeue- O(1)
 """
from collections import deque


class QueueUsingStack:
    def __init__(self) -> None:
        self.stack1= []
        self.stack2= []

    def enquue(self, value):
        if len(self.stack1)==0:
            self.stack1.append(value)
        else:
            while self.stack1:
                self.stack2.append(self.stack1.pop())
            self.stack1.append(value)

            while self.stack2:
                self.stack1.append(self.stack2.pop())

    def dequeue(self):
        if not self.stack1:
            return "empty stack"

        return self.stack1.pop()

""" 
Check for Balanced Brackets in an expression (well-formedness)
O(n)/O(n)
 """
def areBracketsBalanced(expr):
    stack= []
    for chr in expr:
        print(chr)
        if chr in '[{(':
            stack.append(chr)
        else:
            if stack:
                popped= stack.pop()
            else:
                return 'not balanced'

            if chr==']':
                if popped!='[':
                    return 'not balanced'
            elif chr=='}':
                if popped!='{':
                    return 'not balanced'
            elif chr==')':
                if popped!='(':
                    return 'not balanced'
    if stack:
        return 'not balanced'
    else:
        return 'balalced'

""" 
Reverse Stack/String using recursion
Approach: Using the nested recursion to reverse the elements
O(n square)/O(n)
 """
def insert_at_botoom(stack, ele):
    if not stack:
        stack.append(ele)
    
    popped_ele=stack.pop()
    insert_at_botoom(stack, ele)
    stack.append(popped_ele)

def reverse(stack):
    if stack:
        popped_ele= stack.pop()
        reverse(stack)
        insert_at_botoom(stack, popped_ele)


""" 
Special Stack having getMin method
 """

""" 
Method1: Using auxiliary stack
O(1)/O(n)
 """
class specialStack:
    def __init__(self) -> None:
        self.stack= []
        self.stack_aux= []

    def isEmpty(self):
        if len(self.stack)==0:
            return True
        else:
            return False

    def isEmptyAux(self):
        if len(self.stack_aux)==0:
            return True
        else:
            return False

    def peak(self):
        if not self.isEmpty():
            return self.stack[-1]
        else:
            return 'empty stack'

    def peak_aux(self):
        if not self.isEmptyAux():
            return self.stack_aux[-1]
        else:
            return 'empty aux stack'

    def push_aux(self, value):
        self.stack_aux.append(value)

    def push(self, value):
        self.stack.append(value)

        if self.isEmptyAux() or value<self.peak_aux():
            self.push_aux(value)
    
    def pop_aux(self):
        if not self.isEmptyAux():
            return self.stack_aux.pop()
        else:
            return 'aux stack is empty'

    def pop(self):
        if not self.isEmpty():
            popped= self.stack.pop()
        else:
            return 'stack is empty'

        if popped== self.peak_aux():
            self.pop_aux()
        
        return popped
        
    def getMin(self):
        return self.peak_aux()


""" 
Method2: Using encoded values
O(1)/O(1)
 """
class specialStack:
    def __init__(self) -> None:
        self.stack= []
        self.min= -1
        self.Dummy_Val=1000

    def push(self, value):
        if not self.stack or value< self.min:
            self.min= value

        self.stack.append(value*self.Dummy_Val+self.min)

    def pop(self):
        if self.stack:
            popped= self.stack.pop()
            if self.stack:
                self.min= self.stack[-1]%self.Dummy_Val
            else:
                self.min= -1

            return popped//self.Dummy_Val
        else:
            return 'empty stack'

    def peak(self):
        if self.stack:
            return self.stack[-1]//self.Dummy_Val
        else:
            return 'empty stack'

    def getMin(self):
        if self.stack:
            return self.min

"""
Stack using queue
"""
"""
Method1: using two queues
O(n)/O(n) 
 """
class StackUsingQueue:
    def __init__(self) -> None:
        self.queue1= []
        self.queue2= []

    def push(self, value):
        if self.queue1:
            while self.queue1:
                self.queue2.append(self.queue1.pop(0))
            
            self.queue1.append(value)

            while self.queue2:
                self.queue1.append(self.queue2.pop(0))
        else:
            self.queue1.append(value)

    def pop(self):
        if self.queue1:
            return self.queue1.pop(0)
        else:
            return 'queue is empty'

    def peak(self):
        return self.queue1[0]

    def __str__(self) -> str:
        return ', '.join(self.queue1)

"""
Method2: using single queue
O(n)/O(1) 
 """
class StackUsingSingleQueue():
    from collections import deque
    def __init__(self) -> None:
        self.queue= deque()

    def push(self, value):
        if self.queue:
            len_before_append= len(self.queue)
            self.queue.appendleft(value)

            for _ in range(len_before_append):
                self.queue.appendleft(self.queue.pop())
        else:
            self.queue.appendleft(value)

    def pop(self):
        if self.queue:
            return self.queue.pop()
        else:
            return "empty stack"

    def peak(self):
        if self.queue:
            return self.queue[len(self.queue)-1]
        else:
            return "empty stack"

    def __str__(self) -> str:
        return ', '.join(self.queue)

""" 
Expresstion evaluation
O(n)/O(n)
 """
# expression where tokens are 
# separated by space.

# Function to find precedence
# of operators.
def precedence(op):
	
	if op == '+' or op == '-':
		return 1
	if op == '*' or op == '/':
		return 2
	return 0

# Function to perform arithmetic
# operations.
def applyOp(a, b, op):
	
	if op == '+': return a + b
	if op == '-': return a - b
	if op == '*': return a * b
	if op == '/': return a // b

# Function that returns value of
# expression after evaluation.
def evaluate(tokens):
	
	# stack to store integer values.
	values = []
	
	# stack to store operators.
	ops = []
	i = 0
	
	while i < len(tokens):
		
		# Current token is a whitespace,
		# skip it.
		if tokens[i] == ' ':
			i += 1
			continue
		
		# Current token is an opening 
		# brace, push it to 'ops'
		elif tokens[i] == '(':
			ops.append(tokens[i])
		
		# Current token is a number, push 
		# it to stack for numbers.
		elif tokens[i].isdigit():
			val = 0
			
			# There may be more than one
			# digits in the number.
			while (i < len(tokens) and
				tokens[i].isdigit()):
			
				val = (val * 10) + int(tokens[i])
				i += 1
			
			values.append(val)
			
			# right now the i points to 
			# the character next to the digit,
			# since the for loop also increases 
			# the i, we would skip one 
			# token position; we need to 
			# decrease the value of i by 1 to
			# correct the offset.
			i-=1
		
		# Closing brace encountered, 
		# solve entire brace.
		elif tokens[i] == ')':
		
			while len(ops) != 0 and ops[-1] != '(':
			
				val2 = values.pop()
				val1 = values.pop()
				op = ops.pop()
				
				values.append(applyOp(val1, val2, op))
			
			# pop opening brace.
			ops.pop()
		
		# Current token is an operator.
		else:
		
			# While top of 'ops' has same or 
			# greater precedence to current 
			# token, which is an operator. 
			# Apply operator on top of 'ops' 
			# to top two elements in values stack.
			while (len(ops) != 0 and
				precedence(ops[-1]) >=
				precedence(tokens[i])):
						
				val2 = values.pop()
				val1 = values.pop()
				op = ops.pop()
				
				values.append(applyOp(val1, val2, op))
			
			# Push current token to 'ops'.
			ops.append(tokens[i])
		
		i += 1
	
	# Entire expression has been parsed 
	# at this point, apply remaining ops 
	# to remaining values.
	while len(ops) != 0:
		
		val2 = values.pop()
		val1 = values.pop()
		op = ops.pop()
				
		values.append(applyOp(val1, val2, op))
	
	# Top of 'values' contains result,
	# return it.
	return values[-1]

