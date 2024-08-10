""" 
implement queue using stack
Time: enqueue- O(n) dequeue- O(1)
Stpace: enqueue- O(n) dequeue- O(1)
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
            while len(self.stack1)>0:
                self.stack2.append(self.stack1.pop())
            self.stack1.append(value)

            while len(self.stack2)>0:
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
