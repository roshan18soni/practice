class Stack:
    def __init__(self) -> None:
        self.list=[]
    
    def __str__(self) -> str:
        values= [str(v) for v in reversed(self.list)]
        return '\n'.join(values)

    def isEmpty(self):
        if self.list==[]:
            return True
        else:
            return False

    def push(self, value):
        self.list.append(value)
        return "pushed"

    def pop(self):
        if self.isEmpty():
            return 'no element'
        else:
            return self.list.pop()

    def peek(self):
        if self.isEmpty():
            return 'no elements'
        else:
            return self.list[len(self.list)-1]

    def delete(self):
        self.list= None