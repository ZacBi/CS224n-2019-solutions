class Test():
    def __init__(self, number):
        self.number = number


a = list(range(10))
b = [Test(n) for n in a]
c = b[:]
# c[1].number = 2
# c = list(range(10))

# print(len(b), len(c), b[1].number)
for test in c:
    if test.number == 1:
        c.remove(test)
# c.pop(0)
# c.pop(0)
print(len(c), len(b))