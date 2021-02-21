from a import a

a = {"a": 2}
def test():
    print(a)
test()
a = {"a": 1}
test()

if __name__ == "__main__":
    test()