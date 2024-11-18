
import math  # part of Task4

# Task1
print ("Hello World")

# Task2
def initGraphics():
    print("Graphics Initialised.")

initGraphics()

# Task3
def runInLoops():
    for gpu_core in range(1,10):
        print (f"GPU Core {gpu_core} is active.")
    
runInLoops()

# Task4
print(f" Which is greater 10^4 != 4^10? Hint : 4^8 is {math.pow(4,8)}")

