
import math

print ("Hello World")

def initGraphics():
    print("Graphics Initialised.")

initGraphics()


def runInLoops():
    for gpu_core in range(1,10):
        print (f"GPU Core {gpu_core} is active.")
    
runInLoops()

print(f" Which is greater 10^4 != 4^10? Hint : 4^8 is {math.pow(4,8)}")

