"""
Standalone program to print all the contents of any pickle file
"""
import pickle
import pprint
import sys


def main():
    f = open(sys.argv[1], 'rb')
    
    while True:
        try:
            # pickle loads are FIFO wrt dumps
            stored_obj = pickle.load(f)
        except EOFError:
            # encountered when all pickle dumps are read and EOF is reached
            break
       
        pprint.pprint(stored_obj)

    f.close()

if __name__ == "__main__":
    main()