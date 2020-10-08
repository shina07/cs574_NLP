import sys
from nltk.parse.malt import MaltParser

PATH_TO_MALTPARSER = "maltparser-1.9.2"
PATH_TO_MODEL = "kaist-conll.mco"
def print_usage ():
	print ("usage: $ python3 parser.py <input text>")

if __name__ == '__main__':
	argv = sys.argv[1:]
	argc = len (sys.argv)

	if (argc != 2):
		print_usage ()
		sys.exit ()


	user_input = argv [0]
	tokens = user_input.split ()

	mp = MaltParser (PATH_TO_MALTPARSER, PATH_TO_MODEL)
	graph = mp.parse_one (tokens).tree()
	print (graph)



