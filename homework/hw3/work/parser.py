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

	# if ((argc < 3) or (argc > 5) or (argc / 2 != 1)):
	# 	print_usage ()
	# 	sys.exit ()

	# mode = 0
	# user_input = ""

	# for i in range (len (argv)):
	# 	if argv [i] == "-m":
	# 		if argv [i + 1] == "sentence":
	# 			continue
	# 		elif argv [i + 1] == "file"

	user_input = argv [0]
	tokens = user_input.split ()

	sent = 'Time files like banana'.split ()

	mp = MaltParser (PATH_TO_MALTPARSER, PATH_TO_MODEL)
	graph = mp.parse_one (sent).tree()
	print (graph)



