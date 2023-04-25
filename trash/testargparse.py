import argparse

#definition du parser 
parser = argparse.ArgumentParser()
parser.add_argument("--echo", default="hello world", help="echo the string you use here")
parser.add_argument("--device", type=str, default="local", help="device to use ('local' or 'cluster')")

#recuperation des arguments
args = parser.parse_args()
print(args)
print(args.echo)
print(args.device)