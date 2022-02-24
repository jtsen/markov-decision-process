import argparse
from alg import mdp
if __name__ == '__main__':
    """
    Driver program for the using the converter as well as the DPLL algorithm
    """
    #create the argument parser object and add the required and optional flags and required graph file
    parser = argparse.ArgumentParser()
    parser.add_argument("-df", 
                        dest="discount_factor", 
                        default=1.0, 
                        help="discount factor; default=1.0")
    parser.add_argument("-min", 
                        dest="minimize",
                        action="store_true", 
                        default=False, 
                        help="minimize values as costs; default=max")
    parser.add_argument("-tol", 
                        dest="tolerance", 
                        default=0.01, 
                        help="value iteration float tolerance; default=0.01")
    parser.add_argument("-iter", 
                        dest="iteration", 
                        default=100, 
                        help="value iteration cutoff; default=100")
    parser.add_argument("input_file")

    args = parser.parse_args()

    values, policy = mdp(args.input_file, 
                        float(args.discount_factor), 
                        args.minimize,
                        float(args.tolerance),
                        float(args.iteration))