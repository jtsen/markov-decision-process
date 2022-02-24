from enum import Enum
import copy

class node_type(Enum):
    """
    Class for node types
    """
    D = 'Decision'
    C = 'Chance'
    T = 'Terminal'
    
def value_iteration(rewards, edges, nodes, policy, ntypes, df, tol, iteration, min_toggle):
    """
    Value iteration function for the Markov Decision Process.
    Given a policy and a decision tree, outputs the values at
    each state given a tolerance and iteration limit.

    Args:
        rewards ([dict]): [base rewards at each state]
        edges ([dict]): [nodes and their outgoing edges (next states)]
        nodes ([list]): [node names]
        policy ([dict]): [policy used to generate V]
        ntypes ([type]): [node types]
        df ([float]): [discount factor]
        tol ([float]): [tolerance]
        iteration ([int]): [iteration limit]
        min_toggle ([type]): [minimize or maximize]
        
    Returns:
        [dict]: [values at each state after convergence or iteration limit]
    """
    it = 0
    old_values, values = rewards.copy(), rewards.copy()
    while True:
        for node in nodes:
            if ntypes[node] == node_type.T:
                continue
            inter = {}
            for action in edges[node]:
                inter[action] = rewards[node] + (df * sum(old_values[a]*policy[node][edges[node].index(a)] for a in edges[node]))
            if min_toggle:
                values[node] = min(inter.values())
            else:
                values[node] = max(inter.values())
        if all(values[n]-old_values[n]<tol for n in nodes) or it>iteration:
            break
        else:
            old_values = values.copy()
            it+=1
    return values
    
def policy_iteration(V, rewards, nodes, edges, pol, ntypes, min_toggle):
    """
    Policy iteration step of the Markov Decision Process.
    This function loops through all possible future states
    of all states and chooses the best possible action at
    each of the states and returns the policy.

    Args:
        V ([dict]): [result from value iteration]
        rewards ([dict]): [base rewards at each state]
        nodes ([list]): [node names]
        edges ([dict]): [nodes and their outgoing edges (next states)]
        pol ([dict]): [policy used to generate V]
        ntypes ([type]): [node types]
        min_toggle ([type]): [minimize or maximize]

    Returns:
        [dict]: [a new policy]
    """
    policy = copy.deepcopy(pol)
    new_policy = copy.deepcopy(policy)
    new_V = {}
    for node in nodes:
        if ntypes[node] == node_type.D:
            curr_max = V[node]
            curr_policy = copy.deepcopy(new_policy[node])
            for i in range(len(edges[node])-1):
                new_policy[node].append(new_policy[node].pop(0))
                new_V[node] = rewards[node]+sum(V[next_node] * new_policy[node][edges[node].index(next_node)] for next_node in edges[node])
                if min_toggle:
                    if new_V[node] < curr_max:
                        curr_max = new_V[node]
                        curr_policy=copy.deepcopy(new_policy[node])
                else:
                    if new_V[node] > curr_max:
                        curr_max = new_V[node]
                        curr_policy = copy.deepcopy(new_policy[node])
            policy[node]=copy.deepcopy(curr_policy)
    return policy
    
def mdp(filename, df=1.0, min_toggle=False, tol=0.01, iter=100):
    """
    Markov Decision Process function;
    Pseudo-code given by Professor Paul Bethe from NYU
    
    pi = initial policy (arbitrary)
    V = initial values (perhaps using rewards)
    for {
      V = ValueIteration(pi) // computes V using stationery P
      pi' = GreedyPolicyComputation(V) // computes new P using latest V
      if pi == pi' then return pi, V
      pi = pi'
    }

    Args:
        filename ([string]): [input file of probabilities, edges, rewards/costs]
        df (float, optional): [discount factor]. Defaults to 1.0.
        min_toggle (bool, optional): [minimize for cost or not]. Defaults to False.
        tol (float, optional): [tolerance]. Defaults to 0.01.
        iter (int, optional): [iterations]. Defaults to 100.

    Returns:
        [dict]: [values of all the nodes]
        [dict]: [the optimal policy for each node]
    """
    rewards, edges, percents, nodes, ntypes = read_input(filename)
    policy = copy.deepcopy(percents)
    while True:
        V = value_iteration(rewards, edges, nodes, policy, ntypes, df, tol, iter, min_toggle)
        new_policy = policy_iteration(V, rewards, nodes, edges, policy, ntypes, min_toggle)
        if all(policy[n] == new_policy[n] for n in policy.keys()):
            break
        else:
            policy=copy.deepcopy(new_policy)
    
    #print output to console
    for i, (k,v) in enumerate(sorted(policy.items())):
        if ntypes[k] == node_type.D:
            dest = edges[k][policy[k].index(max(policy[k]))]
            print("{} -> {}".format(k, dest))

    for i, (k,v) in enumerate(sorted(V.items())):
        print("{}={}".format(k,round(v,4))),
        
    return V, policy

def read_input(filename):
    """
    Input file parser that categorizes different input
    nodes as Terminal, Chance, Decision

    Args:
        filename ([string]): [the input file within the same directory]

rewards, edges, percents, sorted(nodes), ntypes
    Returns:
        [dict]: [rewards of each node]
        [dict]: [outgoing edges of each node]
        [dict]: [first policy]
        [list]: [sorted list of node names]
        [dict]: [type of nodes]
    """
    rewards, edges, percents, nodes = {}, {}, {}, []
    with open(filename, 'r') as file:
        input_line = file.readline()
        while(input_line):
            if input_line.startswith("#"): #skip comments
                input_line = file.readline() #read the next line
                continue
            input_line=input_line.replace("\n","") #remove white space
            if "=" in input_line:
                curr_line = input_line.replace(" ","").split("=")
                if curr_line[0] not in nodes:
                    nodes.append(curr_line[0])
                if curr_line[0] not in rewards.keys():
                    rewards[curr_line[0]]=float(curr_line[1])
            elif "%" in input_line:
                curr_line = input_line.split("%")
                curr_line[0]=curr_line[0].strip(" ")
                #https://stackoverflow.com/questions/3845423/remove-empty-strings-from-a-list-of-strings
                prob_list = filter(None,curr_line[1].split(" "))
                new_prob_list=[]
                for prob in prob_list:
                    new_prob_list.append(float(prob))
                if curr_line[0] not in nodes:
                    nodes.append(curr_line[0])
                if curr_line[0] not in percents.keys():
                    percents[curr_line[0]]=new_prob_list
            elif ":" in input_line:
                curr_line = input_line.replace(" ", "").split(":")
                edge_list = curr_line[1].strip("[]").split(",")
                if curr_line[0] not in nodes:
                    nodes.append(curr_line[0])
                if curr_line[0] not in edges.keys():
                    edges[curr_line[0]]=edge_list
            
            input_line = file.readline() #read the next line
    for node in nodes:
        if node not in rewards.keys():
            rewards[node]=0.0   
    ntypes = dict.fromkeys(nodes)
    for node in nodes:
        if node not in edges.keys():
            ntypes[node]=node_type.T
            continue
        if node not in percents.keys() and node in edges.keys():
            ntypes[node]=node_type.C
            percents[node]=[1.0]
            continue
        if node not in rewards.keys() and node in edges.keys():
            rewards[node]=0.0
            continue
        ntypes[node]=node_type.C
    for i, (k, v) in enumerate(percents.items()):
        if len(v) == 1 and len(edges[k])>1:
            ntypes[k]=node_type.D
            if not len(edges[k])-1:
                continue
            else:
                fill_val = (1.0 - v[0]) / float(len(edges[k])-1)
                for index in range(len(edges[k])-1):
                    percents[k].append(fill_val)
        
    return rewards, edges, percents, sorted(nodes), ntypes