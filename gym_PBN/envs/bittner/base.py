"""
This file contains arcane magics.
"""
import copy
import itertools
import pickle
import random
import time
from collections import defaultdict
from os import path

import networkx as nx
import numpy as np
from scipy.special import smirnov
from scipy.stats import logistic


class Node:
    def __init__(self, index, bittnerIndex, name, ID, LUTflag=False):
        self.index = index
        self.bittnerIndex = bittnerIndex
        self.name = name
        self.ID = ID
        self.LUTflag = LUTflag

        self.CODsum = 0
        self.value = None
        self.predictors = []
        self.inputNodes = None
        self.truth_table = None

    def add_predictors(self, predictors):
        IDstoPrint = []
        for COD, A, inputIDs in predictors:
            if type(COD) == type(None):
                pass
            else:
                self.CODsum += COD
                if len(self.predictors) == 0:
                    self.predictors += [(inputIDs, A, COD)]
                else:
                    _, _, prevCOD = self.predictors[-1]
                    currCOD = prevCOD + COD
                    self.predictors += [(inputIDs, A, currCOD)]
                    for inID in list(inputIDs):
                        if inID not in IDstoPrint:
                            IDstoPrint = IDstoPrint + [inID]

    def addInputNode(self, inputNode):
        self.inputNodes += [inputNode]

    def addInputNodes(self, inputNodes):
        self.inputNodes = np.append(self.inputNodes, inputNodes)

    def getInputNodes(self):
        inputList = []
        for pred in self.predictors:
            for ID in pred[0]:
                if not ID in inputList:
                    inputList.append(ID)
        return inputList

    def setValue(self, value):
        self.value = value
        # self.Predstep = lambda self, state: print('reduced'); return self.value

    def addLUT(self, LUT, inputIDs):
        self.LUT = LUT
        self.inputIDs = inputIDs

    def getStateProbs(self, state):
        probs = [0] * 2
        prevCOD = 0
        for IDs, A, COD in self.predictors:
            X = np.ones((len(IDs) + 1, 1))
            for j in range(len(IDs)):
                ID = IDs[j]
                x = state[ID]
                X[j] = x
            X[len(IDs)] = state[self.ID]
            currCOD = COD - prevCOD
            prevCOD = COD
            Ypred = np.matmul(X.T, A)
            Ypred = logistic.cdf(Ypred)
            if Ypred < 0.5:
                Y = 0
            else:
                Y = 1
            probs[Y] += currCOD / self.CODsum
        return probs

    def Predstep(self, state):
        if self.value is None:
            raise Exception("Forgot to initialise the states")

        # choose predictor
        r = random.random() * self.CODsum
        IDs, A, COD = self.predictors[0]
        for IDs, A, COD in self.predictors:
            if COD > r:
                break

        # relevant genes vector
        X = np.ones((len(IDs) + 1, 1))
        for j, ID in enumerate(IDs):
            x = state[ID]
            X[j] = x
        X[len(IDs)] = state[self.ID]

        # predict
        # I think something was missing here, probably sigmoid
        # compare paper DOI:10.1117/1.1289142 equation (2) and following paragraph
        # I added simgmoid, it makes the most sense to me
        Ypred = np.matmul(X.T, A)

        # logistic.cdf turs out to be expensive to compute
        # Ypred = logistic.cdf(Ypred)
        # x < 0. <=> sigmoid(x) < 0.5
        if Ypred < 0.:
            Y = 0
        else:
            Y = 1
        return Y

    def LUTstep(self, state):
        raise ValueError("you shouldn't be here")
        X = np.empty((len(self.inputIDs), 1))
        for j in range(len(self.inputIDs)):
            ID = self.inputIDs[j]
            x = state[ID]
            X[j] = x
        inputInt = int(integerize(X)[0])
        LUTrow = self.LUT[inputInt, :]

        r = random.random()
        if r < LUTrow[0]:
            Y = 0
        else:
            Y = 1
        return Y

    def step(self, state, verbose=False):
        # if self.used_by_agent:
        #     return self.value

        if self.LUTflag:
            Y = self.LUTstep(state)
        else:
            Y = self.Predstep(state)
        self.value = Y
        return Y


def genBoolList(n, length, b):
    output = np.zeros((length), dtype=int)
    for i in range(length):
        output[i] = n % b
        n = n // b
    return output


def integerize(state):
    output = 0
    for i in range(len(state)):
        output += state[i] * (2**i)
    return output


def KSstatistic(states1, states2, nMax):
    # Oh boy let's go
    M = states1.size
    maxDist = 0
    for x in range(nMax):
        dist = abs(states1[x] - states2[x])
        if dist > maxDist:
            maxDist = dist
    D = maxDist
    signif = smirnov(M, D)
    print("D: {0}".format(D))
    #    print("Alpha: {0}".format(signif))
    return D, signif


def indicatorF(inp, x):
    if inp <= x:
        return 1
    else:
        return 0


# Object representing the Graph of the Probabilistic/random boolean network
class Graph:
    def __init__(self, base):
        self.nodes = []
        self.edges = []  # Indexes in the form of tuples (from->to)
        self.base = base
        self.perturbations = False
        self.p = 0.001
        self.is_directed = True
        self.fliped = set()

    @property
    def N(self):
        return len(self.nodes)

    def genSTG(self, savepath=None):
        if savepath is not None:
            if path.exists(savepath):
                return pickle.load((open(savepath, "rb")))

        graphNodes = {}
        stg = nx.DiGraph()
        stg.add_nodes_from(itertools.product([0, 1], repeat=len(self.nodes)))

        for possibleState in itertools.product([0, 1], repeat=len(self.nodes)):

            self.setState(possibleState)
            nextStates = self.getNextStates()

            for state in nextStates:
                stg.add_edge(possibleState, state)

        if savepath is not None:
            pickle.dump(graphNodes, open(savepath, "wb"))
        return stg

    # async version of getNextStates
    def getNextStates(self, state=None):
        probs = []
        state = self.getState() if state is None else state
        for node in self.nodes:
            if node.truth_table is None:
                named_state = {node.ID: s for node, s in zip(self.nodes, state)}
                prob = node.getStateProbs(named_state)[1]
            else:
                tt = node.truth_table
                for i in node.predictors[0]:
                    tt = tt[state[i]]
                prob = tt
            probs.append([1. - prob, prob])

        nextStates = defaultdict(float)

        for i in range(len(state)):
            nextState = list(state)
            prob = probs[i]

            if prob[0] > 0.:
                nextState[i] = 0
                nextStates[tuple(nextState)] += prob[0] / len(state)
                #print(f"{i} added {nextState} to {state}")
            if prob[1] > 0.:
                nextState[i] = 1
                nextStates[tuple(nextState)] += prob[1] / len(state)
                #print(f"{i} added {nextState} to {state}")

        return nextStates

    def getPrevStates(self, state=None):
        state = self.getState() if state is None else state
        prev_states = set()

        if state in self.getNextStates(state):
            prev_states.add(state)

        for i, node in enumerate(self.nodes):
            state_list = list(state)
            state_list[i] = 1 - state_list[i]
            ns = self.getNextStates(state_list)

            if state in ns.keys():
                prev_states.add(tuple(state_list))

        return prev_states

    # sync version of getNextStates
    def sync_getNextStates(self):
        probs = []
        for node in self.nodes:
            prob = node.getStateProbs(self.getState())
            probs = probs + [prob]
        a = [[0, 1]] * len(self.nodes)
        possibleStates = list(itertools.product(*a))
        nextStates = defaultdict(float)
        for state in possibleStates:
            p = 1
            for i in range(len(state)):
                p *= probs[i][state[i]]
            if p > 0:
                nextStates[state] = p
        return nextStates

    def add_nodes(self, nodeList):
        self.nodes = nodeList

    def addCon(self, conn):
        k = conn.shape[0] - 1
        self.k = k
        for i in range(conn.shape[1]):
            targID = conn[k, i]
            targNode = self.getNodeByID(targID)
            for j in range(k):
                predID = conn[j, i]
                predNode = self.getNodeByID(predID)
                self.edges = self.edges + [(predNode.index, targNode.index)]
                targNode.addInputNode(predNode)

    def addEdge(self, startIndex, endIndex):
        self.nodes[endIndex].addInputNode(self.nodes[startIndex])
        self.edges = self.edges + [(self.nodes[startIndex], self.nodes[endIndex])]

    def flipNode(self, index):
        if index < len(self.nodes):
            self.nodes[index].setValue(self.nodes[index].value ^ True)
        else:
            raise ValueError(f"Invalid action, no node at index {index}")

    def synch_step(self):
        if self.perturbations:
            pertFlag = np.random.rand(len(self.nodes)) < self.p
            if pertFlag.any():
                oldState = self.getState()
                for i in range(len(oldState.keys())):
                    if pertFlag[i]:
                        flipid = list(oldState.keys())[i]
                        oldState[flipid] = oldState[flipid] ^ 1
                self.setState(list(oldState.values()))
            else:
                oldState = self.getState()
                for i in range(0, len(self.nodes)):
                    self.nodes[i].step(oldState)
        else:
            oldState = self.getState()
            for i in range(0, len(self.nodes)):
                self.nodes[i].step(oldState)

    # async. step
    def step(self, changed_nodes: list = None, i=None):
        oldState = self.getLabeledState()
        i = random.randint(0, len(self.nodes) - 1) if i is None else i
        # while i in changed_nodes:
        #     i = random.randint(0, len(self.nodes) - 1)
        self.nodes[i].step(oldState)
        return self.getState()

    def getLabeledState(self):
        outputState = {}
        for node in self.nodes:
            outputState[node.ID] = node.value
        return outputState

    def getState(self):
        outputState = {}
        for node in self.nodes:
            outputState[node.ID] = node.value
        return tuple(outputState.values())

    def getNames(self):
        names = []
        for i in range(0, len(self.nodes)):
            names.append([self.nodes[i].name])
        return names

    def getIDs(self):
        IDs = [node.ID for node in self.nodes]
        return IDs

    def getNodeByID(self, ID):
        for node in self.nodes:
            if node.ID == ID:
                return node
        print("Node with ID {0} not found.".format(ID))
        return None

    def printGraph(self, path, dist=10, charLim=10):
        self.G = nx.DiGraph()
        for node in self.nodes:
            self.G.add_node(str(node.name[0][:charLim]))
            inputIDs = node.getInputNodes()
            print(inputIDs)
            for ID in inputIDs:
                inNode = self.getNodeByID(ID)
                self.G.add_edge(
                    str(inNode.name[0][:charLim]), str(node.name[0][:charLim])
                )
        # pos = nx.spring_layout(self.G, k=dist)
        # nx.draw_networkx_nodes(self.G, pos, node_size=500)
        # nx.draw_networkx_edges(self.G, pos)
        labels = {}
        for node in self.G.nodes():
            labels[node] = node
        # nx.draw_networkx_labels(self.G, pos, labels, font_size=8)
        # plt.savefig(path)
        return self.G

    def setState(self, state):
        for x in range(0, len(self.nodes)):
            self.nodes[x].value = int(state[x])

    def genRandState(self):
        for x in range(0, len(self.nodes)):
            self.nodes[x].value = int(random.randint(0, self.base - 1))

    def getAttractors(self, verbal=False):
        attractors = list(attractorSetFinder(self.nodes, self.edges, verbal=True))
        return attractors


# wiesza mi kompa na N = 10
# i will mark it as deprecated
def dep_findAttractors(STG):
    STG = stripSTG(STG)
    unknowns = {}
    GA = {}
    NGA = {}
    for key in list(STG.keys()):
        unknowns[key] = [key]  # (Status, tags)
    unknowns, GA, NGA = identify(unknowns, GA, NGA, STG)
    while len(unknowns) > 0:
        print(f"Unknowns to clean up left: {len(unknowns)}, {len(GA)}, {len(NGA)}, {len(STG)}")
        toRemove = list(unknowns.keys())[0]
        print("remove node")
        unknowns, GA, NGA, STG = removeNode(unknowns, GA, NGA, toRemove, STG)
        print("identify")
        unknowns, GA, NGA = identify(unknowns, GA, NGA, STG)
        print("GA: {0}".format(GA))
    return GA


def findAttractors(stg):
    return list(nx.attracting_components(stg))


def stripSTG(STG):
    for node in list(STG.keys()):
        inputs, outputs = STG[node]
        newInputs = list(inputs.keys())
        newOutputs = list(outputs.keys())
        STG[node] = (newInputs, newOutputs)
    return STG


def removeNode(unknowns, GA, NGA, toRemove, STG):
    inNodes = STG[toRemove][0]
    outNodes = STG[toRemove][1]
    #print("n In: {0}".format(len(inNodes)))
    #print("n Out: {0}".format(len(outNodes)))
    loop_counter = -0
    for inn in inNodes:
        if inn != toRemove:

            # Upstream passing tags
            if inn in GA.keys():
                oldTags = GA[inn]
                newTags = oldTags + unknowns[toRemove]
                GA[inn] = newTags
            if inn in NGA.keys():
                oldTags = NGA[inn]
                newTags = oldTags + unknowns[toRemove]
                NGA[inn] = newTags
            if inn in unknowns.keys():
                oldTags = unknowns[inn]
                newTags = oldTags + unknowns[toRemove]
                unknowns[inn] = newTags
            # Tags passed, happy days.

            inNodeIn, inNodeOut = STG[
                inn
            ]  # Outputs of This particular input (looping over). Need to add outputs to it
            for out in outNodes:  # My outputs
                if (
                    not out in inNodeOut
                ):  # If this output is also an output of the input node
                    inNodeOut = inNodeOut + [out]
                # By now all my outputs should be added to the outputs of the input.
            inNodeOut.remove(toRemove)  # Remove self from the outputs of the input.
            STG[inn] = (inNodeIn, inNodeOut)  # Push it in to the STG!

    # Adding my inputs to the inputs of my outputs. Also removing self.
    for out in outNodes:  # Going through my outputs
        if out != toRemove:
            outNodesIn, outNodesOut = STG[
                out
            ]  # The input of the outputs being considered.
            for inn in inNodes:  # Going through my inputs
                if not inn in outNodesIn:  # If my input isn't already there
                    outNodesIn = outNodesIn + [inn]  # Add the input
            outNodesIn.remove(toRemove)  # Removing self aswell
            STG[out] = (outNodesIn, outNodesOut)  # Finalize.

    del STG[toRemove]  # Removing self from STG.

    del unknowns[toRemove]
    return unknowns, GA, NGA, STG


def identify(
    unknowns, GA, NGA, STG
):  # Identifies Guaranteed attractors, garanteed non-attractors, and unknown nodes.
    for unTuple in list(unknowns.keys()):
        inNodes, outNodes = STG[unTuple]
        if len(inNodes) == 0:
            NGA[unTuple] = unknowns[unTuple]
            del unknowns[unTuple]
        if list(outNodes)[0] == unTuple and len(outNodes) == 1:
            GA[unTuple] = unknowns[unTuple]
            del unknowns[unTuple]
    return unknowns, GA, NGA


def permutationWrapper(possibleStates):
    acc = []
    stateSoFar = []
    prob = 1
    acc = nodePermutations(stateSoFar, possibleStates, acc, prob)
    return acc


# Recursively does some jank
def nodePermutations(stateSoFar, possibleStates, acc, prob):
    if len(possibleStates) == 0:
        return acc + [(stateSoFar, prob)]
    else:
        for i in range(0, len(possibleStates[0])):
            # Tag along the state currently being explored
            tempSSF = stateSoFar + possibleStates[0][i][0]
            # Run the same function again, without the first possible state which is being explored
            acc = nodePermutations(
                tempSSF, possibleStates[1:], acc, prob * possibleStates[0][i][1]
            )
        return acc


# I don't think its ever used.
# Why is it here?
# Graph that represents the State Transotion Graph of a Boolean Network.
class dep_StateGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.G = nx.DiGraph()

    def addState(self, node):
        if node not in self.nodes:
            self.nodes += [node]
            self.G.add_node(str(node))

    def addEdge(self, edge):
        self.edges += [edge]
        self.G.add_edge(str(edge[0]), str(edge[1]))

    def printStateGraph(self):
        pos = nx.spring_layout(self.G, k=0.9)
        # nx.draw_networkx_nodes(self.G, pos, node_size=500)
        # nx.draw_networkx_edges(self.G, pos)
        labels = {}
        for node in self.G.nodes():
            labels[node] = node
        # nx.draw_networkx_labels(self.G, pos, labels, font_size=8)
        # plt.savefig("ST_graph.png")
        return self.G

    def getAttractors(self, verbal=False):
        attractors = attractorSetFinder(self.nodes, self.edges, verbal)
        return attractors


def attractorSetFinder(nodes, edges, verbal):
    attractors = list()
    n_nodes = len(nodes)
    # row-from node, col-to Node
    transMatrix = np.zeros((n_nodes, n_nodes), dtype=float)

    i = 0 if verbal else False
    for edge in edges:
        if verbal:
            print(str(i) + " out of " + str(len(edges)), end="\r")
            i += 1
        inNode = binListToInt(edge[0])
        outNode = binListToInt(edge[1])
        transMatrix[inNode][outNode] = edge[2]
    if verbal:
        print()
    adjMatrix = transMatrix > 0
    flags = computeFlags(adjMatrix)
    tags = copy.deepcopy(nodes)
    for i in range(0, len(nodes)):
        tags[i] = [nodes[i]]
    simplified = checkSimplified(flags)
    while not simplified:
        # Picking a node index
        i = pickUnconfirmedNode(flags)
        # Removing the node
        nodesIn = adjMatrix.T[i]
        nodesOut = adjMatrix[i]
        for j in range(0, len(nodesIn)):
            # If the considered node is an input node
            if nodesIn[j] == True and not tags[j] == None and not j == i:
                # Add the current tag to the input node's tags
                tags[j] = joinTags(tags[j], tags[i])
                # Connect the input node's outputs to current nodes outputs
                inNodeIndex = binListToInt(tags[j][0])
                adjMatrix[inNodeIndex] = np.logical_or(adjMatrix[inNodeIndex], nodesOut)
                adjMatrix[inNodeIndex][i] = False
        for j in range(0, len(nodesOut)):
            if nodesOut[j] == True and not tags[j] == None and not j == i:
                adjMatrix[i][j] = False
        adjMatrix[i][i] = False
        tags[i] = None
        flags = computeFlags(adjMatrix)
        simplified = checkSimplified(flags)
        if verbal:
            n_sorted = countSimplified(flags)
            print(str(n_sorted) + " out of " + str(flags.shape[0]), end="\r")
    for i in range(0, len(flags)):
        if flags[i][0] and not flags[i][1]:
            attractors.append(tags[i])
    if verbal:
        print("")
    return attractors


def computeFlags(adjMatrix):
    # Flags
    # First collumn true means that it has no outputs beside itself
    # second flag means that it has no inputs beside itself, and that it has outputs beside itself
    flags = np.ones((adjMatrix.shape[0], 2), dtype=bool)
    # Iterate over each row of the matrix, which means iterating over each of
    # the output vector for each node.

    for i in range(len(adjMatrix)):
        for j in range(len(adjMatrix[i])):
            # Found an output node which is not itself
            if adjMatrix[i][j] == True and not i == j:
                flags[i][0] = False
    # Same, buth with collumns, which shows the inputs for node i
    for i in range(len(adjMatrix)):
        # print(adjMatrix.T[i].astype(int))
        for j in range(len(adjMatrix.T[i])):
            if adjMatrix.T[i][j] == True and not i == j:
                # Found an input node which is not itself
                flags[i][1] = False
    return flags


def pickUnconfirmedNode(flags):
    for i in range(0, len(flags)):
        if not flags[i][0] and not flags[i][1]:
            return i
    return None


def checkSimplified(flags):
    nodeSorted = np.logical_or(flags.T[0], flags.T[1]).T
    allSorted = True
    for node in nodeSorted:
        allSorted = allSorted and node
    return allSorted


def countSimplified(flags):
    nodeSorted = np.logical_or(flags.T[0], flags.T[1]).T
    n_sorted = 0
    for node in nodeSorted:
        if node:
            n_sorted += 1
    return n_sorted


def joinTags(firstTags, secondTags):
    for tag in secondTags:
        if tag not in firstTags:
            firstTags = firstTags + [tag]
    return firstTags


def binListToInt(
    inList, b
):  # Since it's done backwards (most significant values on the right). this is backwards.
    out = 0
    for i in range(len(inList)):
        out += (b**i) * inList[i]
    return out


def runAllFunctions(self):
    """
    Having the current state as input, computes the ouptut for this node under each of the functions,
    and then adds up the probabilities for ending up at that node state
    """
    if not self.functions == None:
        outputValues = [([0], 0), ([1], 0)]
        inputValues = []
        for i in range(0, len(self.inputNodes)):
            inputValues += [int(self.inputNodes[i].value)]
        for function in self.functions:
            currentOutput = int(function[1](inputValues))
            if currentOutput not in outputValues:
                outputValues[currentOutput] = (
                    [currentOutput],
                    function[0] + outputValues[currentOutput][1],
                )
        return outputValues
    else:
        return [([0], 0.5), ([1], 0.5)]
