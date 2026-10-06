# I stole this from cabean python
import pickle

from colomoto.types import Hypercube, PartialState

from gym_PBN.utils.get_cabean_model import get_model


def parse_state(spec):
    spec = spec[0:len(spec):2]
    return PartialState([(f"x{i}", int(v) if v != "-" else "*") for i, v in enumerate(spec)])


def parse_attractors(cabean_out):
    attractors = {}
    num = None
    for line in cabean_out.split('\n'):
        if line.startswith("=") and "=== find attractor #" in line:
            parts = line.split()
            num = int(parts[3][1:]) - 1
            size = int(parts[5])
        elif num is not None:
            if line.startswith(":"):
                pass
            elif not line:
                num = None
            else:
                state = parse_state(line.split()[0])
                state = Hypercube(state)
                if num not in attractors:
                    attractors[num] = [tuple(state.values())]
                else:
                    print("appending to ", attractors[num])
                    attractors[num].append(tuple(state.values()))
    return attractors


def get_attractors(env):
    filename = f"data/attractors_{env.name}.pkl"
    #
    # try:
    #     with open(filename, 'rb') as attractors_file:
    #         return pickle.load(attractors_file)
    #
    # except FileNotFoundError:
    # wb+ does not erase the original file
    with open(filename, 'wb+') as attractors_file:
        cabean_out = get_model(env)
        attractors = list(parse_attractors(cabean_out).values())
        print("i generated:")
        print(attractors)
        pickle.dump(attractors, attractors_file)
        return attractors

# for testing
sample_cabean_out = r"""***************************************************************************
                       CABEAN 2.0.0 
 Please check http://satoss.uni.lu/software/CABEAN/ for the latest release.
 Please send any feedback to <cui.su@uni.lu>
***************************************************************************

Command line: cabean model_from_jinja.ispl
======================== find attractor #1 : 4 states ========================
: 6 nodes 1 leaves 4 minterms
1-0-1-0-----1-  1

======================== find attractor #2 : 1 states ========================
: 8 nodes 1 leaves 1 minterms
1-0-1-1-1-1-0-  1

======================== find attractor #3 : 1 states ========================
: 8 nodes 1 leaves 1 minterms
1-0-1-1-1-1-1-  1

======================== find attractor #4 : 1 states ========================
: 8 nodes 1 leaves 1 minterms
1-1-1-1-1-1-0-  1

number of attractors = 4
time for attractor detection=0
"""

# print(set(parse_attractors(sample_cabean_out).values()))
