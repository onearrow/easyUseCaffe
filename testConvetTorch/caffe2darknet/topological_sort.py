from collections import deque

def topological_sort(graph):
    order, enter, state = deque(), set(graph), {}

    GRAY, BLACK = 0, 1

    def dfs(node):
        state[node] = GRAY
        for k in graph.get(node, ()):
            sk = state.get(k, None)
            if sk == GRAY: raise ValueError("cycle")
            if sk == BLACK: continue
            enter.discard(k)
            dfs(k)
        order.appendleft(node)
        state[node] = BLACK

    while enter: dfs(enter.pop())
    return order

if __name__ == '__main__':

    # Simple:
    # a --> b
    #   --> c --> d
    #   --> d
    graph1 = {
        "a": ["b", "c", "d"],
        "b": [],
        "c": ["d"],
        "d": []
    }

    # 2 components
    graph2 = {
        "a": ["b", "c", "d"],
        "b": [],
        "c": ["d"],
        "d": [],
        "e": ["g", "f", "q"],
        "g": [],
        "f": [],
        "q": []
    }

    # cycle
    graph3 = {
        "a": ["b", "c", "d"],
        "b": [],
        "c": ["d", "e"],
        "d": [],
        "e": ["g", "f", "q"],
        "g": ["c"],
        "f": [],
        "q": []
    }

    # a --> b1 --> c1 --> d1
    #   --> b2 --> c2 --> d1
    #   --> b3 --> c3 --> d2
    # d1 --> d3
    # d2 --> d3
    graph4 = {
        "a": ["b1", "b2", "b3"],
        "b1": ["c1"],
        "c1": ["d1"],
        "b2": ["c2"],
        "c2": ["d1"],
        "b3": ["c3"],
        "c3": ["d2"],
        "d1": ["d3"],
        "d2": ["d3"],
    }
    # check how it works
    print(topological(graph1))
    print(topological(graph2))
    print(topological(graph4))
    try: topological(graph3)
    except ValueError: print("Cycle!")

# The MIT License (MIT)
# Copyright (c) 2014 Alexey Kachayev
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
# is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
# ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
