import numpy as np
from sklearn.datasets import load_boston
from sklearn.utils import shuffle, resample

class Node(object):
    def __init__(self, inbound_nodes=[]):

        # 进入该结点的所有结点
        self.inbound_nodes = inbound_nodes

        # 记录输入结点的所有输出结点
        self.outbound_nodes = []

        for n in self.inbound_nodes:

            n.outbound_nodes.append(self)

        self.gradients = {}

        self.value = None

    def forward(self):
        """
        根据输入结点计算输出值, 并将值保存在 self.value 中
        :return:
        """

        raise NotImplemented

    def backward(self):
        raise NotImplemented


class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):

        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]
            self.gradients[self] = grad_cost * 1

class CaculatNode(Node):
    def __init__(self, f, *nodes):
        Node.__init__(self, nodes)
        self.func = f

    def forward(self):
        self.value = self.func(map(lambda n: n.value, self.inbound_nodes))

class Add(Node):
    def __init__(self, *nodes):
        Node.__init__(self, nodes)


    def forward(self):
        self.value = sum(map(lambda n: n.value, self.inbound_nodes))
        ## when execute forward, this node caculate value as defined.


class Linear(Node):
    def __init__(self, nodes, weights, bias):
        Node.__init__(self, [nodes, weights, bias])

    def forward(self):
        inbound_nodes = self.inbound_nodes[0].value

        weights = self.inbound_nodes[1].value

        bias = self.inbound_nodes[2].value

        self.value = np.dot(inbound_nodes, weights) + bias

    def backward(self):

        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for n in self.outbound_nodes:

            grad_cost = n.gradients[self]

            self.gradients[self.inbound_nodes[0]] = np.dot(grad_cost, self.inbound_nodes[1].value.T)
            self.gradients[self.inbound_nodes[1]] = np.dot(self.inbound_nodes[0].value.T, grad_cost)
            self.gradients[self.inbound_nodes[2]] = np.sum(grad_cost, axis=0, keepdims=False)


class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1. / (1 + np.exp(-1 * x))

    def forward(self):
        self.x = self.inbound_nodes[0].value
        self.value = self._sigmoid(self.x)

    def backward(self):
        self.partial = self._sigmoid(self.x) * (1 - self._sigmoid(self.x))

        # y = 1 / (1 + e^-x)
        # y' = 1 / (1 + e^-x) (1 - 1 / (1 + e^-x))

        self.gradients = {n: np.zeros_like(n.value) for n in self.inbound_nodes}

        for n in self.outbound_nodes:
            grad_cost = n.gradients[self]

            self.gradients[self.inbound_nodes[0]] = grad_cost * self.partial

class MSE(Node):
    def __init__(self, y, a):
        Node.__init__(self, [y, a])


    def forward(self):
        y = self.inbound_nodes[0].value.reshape(-1, 1)
        a = self.inbound_nodes[1].value.reshape(-1, 1)
        assert(y.shape == a.shape)

        self.m = self.inbound_nodes[0].value.shape[0]
        self.diff = y - a

        self.value = np.mean(self.diff**2)


    def backward(self):
        self.gradients[self.inbound_nodes[0]] = (2 / self.m) * self.diff
        self.gradients[self.inbound_nodes[1]] = (-2 / self.m) * self.diff

def forward_and_backward(outputnode, graph):

    for n in graph:
        n.forward()

    for n in  graph[::-1]:
        n.backward()

def topological_sort(feed_dict):

    input_nodes = [n for n in feed_dict.keys()]

    G = {}
    nodes = [n for n in input_nodes]
    while len(nodes) > 0:
        n = nodes.pop(0)
        if n not in G:
            G[n] = {'in': set(), 'out': set()}
        for m in n.outbound_nodes:
            if m not in G:
                G[m] = {'in': set(), 'out': set()}
            G[n]['out'].add(m)
            G[m]['in'].add(n)
            nodes.append(m)

    L = []
    S = set(input_nodes)
    while len(S) > 0:
        n = S.pop()

        if isinstance(n, Input):
            n.value = feed_dict[n]

        L.append(n)
        for m in n.outbound_nodes:
            G[n]['out'].remove(m)
            G[m]['in'].remove(n)
            if len(G[m]['in']) == 0:
                S.add(m)
    return L

def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]

if __name__ == '__main__':

    data = load_boston()

    X_ = data['data']

    y_ = data['target']

    # Normalize data
    X_ = (X_ - np.mean(X_, axis=0)) / np.std(X_, axis=0)

    n_features = X_.shape[1]

    n_hidden = 10

    W1_ = np.random.randn(n_features, n_hidden)

    b1_ = np.zeros(n_hidden)

    W2_ = np.random.randn(n_hidden, 1)

    b2_ = np.zeros(1)

    # Neural network
    X, y = Input(), Input()

    W1, b1 = Input(), Input()

    W2, b2 = Input(), Input()

    l1 = Linear(X, W1, b1)

    s1 = Sigmoid(l1)

    l2 = Linear(s1, W2, b2)

    cost = MSE(y, l2)

    feed_dict = {
        X: X_,
        y: y_,
        W1: W1_,
        b1: b1_,
        W2: W2_,
        b2: b2_
    }

    epochs = 5000

    # total number of examples
    m = X_.shape[0]

    batch_size = 16

    setps_per_epoch = m // batch_size

    graph = topological_sort(feed_dict)

    trainables = [W1, b1, W2, b2]

    print("Total number of examples = {}".format(m))

    for i in range(epochs):
        loss = 0

        for j in range(setps_per_epoch):
            X_batch, y_batch = resample(X_, y_, n_samples=batch_size)

            X.value = X_batch

            y.value = y_batch

            _ = None

            forward_and_backward(_, graph)

            rate = 1e-2

            sgd_update(trainables, rate)

            loss += graph[-1].value

        if i % 100 == 0:
            print("Epoch: {}, Loss: {:.3f}".format(i + 1, loss / setps_per_epoch))
