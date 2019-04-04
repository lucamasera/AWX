import numpy as np
import networkx as nx
import keras

from itertools import chain


to_skip = ['root', 'GO0003674', 'GO0005575', 'GO0008150']


class arff_data():
    def __init__(self, arff_file, is_GO, is_test=False):
        self.X, self.Y, self.A, self.terms, self.g = parse_arff(arff_file=arff_file, is_GO=is_GO, is_test=is_test)
        self.to_eval = [t not in to_skip for t in self.terms]
        r_, c_ = np.where(np.isnan(self.X))
        m = np.nanmean(self.X, axis=0)
        for i, j in zip(r_, c_):
            self.X[i,j] = m[j]

            
def parse_arff(arff_file, is_GO=False, is_test=False):
    with open(arff_file) as f:
        read_data = False
        X = []
        Y = []
        g = nx.DiGraph()
        feature_types = []
        skip_last = False
        for l in f:
            if l.startswith('@ATTRIBUTE'):
                if l.startswith('@ATTRIBUTE class'):
                    h = l.split('hierarchical')[1].strip()
                    for branch in h.split(','):
                        terms = branch.split('/')
                        if is_GO:
                            g.add_edge(terms[1], terms[0])
                        else:
                            if len(terms)==1:
                                g.add_edge(terms[0], 'root')
                            else:
                                for i in range(2, len(terms) + 1):
                                    g.add_edge('.'.join(terms[:i]), '.'.join(terms[:i-1]))
                    nodes = sorted(g.nodes(), key=lambda x: (nx.shortest_path_length(g, x, 'root'), x) if is_GO else (len(x.split('.')),x))
                    nodes_idx = dict(zip(nodes, range(len(nodes))))
                    g_t = g.reverse()
                elif l.startswith('@ATTRIBUTE key'):
                    skip_last = True
                else:
                    _, _, f_type = l.split()
                    if f_type == 'numeric':
                        feature_types.append(lambda x: [float(x)] if x != '?' else [np.nan])
                    else:
                        cats = f_type[1:-1].split(',')
                        d = {key:keras.utils.to_categorical(i, len(cats)).tolist() for i,key in enumerate(cats)}
                        feature_types.append(lambda x: d.get(x, [np.nan]*len(cats)))
            elif l.startswith('@DATA'):
                read_data = True
            elif read_data:
                y_ = np.zeros(len(nodes))
                d_line = l.split('%')[0].strip().split(',')
                lab = d_line[len(feature_types)].strip()
                X.append(list(chain(*[feature_types[i](x) for i, x in enumerate(d_line[:len(feature_types)])])))
                for t in lab.split('@'):
                    y_[map(nodes_idx.get, nx.ancestors(g_t, t.replace('/', '.')))] = 1
                    y_[nodes_idx[t.replace('/', '.')]] = 1
                Y.append(y_)
        X = np.array(X)
        Y = np.stack(Y)

    return X, Y, np.array(nx.to_numpy_matrix(g, nodelist=nodes, order=nodes)), nodes, g


def initialize_dataset(name, datasets):
    is_GO, train, val, test = datasets[name]
    return arff_data(train, is_GO), arff_data(val, is_GO), arff_data(test, is_GO, True)