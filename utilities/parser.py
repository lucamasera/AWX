import numpy as np
import sys


class arff():

    def __init__(self, path=None, is_GO=False, is_train=False, keep_root=True, auto_load=True):
        assert not (path is None and auto_load)
        self.is_GO = is_GO
        self.keep_root = keep_root
        self.is_train = is_train
        self.auto_load = auto_load

        self.attributes = {}
        self.description = ''
        self.is_numeric = []
        self.relation = ''
        self.classes = ''
        self.data = []
        
        self.parents = self.childrens = None

        self.A = self.R = self.C = self.CD = self.D = self.X = self.Y = None
        self.label_map = None
        
        if auto_load:
            self.load_arff(path)
            self.transform_labels()
            self.transform_data()

            self.validate_data()

    def __unicode__(self):
        return 'name: {}\nattributes: {}\nexamples: {}\nlabels: {}\n'.format(self.relation, self.X.shape[1], self.X.shape[0], self.Y.shape[1])

    def __str__(self):
        return self.__unicode__()

    def load_arff(self, path):
        with open(path, 'rb') as f:
            for line in f:
                line = line.strip()
                if line:
                    if line.startswith('@'):
                        if line.startswith('@ATTRIBUTE class'):
                            self.classes = line.split()[3]
                        elif line.startswith('@ATTRIBUTE'):
                            attribute_line = line.split()
                            if attribute_line[2].lower() in self.attributes:
                                self.attributes[attribute_line[2].lower()].append(attribute_line[1])
                            else:
                                self.attributes[attribute_line[2].lower()] = [attribute_line[1]]
                            self.is_numeric.append(attribute_line[2].lower() == 'numeric')
                        elif line.startswith('@RELATION'):
                            relation_line = line.split()
                            self.relation = relation_line[1].replace('\'', '')
                    else:
                        data_line = line.split(',')
                        self.data.append((data_line[-1], map(lambda x: float(x) if x != '?' else np.nan, [d for n, d in zip(self.is_numeric, data_line[:-1]) if n])))

    def transform_labels(self):
        classes = set(['root'])
        self.parents = dict()
        self.childrens = dict()

        for p in self.classes.split(','):
            path = p.split('/')
            if self.is_GO:
                a,b = path
            else:
                if len(path)==1:
                    a = 'root'
                    b = path[0]
                else:
                    a = '.'.join(path[:-1])
                    b = '.'.join(path)

            classes.update([a,b])

            if b in self.parents:
                self.parents[b].append(a)
            else:
                self.parents[b] = [a]

            if a in self.childrens:
                self.childrens[a].append(b)
            else:
                self.childrens[a] = [b]
        
        self.classes = classes
        
        self.A = np.zeros((len(self.classes), len(self.classes)), dtype=np.int0)
        
        i = 0
        parent_to_add = ['root']
        self.label_map = {'root':-1}
        
        while parent_to_add:
            p = parent_to_add.pop(0)
            kids = self.childrens.get(p, [])
            parent_to_add.extend(kids)
            for k in kids:
                if k not in self.label_map:
                    self.label_map[k] = i
                    i += 1
                self.A[self.label_map[k], self.label_map[p]] = 1
        
        self.idx_map = {v:k for k,v in self.label_map.items()}
        
        self.A = self.A[:-1,:]
        self.A = self.A[:,:-1]

        
    def transform_data(self):
        if self.auto_load:
            self.X = np.empty((len(self.data), len(self.attributes['numeric'])))
            self.Y = np.zeros((len(self.data), self.A.shape[0]))

            for i, (y,x) in enumerate(self.data):
                for y_ in y.replace('/', '.').split('@'):
                    self.X[i] = np.array(x)
                    self.Y[i, self.label_map[y_]] = 1
                    to_consider = [z for z in self.parents[y_]]
                    while to_consider:
                        p = to_consider.pop(0)
                        if p!='root':
                            to_consider.extend(self.parents[p])
                            self.Y[i, self.label_map[p]] = 1
        else:
            tmp = self.Y
            self.Y = np.zeros((len(tmp), self.A.shape[0]))
            for i, y_ in enumerate(tmp):
                self.Y[i, self.label_map[y_]] = 1
                to_consider = [z for z in self.parents[y_]]
                while to_consider:
                    p = to_consider.pop(0)
                    if p!='root':
                        to_consider.extend(self.parents[p])
                        self.Y[i, self.label_map[p]] = 1

    def validate_data(self):
        self.X = np.nan_to_num(self.X)