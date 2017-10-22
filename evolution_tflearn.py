import random
import copy

import tensorflow

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression


tflearn_activations = ['linear', 'tanh', 'sigmoid', 'softplus', 'softsign', 'relu', 'relu6', 'leaky_relu', 'prelu', 'elu', 'crelu']
# removed: 'softmax' because only used in categorization nets (in the last layer)
# removed: 'selu' because of stacktrace

tflearn_regularizers = [None, 'L2', 'L1']


class AbstractGenome:
    def mutate(self, rand):
        raise NotImplementedError()

    def create_network(self, network):
        raise NotImplementedError()


class Genome(AbstractGenome):
    def __init__(self, name, genome_code):
        self.name = name
        self.genome_code = genome_code
        self.descr = str(genome_code)
        self.child_count = 0

        self.train_accuracy = 0
        self.test_accuracy = 0
        self.train_confusion = [[]]
        self.test_confusion = [[]]

    def create_network(self, network):
        return self.genome_code.create_network(network)

    def create_mutant(self, rand):
        self.child_count += 1
        mutant_genome_code = copy.deepcopy(self.genome_code)
        mutant_genome_code.mutate(rand)
        return Genome(self.name + "." + str(self.child_count), mutant_genome_code)

    def mutate(self, rand):
        raise NotImplementedError()

    def fitness(self):
        return (self.train_accuracy + self.test_accuracy) / 2

    def __str__(self):
        return "Genome({}, {}, {})"\
            .format(self.name,
                    self.fitness(),
                    self.descr)


class Conv2DGenome(AbstractGenome):
    sizes = [2**x for x in range(8)]
    strides = [x + 1 for x in range(8)]
    activations = tflearn_activations
    regularizers = tflearn_regularizers
    max_pools = [x for x in range(4)]

    def __init__(self):
        self.size = 32
        self.stride = 1
        self.activation = 'relu'
        self.regularizer = 'L2'
        self.max_pool = 2

    def __str__(self):
        return "Conv2D({}, {}, {}, {}, {})"\
            .format(self.size,
                    self.stride,
                    self.activation,
                    self.regularizer,
                    self.max_pool)

    def mutate(self, rand):
        r = rand.randint(0, 3)
        if r == 0:
            self.size = rand.choice(Conv2DGenome.sizes)
        elif r == 1:
            self.stride = rand.choice(Conv2DGenome.strides)
        elif r == 2:
            self.activation = rand.choice(Conv2DGenome.activations)
        elif r == 3:
            self.regularizer = rand.choice(Conv2DGenome.regularizers)
        elif r == 4:
            self.max_pool = rand.choice(Conv2DGenome.max_pools)

    def create_network(self, network):
        network = conv_2d(network, self.size, self.stride, activation=self.activation, regularizer=self.regularizer)
        if self.max_pool > 0:
            network = max_pool_2d(network, self.max_pool)
        network = local_response_normalization(network)
        return network


class FullyConnectedGenome(AbstractGenome):
    sizes = [2**(x+2) for x in range(8)]
    activations = tflearn_activations

    def __init__(self):
        self.size = 32
        self.activation = 'tanh'
        self.dropout = 0.8

    def __str__(self):
        return "FullyConnected({}, {}, {})"\
            .format(self.size,
                    self.activation,
                    self.dropout)

    def mutate(self, rand):
        r = rand.randint(0, 2)
        if r == 0:
            self.size = rand.choice(Conv2DGenome.sizes)
        elif r == 1:
            self.activation = rand.choice(Conv2DGenome.activations)
        elif r == 2:
            self.dropout = rand.randint(0, 10) / 10

    def create_network(self, network):
        network = fully_connected(network, self.size, activation=self.activation)
        if self.dropout > 0:
            network = dropout(network, self.dropout)
        return network


class ConvNetGenome(AbstractGenome):
    template_nodes = [Conv2DGenome(), FullyConnectedGenome()]

    def __init__(self, max_count=6):
        self.nodes = list()
        self.max_count = max_count

    def __str__(self):
        result = "ConvNet("
        for i, n in enumerate(self.nodes):
            if i > 0:
                result += ", "
            result += str(n)
        return result + ")"

    def mutate(self, rand):
        r = rand.randint(0, 5)
        if (r == 0 or len(self.nodes) == 0) and len(self.nodes) < self.max_count:
            self.nodes.insert(rand.randint(0, len(self.nodes)), self.create_node(rand))
            return

        r = rand.randint(0, 5)
        if r == 0 and len(self.nodes) > 1:
            del(self.nodes[rand.randint(0, len(self.nodes) - 1)])
            return

        r = rand.randint(0, len(self.nodes) - 1)
        self.nodes[r].mutate(rand)

    def create_node(self, rand):
        node = copy.deepcopy(rand.choice(ConvNetGenome.template_nodes))
        node.mutate(rand)
        return node

    def create_network(self, input_tensor):
        network = input_tensor
        for n in self.nodes:
            network = n.create_network(network)
        return network


class Evolution:
    def __init__(self, train_data, train_labels, validation_part, test_data, test_labels, input_shape, num_classes, num_epoch=10, data_augmentation=None):
        self.train_data = train_data
        self.train_labels = train_labels
        self.validation_part = validation_part
        self.test_data = test_data
        self.test_labels = test_labels
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_epoch = num_epoch
        self.data_augmentation = data_augmentation

        self.rand = random.Random()
        self.genomes = list()

        self.genomes.append(Genome("1", ConvNetGenome()))

    def run(self, count=10):
        for i in range(count):
            parent = self.pick_random()
            child = parent.create_mutant(self.rand)
            self.add(child)
            self.print_genomes()

    def print_genomes(self):
        print("All genomes:")
        for genome in self.genomes:
            print(genome)
            print("Accuracy train={}")
            print(genome.train_accuracy)
            print("Accuracy test={}")
            print(genome.test_accuracy)
            print("Confusion train={}")
            print(genome.train_confusion)
            print("Confusion test={}")
            print(genome.test_confusion)
            print()

    def pick_random(self):
        max_fitness = 0
        for genome in self.genomes:
            max_fitness = max(genome.fitness(), max_fitness)
        if max_fitness == 0:
            max_fitness = 1

        sum_norm_fitness = 0
        for genome in self.genomes:
            norm_fitness = genome.fitness() / max_fitness
            norm_fitness = norm_fitness * norm_fitness * norm_fitness
            sum_norm_fitness += norm_fitness

        r = self.rand.uniform(0, sum_norm_fitness)
        sum_norm_fitness = 0
        for genome in self.genomes:
            norm_fitness = genome.fitness() / max_fitness
            norm_fitness = norm_fitness * norm_fitness * norm_fitness
            sum_norm_fitness += norm_fitness
            if r <= sum_norm_fitness:
                return genome

        return self.rand.choice(self.genomes)

    def add(self, new_genome):
        for genome in self.genomes:
            if genome.descr == new_genome.descr:
                return

        self.genomes.append(new_genome)
        self.calculate_fitness(new_genome)

    def calculate_fitness(self, genome):
        print("Calculating ", genome)
        try:
            tensorflow.reset_default_graph()

            network = input_data(shape=self.input_shape, name='input', data_augmentation=self.data_augmentation)
            network = genome.create_network(network)
            network = fully_connected(network, self.num_classes, activation='softmax')
            network = regression(network,
                                 optimizer='adam',
                                 learning_rate=0.0001,
                                 to_one_hot=True,
                                 n_classes=self.num_classes,
                                 loss='categorical_crossentropy',
                                 name='target')

            model = tflearn.DNN(network, tensorboard_verbose=0)

            model.fit({'input': self.train_data}, {'target': self.train_labels},
                      n_epoch=self.num_epoch,
                      validation_set=self.validation_part,
                      snapshot_step=100,
                      show_metric=True,
                      run_id=genome.name)

            genome.train_accuracy, genome.train_confusion = self.calculate_metrics(model, self.train_data, self.train_labels)
            genome.test_accuracy, genome.test_confusion = self.calculate_metrics(model, self.test_data, self.test_labels)
        except:
            pass

        print("Calculated ", genome)

    def calculate_metrics(self, model, data, labels):
        predict_labels = [x[0] for x in model.predict_label(data)]

        success_count = 0
        for i in range(len(labels)):
            if labels[i] == predict_labels[i]:
                success_count += 1
        accuracy = success_count / len(labels)

        with model.session.as_default():
            confusion_matrix = tensorflow.confusion_matrix(labels, predict_labels).eval()

        return accuracy, confusion_matrix

def run_test():
    rand = random.Random()
    orig_genome = ConvNetGenome()
    genome = orig_genome
    print("Orig: ", genome)
    for i in range(100):
        genome = copy.deepcopy(genome)
        genome.mutate(rand)
        print(genome)


if __name__ == '__main__':
    run_test()
