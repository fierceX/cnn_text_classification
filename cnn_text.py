import mxnet as mx
from mxnet import gluon
from mxnet.gluon import nn
from mxnet import nd
from mxnet import ndarray as nd
from mxnet import autograd
from data_helpers import *
import pandas as pd


class ConvConcat(nn.HybridBlock):
    def __init__(self,sentence_size,num_embed,**kwargs):
        super(ConvConcat,self).__init__(**kwargs)
        net1 = nn.HybridSequential()
        with net1.name_scope():
            net1.add(nn.Conv2D(channels=100,kernel_size=(3,num_embed),activation='relu'))
            net1.add(nn.MaxPool2D(pool_size=(sentence_size-3+1,1)))

        net2 = nn.HybridSequential()
        with net2.name_scope():
            net2.add(nn.Conv2D(channels=100,kernel_size=(4,num_embed),activation='relu'))
            net2.add(nn.MaxPool2D(pool_size=(sentence_size-4+1,1)))
        
        net3 = nn.HybridSequential()
        with net3.name_scope():
            net3.add(nn.Conv2D(channels=100,kernel_size=(5,num_embed),activation='relu'))
            net3.add(nn.MaxPool2D(pool_size=(sentence_size-5+1,1)))
        
        self.net1 = net1
        self.net2 = net2
        self.net3 = net3
    def hybrid_forward(self,F,x):
        pooled_outputs = []
        pooled_outputs.append(self.net1(x))
        pooled_outputs.append(self.net2(x))
        pooled_outputs.append(self.net3(x))
        
        total_filters = 100 * 3
        concat = F.Concat(*pooled_outputs, dim=1)
        h_pool = F.reshape(concat, (-1, total_filters))
        
        return h_pool

class ReshapeInput(nn.HybridBlock):
    def __init__(self,sentence_size,num_embed,**kwargs):
        super(ReshapeInput,self).__init__(**kwargs)
        self.sentence_size = sentence_size
        self.num_embed = num_embed
    def hybrid_forward(self,F,x):
        return F.reshape(x,(-1,1,self.sentence_size,self.num_embed))

def accuracy(output, label):
    return np.mean(output.argmax(axis=1)==label)

def get_valid_acc(net,valid):
    valid_acc = 0.
    for data,label in valid:
        output = net(data.as_in_context(mx.gpu()))
        valid_acc +=accuracy(nd.softmax(output).asnumpy(),label.asnumpy())
    return valid_acc/len(valid)

def GetData():
    train = pd.read_csv('./labeledTrainData.tsv', header=0, delimiter="\t", quoting=3)
    labels = train['sentiment']
    sentences = []
    for i in range(len(train['review'])):
        sentences.append(review_to_wordlist(train['review'][i]))

    # sentences, labels = load_data_and_labels()

    sentences_padded = pad_sentences(sentences)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    x, y = build_input_data(sentences_padded, labels, vocabulary)

    embed_size = 300
    sentence_size = x.shape[1]
    vocab_size = len(vocabulary)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]
    # split train/valid set
    x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
    y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]

    train_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_train,y_train), batch_size=64,shuffle=True)
    valid_data = gluon.data.DataLoader(gluon.data.ArrayDataset(x_dev,y_dev), batch_size=64,shuffle=True)
    return train_data,valid_data,embed_size,sentence_size,vocab_size

train_data,valid_data,embed_size,sentence_size,vocab_size = GetData()

net = nn.HybridSequential()
with net.name_scope():
    net.add(nn.Embedding(input_dim=vocab_size,output_dim=300))
    net.add(ReshapeInput(sentence_size,300))
    net.add(ConvConcat(sentence_size,300))
    net.add(nn.Dense(2))
net.initialize(ctx = mx.gpu())
# net.load_params('net.params',ctx=mx.gpu())
net.hybridize()

softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
trainer = gluon.Trainer(net.collect_params(), 'Adam')

for epoch in range(5):
    train_loss = 0.
    train_acc = 0.
    valid_acc = 0.
    for i,(data, label) in enumerate(train_data):
        with autograd.record():
            output = net(data.as_in_context(mx.gpu()))
            loss = softmax_cross_entropy(output, label.astype('float32').as_in_context(mx.gpu()))
        loss.backward()
        trainer.step(64)

        _loss = nd.mean(loss).asscalar()
        train_loss += _loss
        _acc = accuracy(nd.softmax(output).asnumpy(), label.asnumpy())
        train_acc += _acc

        if i % 20 == 0:
            valid_acc = get_valid_acc(net,valid_data)
            print("Loss: %f, Train acc %f Valid acc %f" % (train_loss/(i+1), train_acc/(i+1),valid_acc))

    print("Epoch %d. Loss: %f, Train acc %f Valid acc %f" % (
        epoch, train_loss/len(train_data), train_acc/len(train_data)))

net.save_params('net.params')
