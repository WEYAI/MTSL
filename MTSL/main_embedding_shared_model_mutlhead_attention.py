import sys, os
import torch
import time
import uuid
from torch.optim import SGD
from scipy.stats import bernoulli
import argparse

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/utils')
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/model')
import logger, embedding, io_utils, writer
import warnings
from embedding_shared_model_self_attention import EmbeddingSharedModelAttention

warnings.filterwarnings("ignore")
uid = uuid.uuid4().hex[:6]

parser = argparse.ArgumentParser(description='Embedding-Shared Model')
parser.add_argument('--rnn_mode', choices=['RNN', 'LSTM', 'GRU'], help='architecture of rnn')
parser.add_argument('--num_epochs', type=int, default=100, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=16, help='Number of sentences in each batch')
parser.add_argument('--hidden_size', type=int, default=128, help='Number of hidden units in RNN')
parser.add_argument('--num_layers', type=int, default=1, help='Number of layers of RNN')
parser.add_argument('--num_filters', type=int, default=30, help='Number of filters in CNN')
parser.add_argument('--window', type=int, default=30, help='Window size for CNN')
parser.add_argument('--char_dim', type=int, default=30, help='Dimension of Character embeddings')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
parser.add_argument('--decay_rate', type=float, default=0.05, help='Decay rate of learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD')
parser.add_argument('--gamma', type=float, default=0.0, help='weight for regularization')
parser.add_argument('--p_rnn', nargs=2, type=float, help='dropout rate for RNN')
parser.add_argument('--p_in', type=float, default=0.33, help='dropout rate for input embeddings')
parser.add_argument('--p_out', type=float, default=0.5, help='dropout rate for output layer')
parser.add_argument('--bigram', action='store_true', help='bi-gram parameter for CRF')
parser.add_argument('--schedule', type=int, help='schedule for learning rate decay')
parser.add_argument('--embedding_path', help='path for embedding dict')
parser.add_argument('--option_path', help='path for elmo option file')
parser.add_argument('--weight_path', help='path for elmo weight file')
parser.add_argument('--word2index_path', help='path for Word2Index')
parser.add_argument('--out_path', help='path for output')
parser.add_argument('--use_crf', help='use crf')
parser.add_argument('--use_lm', help='use lm')
parser.add_argument('--use_elmo', help='use elmo')
parser.add_argument('--lm_loss', type=float, default=0.05, help='lm loss scale')
parser.add_argument('--lm_mode', choices=['shared', 'unshared'], help='lm mode')
parser.add_argument('--label_type', nargs=2, help='label type')
parser.add_argument('--bucket_auxiliary', type=int, nargs='+', help='bucket auxiliary')
parser.add_argument('--bucket_main', type=int, nargs='+', help='bucket main')
parser.add_argument('--train', nargs=2)
parser.add_argument('--dev', nargs=2)
parser.add_argument('--test', nargs=2)

args = parser.parse_args()

args.rnn_mode = 'LSTM'
args.num_epochs = 200
args.batch_size = 128
args.hidden_size = 256
args.num_layers = 1
args.char_dim = 300
args.num_filters = 30
args.learning_rate = 0.01
args.decay_rate = 0.05
args.schedule = 1
args.gamma = 0.0
args.p_in = 0.33
args.p_rnn = [0.33, 0.5]
args.p_out = 0.5
args.lm_loss = 0.05
base_path = "/home/zutnlp/data/"
# args.embedding_path=base_path+"MTSL/MTSL/data/embedding/glove_300_embedding.txt"
# args.embedding_path = "data/embedding/glove_embedding.txt"  # this is 50 dimensions
args.embedding_path = "/home/zutnlp/data.copy/word2vec/baidubaike.bigram-char"  # this is 300 dimensions
args.option_path = "data/embedding/elmo_option.json"
args.weight_path = "data/embedding/elmo_weight.hdf5"
args.word2index_path = "output/embedding_shared_model/chunk/word2index"
args.out_path = "output/embedding_shared_model/chunk"
args.use_crf = "True"
args.use_lm = "False"
args.use_elmo = "False"
args.lm_mode = "unshared"
args.label_type = ["pos", "ner"]
args.bucket_auxiliary = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
args.bucket_main = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80]
args.train = [base_path + "ontonotes5.0/pos/train.txt", base_path + "ontonotes5.0/ner/train.txt"]
args.dev = [base_path + "ontonotes5.0/pos/dev.txt", base_path + "ontonotes5.0/ner/dev.txt"]
args.test = [base_path + "ontonotes5.0/pos/test.txt", base_path + "ontonotes5.0/ner/test.txt"]
# multihead attentino
# args.key_dim
rnn_mode = args.rnn_mode
train_path = args.train
dev_path = args.dev
test_path = args.test
num_epochs = args.num_epochs
batch_size = args.batch_size
hidden_size = args.hidden_size
num_filters = args.num_filters
learning_rate = args.learning_rate
decay_rate = args.decay_rate
gamma = args.gamma
schedule = args.schedule
p_rnn = tuple(args.p_rnn)
p_in = args.p_in
p_out = args.p_out
bigram = args.bigram
embedding_path = args.embedding_path
option_path = args.option_path
weight_path = args.weight_path
word2index_path = args.word2index_path
char_dim = args.char_dim
num_layers = args.num_layers
window = args.window
momentum = args.momentum
out_path = args.out_path
label_type = args.label_type
use_crf = args.use_crf
use_lm = args.use_lm
use_elmo = args.use_elmo
use_crf = io_utils.parse_bool(use_crf)
use_lm = io_utils.parse_bool(use_lm)
use_elmo = io_utils.parse_bool(use_elmo)
lm_loss = args.lm_loss
lm_mode = args.lm_mode
bucket_auxiliary = args.bucket_auxiliary
bucket_main = args.bucket_main
label_bucket = dict(zip(label_type, [bucket_auxiliary, bucket_main]))

logger = logger.get_logger("Embedding-Shared Model")
logger.info("Use Language Model: %s" % use_lm)
if use_lm:
    logger.info("Language Model mode: %s" % lm_mode)
logger.info("Use CRF: %s" % use_crf)
logger.info("Use ELMo: %s" % use_elmo)
embedd_dict, embedd_dim = embedding.load_embedding_dict(embedding_path)
logger.info("Creating Word2Indexs")
word_word2index, char_word2index, label_word2index_list, = \
    io_utils.create_word2indexs(word2index_path, train_path, label_type,
                                test_paths=[[dev, test] for dev, test in zip(dev_path, test_path)],
                                embedd_dict=embedd_dict, max_vocabulary_size=60000)
pass
logger.info("Word Word2Index Size: %d" % word_word2index.size())
logger.info("Character Word2Index Size: %d" % char_word2index.size())
for i in range(len(label_word2index_list)):
    logger.info("Label %d Word2Index Size: %d" % (i, label_word2index_list[i].size()))

logger.info("Reading Data")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
data_train = []
data_dev = []
data_test = []
num_data = []
num_labels = []
writers = []
for i in range(len(label_type)):
    data_train.append(io_utils.read_data_to_tensor(train_path[i], word_word2index, char_word2index,
                                                   label_word2index_list[i], device, label_type[i], label_bucket,
                                                   use_lm=use_lm, use_elmo=use_elmo))
    num_data.append(sum(data_train[i][1]))
    num_labels.append(label_word2index_list[i].size())
    data_dev.append(io_utils.read_data_to_tensor(dev_path[i], word_word2index, char_word2index,
                                                 label_word2index_list[i], device, label_type[i], label_bucket,
                                                 use_lm=use_lm, use_elmo=use_elmo))
    data_test.append(io_utils.read_data_to_tensor(test_path[i], word_word2index, char_word2index,
                                                  label_word2index_list[i], device, label_type[i], label_bucket,
                                                  use_lm=use_lm, use_elmo=use_elmo))
    writers.append(writer.Writer(label_word2index_list[i]))
if use_elmo:
    word_table = (option_path, weight_path)
else:
    word_table = io_utils.construct_word_embedding_table(embedd_dict, embedd_dim, word_word2index)

logger.info("constructing network...")
network = EmbeddingSharedModelAttention(
    embedd_dim, word_word2index.size(), char_dim, char_word2index.size(), num_labels, num_filters, window, rnn_mode,
    hidden_size, num_layers, embedd_word=word_table, p_in=p_in, p_out=p_out, p_rnn=p_rnn, lm_loss=lm_loss,
    bigram=bigram, use_crf=use_crf, use_elmo=use_elmo, use_lm=use_lm, lm_mode=lm_mode)
network.to(device)
optim = SGD(network.parameters(), lr=learning_rate, momentum=momentum, weight_decay=gamma, nesterov=True)
logger.info("Network: %s, num_layer=%d, hidden=%d, filter=%d, crf=%s" % (
    rnn_mode, num_layers, hidden_size, num_filters, 'bigram' if bigram else 'unigram'))
logger.info(
    "training: l2: %f, #train data: %d, batch: %d, dropout: %.2f" % (
        gamma, num_data[-1], batch_size, p_in))

num_batches = []
for i in range(len(label_type)):
    num_batches.append(int(num_data[i] / batch_size) + 1)
dev_f1 = 0.0
dev_acc = 0.0
dev_precision = 0.0
dev_recall = 0.0
test_f1 = 0.0
test_acc = 0.0
test_precision = 0.0
test_recall = 0.0
best_epoch = 0
lr = learning_rate
for epoch in range(1, num_epochs + 1):
    print('Epoch %d (%s, learning rate=%.4f, decay rate=%.4f (schedule=%d)): ' % (
        epoch, rnn_mode, lr, decay_rate, schedule))
    start_time = time.time()
    train_err = 0.
    train_total = 0.
    num_back = 0
    network.train()
    turn_list = bernoulli.rvs(float(num_batches[-1]) / sum(num_batches), size=sum(num_batches))
    batch = 0
    for turn in turn_list:
        batch += 1
        if turn == 1:
            main_task = True
        else:
            main_task = False
        if use_lm:
            word, char, labels, masks, lengths, word_fw, word_bw = \
                io_utils.get_batch_variable(data_train[turn], batch_size, use_lm=use_lm)
        else:
            word, char, labels, masks, lengths = \
                io_utils.get_batch_variable(data_train[turn], batch_size, use_lm=use_lm)
            word_fw = None
            word_bw = None
        optim.zero_grad()
        if use_crf:
            loss = network.loss(word, char, labels, main_task, word_fw, word_bw, masks, leading_symbolic=1)
        else:
            loss, _ = network.loss(word, char, labels, main_task, word_fw, word_bw, masks, leading_symbolic=1)
        loss.backward()
        optim.step()
        num_inst = word.size(0)
        train_err += loss.data * num_inst
        train_total += num_inst
        time_ave = (time.time() - start_time) / batch
        time_left = (sum(num_batches) - batch) * time_ave
        if batch % 100 == 0:
            sys.stdout.write("\b" * num_back)
            sys.stdout.write(" " * num_back)
            sys.stdout.write("\b" * num_back)
            log_info = 'train: %d/%d loss: %.4f, time left (estimated): %.2fs' % (
                batch, sum(num_batches), train_err / train_total, time_left)
            sys.stdout.write(log_info)
            sys.stdout.flush()
            num_back = len(log_info)
    acc_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    network.eval()
    for i in range(len(label_type)):
        if i == (len(label_type) - 1):
            main_task = True
        else:
            main_task = False
        tmp_filename = out_path + '/%s_dev_%s%d' % (str(uid), label_type[i], epoch)
        writers[i].start(tmp_filename)
        for batch in io_utils.iterate_batch_variable(data_dev[i], batch_size, label_type[i], label_bucket,
                                                     use_lm=use_lm):
            if use_lm:
                word, char, labels, masks, lengths, word_fw, word_bw = batch
            else:
                word, char, labels, masks, lengths = batch
                word_fw = None
                word_bw = None
            if use_crf:
                preds, _ = network.decode(word, char, labels, main_task, masks, leading_symbolic=1)
            else:
                _, preds = network.loss(word, char, labels, main_task, word_fw, word_bw, masks, leading_symbolic=1)
            writers[i].write(word.data.cpu().numpy(), preds.cpu().numpy(), labels.data.cpu().numpy(),
                             lengths.cpu().numpy(), use_elmo)
        writers[i].close()
        acc, precision, recall, f1 = io_utils.evaluate_f1(tmp_filename, out_path, uid)
        acc_list.append(acc)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)
        print(label_type[i] + ' dev acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%%' % (acc, precision,
                                                                                                   recall, f1))

    if dev_f1 < f1_list[-1]:
        dev_f1 = f1_list[-1]
        dev_acc = acc_list[-1]
        dev_precision = precision_list[-1]
        dev_recall = recall_list[-1]
        best_epoch = epoch
        for i in range(len(label_type)):
            if i == (len(label_type) - 1):
                main_task = True
            else:
                main_task = False
            # evaluate on test data when better performance detected
            tmp_filename = out_path + '/%s_test_%s%d' % (str(uid), label_type[i], epoch)
            writers[i].start(tmp_filename)
            for batch in io_utils.iterate_batch_variable(data_test[i], batch_size, label_type[i], label_bucket,
                                                         use_lm=use_lm):
                if use_lm:
                    word, char, labels, masks, lengths, word_fw, word_bw = batch
                else:
                    word, char, labels, masks, lengths = batch
                    word_fw = None
                    word_bw = None
                if use_crf:
                    preds, _ = network.decode(word, char, labels, main_task, masks, leading_symbolic=1)
                else:
                    _, preds = network.loss(word, char, labels, main_task, word_fw, word_bw, masks, leading_symbolic=1)
                writers[i].write(word.data.cpu().numpy(), preds.cpu().numpy(), labels.data.cpu().numpy(),
                                 lengths.cpu().numpy(), use_elmo)
            writers[i].close()
            test_acc, test_precision, test_recall, test_f1 = io_utils.evaluate_f1(tmp_filename, out_path, uid)
            print(label_type[i] + " best dev  acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)"
                  % (acc_list[i], precision_list[i], recall_list[i], f1_list[i], best_epoch))
            print(label_type[i] + " best test acc: %.2f%%, precision: %.2f%%, recall: %.2f%%, F1: %.2f%% (epoch: %d)"
                  % (test_acc, test_precision, test_recall, test_f1, best_epoch))
    if epoch % schedule == 0:
        lr = learning_rate / (1.0 + epoch * decay_rate)
        optim = SGD(network.parameters(), lr=lr, momentum=momentum, weight_decay=gamma, nesterov=True)
