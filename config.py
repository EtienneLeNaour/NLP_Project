###preprocessing options###
word_minfreq=3

###hyper-parameters###
seed=0
batch_size=32
keep_prob=0.7
epoch_size=30
max_grad_norm=5
#language model
word_embedding_dim=100
word_embedding_model="pretrain_word2vec/dim100/word2vec.bin"
lm_enc_dim=200
lm_dec_dim=600
lm_dec_layer_size=1
lm_attend_dim=25
lm_learning_rate=0.2


###sonnet hyper-parameters###
bptt_truncate=2 #number of sonnet lines to truncate bptt
doc_lines=14 #total number of lines for a sonnet

###misc###
verbose=False
save_model=True

###input/output###
output_dir="output"
output_prefix = "model"

train_data="data/sonnet_train.txt"
valid_data="data/sonnet_valid.txt"
test_data="data/sonnet_test.txt"


