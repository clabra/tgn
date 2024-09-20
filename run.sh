################################
# https://github.com/clabra/tgn
################################

################################
# Model Training 
################################
# Self-supervised learning using the link prediction task:

# TGN-attn: Supervised learning on the wikipedia dataset
python train_self_supervised.py --use_memory --prefix tgn-attn --n_runs 10

# TGN-attn-reddit: Supervised learning on the reddit dataset
python train_self_supervised.py -d reddit --use_memory --prefix tgn-attn-reddit --n_runs 10

# Supervised learning on dynamic node classification (this requires a trained model from the self-supervised task, by eg. running the commands above):

# TGN-attn: self-supervised learning on the wikipedia dataset
python train_supervised.py --use_memory --prefix tgn-attn --n_runs 10

# TGN-attn-reddit: self-supervised learning on the reddit dataset
python train_supervised.py -d reddit --use_memory --prefix tgn-attn-reddit --n_runs 10


################################
# Baselines
################################
### Wikipedia Self-supervised

# Jodie
python train_self_supervised.py --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --n_runs 10

# DyRep
python train_self_supervised.py --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --n_runs 10


### Reddit Self-supervised

# Jodie
python train_self_supervised.py -d reddit --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn_reddit --n_runs 10

# DyRep
python train_self_supervised.py -d reddit --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn_reddit --n_runs 10


### Wikipedia Supervised

# Jodie
python train_supervised.py --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn --n_runs 10

# DyRep
python train_supervised.py --use_memory --memory_updater rnn --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn --n_runs 10


### Reddit Supervised

# Jodie
python train_supervised.py -d reddit --use_memory --memory_updater rnn --embedding_module time --prefix jodie_rnn_reddit --n_runs 10

# DyRep
python train_supervised.py -d reddit --use_memory --memory_updater rnn  --dyrep --use_destination_embedding_in_message --prefix dyrep_rnn_reddit --n_runs 10
