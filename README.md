# ChatBot
A simple implementation of RNN-based chat bot.

Applied attention mechanism, mixed teacher forcing and greedy approach in training,
and beam search in decoding. Deal with OOV problem in pre-trained word embeddings.

## Usage

#### Train Models

Simplest (dialog text and word embedding are required)

    python train.py --text ~/data/dialog_text.txt --embed ~/embeds/glove.txt

Specify hyperparameters (learning rate, batch size, epochs, teacher forcing ratio)

    python train.py [...] -r 0.01 -b 32 -e 50 --tfr 0.6

Specify file path (checkpoint directory)

    python train.py [...] --ckpt ~/test/checkpoints/
    
Training on CPU

    python train.py [...] --cpu-only
    

#### Chat with Trained Model

Simplest (dialog text is required)

    python chat.py --text ~/data/dialog_text.txt
    
Specify hyperparameters (mode, beam size)

    python chat.py [...] -m beam -k 5
    python chat.py [...] -m greedy
    
Specify file path

    python train.py [...] --ckpt ~/test/checkpoints/
    
Processing on CPU

    python train.py [...] --cpu-only