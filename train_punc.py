from nlptoolkit.utils.config import Config
from nlptoolkit.punctuation_restoration.trainer import train_and_fit
from nlptoolkit.punctuation_restoration.infer import infer_from_trained

config = Config(task='punctuation_restoration') # loads default argument parameters as above
config.data_path = "./data/train.tags.en-fr.en" # sets training data path
config.batch_size = 128 # 32
config.lr = 5e-5 # change learning rate
config.model_no = 1 # sets model to PuncLSTM
config.max_encoder_len = 100 # max seq len
config.max_decoder_len = 100
train_and_fit(config) # starts training with configured parameters

# inferer = infer_from_trained(config) # initiate infer object, which loads the model for inference, after training model
# inferer.infer_from_input() # infer from user console input
# inferer.infer_from_file(in_file="./data/input.txt", out_file="./data/output.txt") # infer from input file