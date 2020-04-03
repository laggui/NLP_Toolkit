from nlptoolkit.utils.config import Config
from nlptoolkit.punctuation_restoration.infer import infer_from_trained

if __name__ == '__main__':
    config = Config(task='punctuation_restoration') # loads default argument parameters as above
    config.data_path = "/project/cq-training-1/project2/teams/team14/data/unaligned.en" # sets training data path
    config.batch_size = 128 # 32
    config.lr = 5e-4 # change learning rate
    config.model_no = 1 # sets model to PuncLSTM
    config.max_encoder_len = 96 # max seq len
    config.max_decoder_len = 96
    config.num_epochs = 65
    config.checkpoint_path = '/project/cq-training-1/project2/teams/team14/punc_restoration/'
    # initiate infer object, which loads the model for inference
    inferer = infer_from_trained(config)
    # infer from input file
    inferer.infer_from_file(in_file="/project/cq-training-1/project2/teams/team14/data/train.lang1",
                            out_file="/project/cq-training-1/project2/teams/team14/data/train_punctuated.lang1")
