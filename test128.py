import torch
import matplotlib.pyplot as plt
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from data_loader import get_loader 
from model import EncoderCNN, DecoderRNN
from PIL import Image
import json
import scipy.misc

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def main(args):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(args.embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(args.embed_size, args.hidden_size, len(vocab), args.num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(args.encoder_path))
    decoder.load_state_dict(torch.load(args.decoder_path))




    ###### Prepare a batch of test images  #########
    data_loader = get_loader(args.image_path, args.caption_path, vocab, 
                             transform, args.batch_size,
                             shuffle=False, num_workers=args.num_workers) 


    arr_predicted=[]
    arr_actual=[]
    count=0
    f_a=open('actual_label_coco.txt','w+')
    f_p=open('predict_label_coco.txt','w+')

    for i, (images, captions, lengths) in enumerate(data_loader):
            #print(captions.shape,lengths)
            images = images.to(device)
            captions = captions.to(device)
            features = encoder(images)
            outputs = decoder.sample(features)
            for bnum in range(128):
                    output=outputs[bnum].cpu().numpy()
                    predicted_array=[]
                    for wid in output:
                            word=vocab.idx2word[wid]
                            if word == '<end>':
                                    break
                            predicted_array.append(word)

                    predicted=' '.join(predicted_array)
                    f_p.write(predicted)
                    f_p.write('\n')

                    actual_caption=captions[bnum]
                    actual_arr=[]
                    actual_caption=actual_caption.cpu().numpy()
                    for wid in actual_caption:
                            word=vocab.idx2word[wid]
                            actual_arr.append(word)
                    actual=' '.join(actual_arr)
                    f_a.write(actual)
                    f_a.write('\n')
                    print(predicted)
                    print(actual)
                    print(images[bnum].cpu().numpy().transpose(1,2,0).shape)
                    scipy.misc.imsave('temp_dir/{}.jpg'.format(bnum),images[bnum].cpu().numpy().transpose(1,2,0))
                    count=count+1
            f_a.close()
            f_p.close()
            return








    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str,default='data/resizedval2014', help='input image folder for generating caption')
    ##### New parsers added
    parser.add_argument('--caption_path',type=str,default='data/annotations/captions_val2014.json',help='captions folder for the test images')
    parser.add_argument('--batch_size',type=int,default=128,help='batch size of test data')
    parser.add_argument('--num_workers', type=int, default=2)
    ##############
    parser.add_argument('--encoder_path', type=str, default='models/encoder-5-3000.pkl', help='path for trained encoder')
    parser.add_argument('--decoder_path', type=str, default='models/decoder-5-3000.pkl', help='path for trained decoder')
    parser.add_argument('--vocab_path', type=str, default='data/vocab.pkl', help='path for vocabulary wrapper')
    
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1, help='number of layers in lstm')
    args = parser.parse_args()
    main(args)
