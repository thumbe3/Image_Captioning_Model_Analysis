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
                             shuffle=True, num_workers=args.num_workers) 


    actual=[]
    predicted=[]
    count=0
    pdict={}
    adict={}

    for i, (images, captions, lengths) in enumerate(data_loader):
            #print(captions.shape,lengths)
            images = images.to(device)
            captions = captions.to(device)
            features = encoder(images)
            outputs = decoder.sample(features)
            for bnum in range(len(outputs)):
                    output=outputs[bnum].cpu().numpy()
                    predicted_array=[]
                    for wid in output:
                            word=vocab.idx2word[wid]
                            if word == '<end>':
                                    break
                            predicted_array.append(word)

                    predicted.append(' '.join(predicted_array))
                    
                    actual_caption=captions[bnum]
                    actual_arr=[]
                    actual_caption=actual_caption.cpu().numpy()
                    for wid in actual_caption:
                            word=vocab.idx2word[wid]
                            actual_arr.append(word)
                    actual.append(' '.join(actual_arr))
                    if count%128==0:
                        pdict['output']=predicted
                        adict['output']=actual
                        with open('test_set_results/test_prediction.json', 'w') as outfile:
                            json.dump(pdict,outfile)
                        with open('test_set_results/test_actual.json', 'w') as outfile:
                            json.dump(adict,outfile)
                    count=count+1

    pdict['output']=predicted
    adict['output']=actual
    with open('test_set_results/test_prediction.json', 'w') as outfile:
        json.dump(pdict,outfile)
    with open('test_set_results/test_actual.json', 'w') as outfile:
        json.dump(adict,outfile)
        
    







    
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
