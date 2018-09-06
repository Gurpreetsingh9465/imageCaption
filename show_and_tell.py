from model import ShowAndTellModel
from PIL import Image
import numpy as np
import argparse
import sys
import os

def create_arg_parser():
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('image',
                    help='Path to the image directory.')
    return parser

with open("dictionary.txt") as f:
    lines = f.read().split("\n")
   
word2token = {}
token2word = {}
for line in lines[:-1]:
    l = line.split('    ')
    word = l[0]
    token = int(l[1])
    word2token[word] = token
    token2word[token] = word
    
    
model = ShowAndTellModel('optimized.pb')
start_token="<S>"
end_token="</S>"


def getCaption(image):
    state = model.feed_image(image)
    cur_token = word2token[start_token]
    end = word2token[end_token]
    answere = ""
    for i in range(20):
        if cur_token == end:
            break
        t = np.array([cur_token])
        softmax,state,_ = model.inference_step(t,state)
        cur_token = np.argmax(softmax)
        if cur_token == word2token[end_token]:
            break
        answere += token2word[cur_token]+" "
    return answere
    
if __name__=="__main__":
    arg_parser = create_arg_parser()
    parsed_args = arg_parser.parse_args(sys.argv[1:])
    if os.path.exists(parsed_args.image):
       print("File exist")
       path = parsed_args.image
       image = Image.open(path)
       image = np.array(image)
       caption = getCaption(image)
       print("Caption : ",end="")
       print(caption)
    else:
        print("File Not Found {}".format(parsed_args.image))
        
        
        
        
        
        
    