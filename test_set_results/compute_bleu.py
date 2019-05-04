from nltk.translate.bleu_score import sentence_bleu
import json
import argparse

def main(args):
    actual_f=open(args.a_path)
    predicted_f=open(args.p_path)
    act=json.load(actual_f)['output']
    pre=json.load(predicted_f)['output']
    b1,b2,b3,b4=0.0,0.0,0.0,0.0

    for i in range(len(act)):
        pre_sent=pre[i].split(' ')[1:]
        last=act[i].split(' ').index('<end>')
        act_sent=act[i].split(' ')[1:last]
        b1=b1+sentence_bleu([pre_sent],act_sent,weights=(1,0,0,0))
        b2=b2+sentence_bleu([pre_sent],act_sent,weights=(0,1,0,0))
        b3=b3+sentence_bleu([pre_sent],act_sent,weights=(0,0,1,0))
        b4=b4+sentence_bleu([pre_sent],act_sent,weights=(0,0,0,1))
    print("Average BLEU-1 score is "+str(b1/len(act)))
    print("Average BLEU-2 score is "+str(b2/len(act)))
    print("Average BLEU-3 score is "+str(b3/len(act)))
    print("Average BLEU-4 score is "+str(b4/len(act)))







if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p_path', type=str, default='test_prediction.json' , help='path for predicted sentences file')
    parser.add_argument('--a_path', type=str, default='test_actual.json' , help='path for ground truth sentences file')
    args = parser.parse_args()
    main(args)
