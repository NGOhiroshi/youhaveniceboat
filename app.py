import pandas as pd
import numpy as np
import datetime
from flask import Flask, render_template, request, redirect
import os
import time
#Kazutoshimeter
import unicodedata
from janome.tokenizer import Tokenizer
from gensim.models import word2vec
import scipy.spatial as sp

import pickle

app = Flask(__name__)

#pathがどこにあるか確認
path=os.getcwd()
print(path)

## Kazutoshi meter 関数部
t = Tokenizer(wakati=False)

def get_surfaces(text):
    result = []
    for token in t.tokenize(text):
        partOfSpeech = token.part_of_speech.split(',')[0]
        if partOfSpeech in ['名詞','動詞','形容詞','副詞','助詞','助動詞']:
            result.append(token.surface)
    return " ".join(result)

num_features = 250

def avg_document_vector(data, num_features):
    document_vec = np.zeros((len(data), num_features))
    for i, doc_word_list in enumerate(data):
        feature_vec = np.zeros((num_features,), dtype="float32")
        for word in doc_word_list:
            try:
                feature_vec = np.add(
                    feature_vec, skipgram_model.wv.__getitem__(word))
            except:
                pass

        feature_vec = np.divide(feature_vec, len(doc_word_list))
        document_vec[i] = feature_vec
    return document_vec

def get_cosine_similarity(x, y):
    # cosine類似度　= 1 - cosine距離
    # sp.distance.cosine(x, y)の返り値はcosine距離
    return 1 - sp.distance.cosine(x, y)

f = open('./static/pickle/kazutoshimeter.pickle','rb')
kazu_matrix = pickle.load(f)
f.close

f1 = open('./static/pickle/skipgram.pickle','rb')
skipgram_model = pickle.load(f1)
f1.close

f2 = open('./static/pickle/music_dataset.pickle','rb')
dataset = pickle.load(f2)
f2.close


##日付取得
os.environ["TZ"] = "Asia/Tokyo"
time.tzset()
tday=datetime.date.today()
yday= tday - datetime.timedelta(days=1)
d_tday=tday.strftime('20%y%m%d')
d_yday=yday.strftime('20%y%m%d')

##本日のレース予測
#d_tday= '20200612' #特別に設定したいとき 
df_pred = pd.read_csv(str(path)+'/static/report/predict_'+d_tday+'.csv')
pred_num_tday = len(df_pred)
header = df_pred.columns
record = df_pred.values.tolist()

##昨日の予測結果
#d_yday= '20200612' #特別に設定したいとき
df_ans = pd.read_csv(str(path)+'/static/report/result_'+d_yday+'.csv')
df_ans = df_ans.fillna("")
header_ans = df_ans.columns
record_ans = df_ans.values.tolist()
pred_num = len(df_ans)-len(df_ans[df_ans["Result"]=="Cancel"])
hit_num = len(df_ans[df_ans["Result"]=="Hit"])
hit_ratio = round(hit_num/pred_num*100,1)
pay = 100*pred_num
getmoney = df_ans["Payoff"].sum()
pay_return = round(getmoney/pay*100,1)

if hit_ratio >= 10:
    msg = "的中率10%超えるなんて..昨日は良い日でしたね"
elif hit_ratio > 6:
    msg = "なんともネタにもしにくい的中率.."
else:
    msg = "こういう日もあるよね。前を向いて歩こう"

if pay_return > 100:
    msg2 = "今日はちょっと良いビールでも買いましょうか"
elif pay_return > 75:
    msg2 = "お金を稼ぐって大変ですね"
else:
    msg2 = "涙の数だけ強くなれるよ"
    



##各ページへのレンダリング
@app.route('/', methods = ['POST', 'GET'])
def top_page():
    if request.method == "GET":
        return render_template('top_page.html')
    else:
        print('POSTされました')
        kazutoshi = request.form["kazutoshi"]
        
        df = pd.DataFrame(columns=['Title','Sentence'])
        df.loc[0,'Title']='入力値'
        df.loc[0,'Sentence']=kazutoshi
        df["Sentence"] = df["Sentence"].str.normalize("NFKC")
    
        text_tokenized = []
        for text in df["Sentence"]:
            text_tokenized.append(get_surfaces(text))
        df['Sentence_tokenized'] = text_tokenized
        
        sentences_input = [token.split(" ") for token in df.Sentence_tokenized]
        input_matrix = avg_document_vector(data=sentences_input, num_features=250)
        
        kazu_similarity = []

        # content_idxを変更することで、ベースとなる記事を変更できます
        content_idx = 0
        for i in range(len(kazu_matrix)):
            # cosine類似度　= 1 - cosine距離
            sim = get_cosine_similarity(input_matrix[content_idx], kazu_matrix[i])
            kazu_similarity.append(sim)

        kazu_similarity = np.array(kazu_similarity)

        topN = 1

        # 類似度の低い順にソートした結果のインデックスを用意して、降順に並び替えます
        arg_sort = np.argsort(kazu_similarity)
        arg_sort = arg_sort[::-1]

        # 比較対象のベースとなる記事を除いた類似度の高い記事を選びます
        selected_kazu_idx = arg_sort[0:topN]
        
        print(dataset.loc[selected_kazu_idx,["Sentence","Title"]])
        lyrics = "~ "+dataset.loc[selected_kazu_idx,["Sentence"]].values[0][0] + " ~"
        title = dataset.loc[selected_kazu_idx,["Title"]].values[0][0] + " / Mr.children"
        
        return render_template('top_page.html',lyrics=lyrics,title=title)

    
@app.route('/thissite')
def this_site():
    return render_template('this_site.html')

@app.route('/disclaimer')
def disclaimer():
    return render_template('disclaimer.html')

@app.route('/tshirts')
def tshirts():
    return render_template('tshirts.html')

@app.route('/prediction')
def prediction():
    return render_template('prediction.html', tday=tday,header=header,record=record,pred_num=pred_num_tday)

@app.route('/result')
def result():
    return render_template('result.html', yday=yday,header=header_ans,record=record_ans,msg=msg,msg2=msg2,\
                          pred_num=pred_num,hit_num=hit_num,\
                           hit_ratio=hit_ratio,pay=pay,getmoney=getmoney,\
                          pay_return=pay_return)

#テスト用のページ
@app.route('/hello_earth')
def hello_earth():
    name = 'Universe'
    return render_template('hello_earth.html', name=name)

if __name__ == '__main__':
    app.run(host='0.0.0.0',debug=True)
