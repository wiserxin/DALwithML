import os
import re
import string
from random import randint
from textattack.augmentation import EmbeddingAugmenter

# # demo code
# augmenter = EmbeddingAugmenter(transformations_per_example=6)
# s = 'What I cannot create, I do not understand.'
# print(augmenter.augment(s))
# >> ['Quoi I cannot create, I do not understand.',
#  'Whar I cannot create, I do not understand.',
#  'What I cannot create, I do not comprehend.',
#  'What I cannot creating, I do not understand.',
#  'What I notable create, I do not understand.',
#  'What I significant create, I do not understand.']

def removePunctuation(text):
    temp = []
    for c in text:
        if c not in string.punctuation:
            temp.append(c)
    newText = ''.join(temp)
    return newText

def clean_str(string):
    # remove stopwords
    # string = ' '.join([word for word in string.split() if word not in cachedStopWords])
    string = removePunctuation(string)
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def generateLongSentence(text,augmenter,spliter=","):
    '''
    文字太长生成过程特别慢，因此只对部分数据进行生成
    就不做复杂的逻辑了，直接通过逗号split，
    然后随机挑一个进行生成，最后拼起来就是了
    :param text:
    :param augmenter:
    :return:
    '''
    MAX_LENGTH = 30
    splited_text = text.split(spliter)
    argument_index = randint(0,len(splited_text)-1)

    temp = splited_text[argument_index]
    if len(temp.split(" ")) <=MAX_LENGTH :
        generated = augmenter.augment(temp)
    else:
        temp = temp.find(" ",MAX_LENGTH)
        back = splited_text[argument_index][temp:]
        temp = splited_text[argument_index][:temp]
        temp = augmenter.augment(temp)
        generated = [ i+back for i in temp ]

    # front = splited_text[:argument_index]
    # back  = splited_text[argument_index+1:]

    result = list()
    for i in generated:
        t = splited_text.copy()
        t[argument_index] = i
        result.append(",".join(t))

    return result

def generateAAPD(datapath,transformations_per_example):
    augmenter = EmbeddingAugmenter(transformations_per_example=transformations_per_example)

    # [{'text':"...", 'catgy':['cat1','cat2',...] },]
    file_names = ['aapd_doc', 'aapd_tag']
    path = os.path.join(datapath, file_names[0])
    assert os.path.exists(path)
    with open(path) as f1:
        docs = f1.readlines()

    path = os.path.join(datapath, file_names[1])
    assert os.path.exists(path)
    with open(path) as f1:
        tags = f1.readlines()

    assert len(docs) == len(tags)

    generated_doc = open( os.path.join(datapath,'generated_aapd_doc_'+str(transformations_per_example)) ,'w')
    generated_tag = open( os.path.join(datapath,'generated_aapd_tag_'+str(transformations_per_example)) ,'w')
    for i, text, tag in zip(range(len(docs)), docs, tags):
        print("{}:{}  {}%        ".format(i, len(tags), 100 * float(i) / len(docs)), end="\r", flush=True)
        text_generated = generateLongSentence(text,augmenter)

        # 有时无法生成足量的样本，因此复制出适量的出来
        if len(text_generated) < transformations_per_example:
            while(len(text_generated)<transformations_per_example):
                text_generated.append(text_generated[0])
        assert len(text_generated) == transformations_per_example

        for j in text_generated:
            generated_doc.write(j)
            generated_tag.write(tag)
    print("\ndone!")
    generated_tag.close()
    generated_doc.close()

def generateStackFromQuesAndAns(ques,ans,augmenter,transformations_per_example):
    quesg = augmenter.augment(ques) # question generated

    if len(quesg)==transformations_per_example:
        # 仅靠ques就足以生成
        return [ [q,ans] for q in quesg ]
    else:
        # 还需加入ans来生成
        ansg = augmenter.augment(ans)# answer generated
        temp = [[q, a] for q in quesg for a in ansg]
        temp.extend([[q, ans] for q in quesg])
        temp.extend([[ques, a] for a in ansg])
        if len(temp) >= transformations_per_example:
            # 二者足以生成
            return temp[:transformations_per_example]
        else:
            # 不足以生成，因此强行使用原始数据扩充
            while len(temp)<transformations_per_example:
                temp.append([ques,ans])
            return temp

def generateStack(datapath,transformations_per_example,dump_result=False):
    # [{'text':"...", 'catgy':['cat1','cat2',...] },]
    import csv
    file_names = ['stackdata_utf8.csv']
    augmenter = EmbeddingAugmenter(transformations_per_example=transformations_per_example)
    path = os.path.join(datapath, file_names[0])
    assert os.path.exists(path)
    result = dict()

    data = list()
    with open(path, 'r', encoding='utf-8') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data.append(row)

    label2id = {".net": 0, "c#": 1, "database": 2, "mysql": 3, "sql-server": 4, "sql-server-2005": 5, "php": 6,
                "objective-c": 7, "iphone": 8, "web-services": 9, "windows": 10, "python": 11, "sql": 12, "css": 13,
                "html": 14, "asp.net": 15, "regex": 16, "c++": 17, "javascript": 18, "vb.net": 19, "visual-studio": 20,
                "asp.net-mvc": 21, "string": 22, "winforms": 23, "ajax": 24, "linq-to-sql": 25, "linq": 26,
                "performance": 27, "c": 28, "java": 29, "wpf": 30, "oop": 31, "wcf": 32, "multithreading": 33,
                "ruby": 34, "ruby-on-rails": 35, "tsql": 36, "jquery": 37, "xml": 38, "arrays": 39, "django": 40,
                "android": 41, "cocoa-touch": 42, }
    loaded_data = list()
    picked_data = list()
    g_data = list()
    g_loaded_data = list()
    for i in range(1, len(data)):
        print("{}:{}  {}%        ".format(i, len(data), 100 * float(i) / len(data)), end="\r", flush=True)

        tags = data[i][3]
        # 分隔符是###,为了处理c#,需要倒转再倒转
        tags = tags[::-1].split("###")[:-1]
        if len(tags) == 1:
            continue  # 跳过只有一个标签的数据
        tags = [onetag[::-1] for onetag in tags]
        used_tags = []
        for onetag in tags:
            if onetag in label2id.keys():
                used_tags.append(onetag)
        if len(used_tags) <= 1:
            continue  # 跳过筛选后只有一个标签的数据

        picked_data.append(data[i])

        text = data[i][1] + ' ' + data[i][2]
        text = clean_str(text)
        loaded_data.append({'text': text, 'catgy': [label2id[onetag] for onetag in used_tags]})

        g_temp = generateStackFromQuesAndAns(data[i][1],data[i][2],augmenter,transformations_per_example)
        for gt in g_temp:
            text = gt[0] + ' ' + gt[1]
            text = clean_str(text)
            g_data.append( [ data[i][0],gt[0],gt[1],data[i][3] ] )
            g_loaded_data.append({'text': text, 'catgy': [label2id[onetag] for onetag in used_tags]})

    result = {
        "full_csv_data" : data,
        "picked_csv_data" : picked_data,
        "generated_csv_data" : g_data,
        "loaded_data" : loaded_data,
        "generated_loaded_data" : g_loaded_data,
        "label2id":label2id,
        "transformations_per_example" :transformations_per_example,
    }

    if dump_result:
        with open( os.path.join(datapath,'generated_stack_'+str(transformations_per_example)+".pkl") ,'wb') as f:
            import pickle
            pickle.dump(result,f)
    return result


if __name__ == '__main__':
    # generateAAPD(r'D:\我的文件夹\学习\实验室\多标签主动学习项目\datasets\aapd',3)

    # augmenter = EmbeddingAugmenter(transformations_per_example=6)
    # s = 'What I cannot create, I do not understand.'
    # s = "the relation between pearson 's correlation coefficient and salton 's cosine measure is revealed based on the different possible values of the division of the l1 norm and the l2 norm of a vector these different values yield a sheaf of increasingly straight lines which form together a cloud of points , being the investigated relation the theoretical results are tested against the author co citation relations among 24 informetricians for whom two matrices can be constructed , based on co citations the asymmetric occurrence matrix and the symmetric co citation matrix both examples completely confirm the theoretical results the results enable us to specify an algorithm which provides a threshold value for the cosine above which none of the corresponding pearson correlations would be negative using this threshold value can be expected to optimize the visualization of the vector space"
    # s = "the relation between pearson 's correlation coefficient and salton 's cosine measure is revealed based on the different possible values of the division of the l1 norm and the l2 norm of a vector these different values yield a sheaf of increasingly straight lines which form together"
    # for i in generateLongSentence(s,augmenter):
    #     print(i)

    # generateStack(r'D:\我的文件夹\学习\实验室\多标签主动学习项目\datasets\stackOverflow',3,dump_result=True)

    pass
