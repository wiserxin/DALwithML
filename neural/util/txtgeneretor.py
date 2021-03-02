import os
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

        for i in text_generated:
            generated_doc.write(i)
            generated_tag.write(tag)
    print("\ndone!")
    generated_tag.close()
    generated_doc.close()

if __name__ == '__main__':
    generateAAPD(r'D:\我的文件夹\学习\实验室\多标签主动学习项目\datasets\aapd',3)

    # augmenter = EmbeddingAugmenter(transformations_per_example=6)
    # s = 'What I cannot create, I do not understand.'
    # s = "the relation between pearson 's correlation coefficient and salton 's cosine measure is revealed based on the different possible values of the division of the l1 norm and the l2 norm of a vector these different values yield a sheaf of increasingly straight lines which form together a cloud of points , being the investigated relation the theoretical results are tested against the author co citation relations among 24 informetricians for whom two matrices can be constructed , based on co citations the asymmetric occurrence matrix and the symmetric co citation matrix both examples completely confirm the theoretical results the results enable us to specify an algorithm which provides a threshold value for the cosine above which none of the corresponding pearson correlations would be negative using this threshold value can be expected to optimize the visualization of the vector space"
    # s = "the relation between pearson 's correlation coefficient and salton 's cosine measure is revealed based on the different possible values of the division of the l1 norm and the l2 norm of a vector these different values yield a sheaf of increasingly straight lines which form together"
    # for i in generateLongSentence(s,augmenter):
    #     print(i)