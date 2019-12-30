# coding:utf-8
# ！！！！
#！！！！本注释皆为跟在语句后面的注释，程序语句的注释都在其后面或下面！！！！
#！！！！
import jieba #“结巴”中文分词（一个python中文分词组件）
import numpy as np #numpy(导入为np)是一个python数学运算组件，内置常见的数学运算
import collections #collections是Python内建的一个集合模块，提供了许多有用的集合类。（数学上的那个集合）
from sklearn import feature_extraction #sklearn内的特征提取工具包
#sklearn内的feature_extraction工具包下属的文本分析用的text下属的TfidfTransformer函数。
from sklearn.feature_extraction.text import CountVectorizer #CountVectorizer类会将文本中的词语转换为词频矩阵 
#例如矩阵中包含一个元素a[i][j]，它表示j词在i类文本下的词频。它通过fit_transform函数计算各个词语出现的次数，通过get_feature_names()可获取词袋中所有文本的关键字，通过toarray()可看到词频矩阵的结果。
from sklearn.feature_extraction.text import TfidfTransformer #TfidfTransformer用于统计vectorizer（CountVectorizer的输出）中每个词语的TF-IDF值。
#smooth_idf=True:
#idf = log((1+总文档数)/(1+包含ti的文档数)) +1
#smooth_idf=False:
#idf = log log((总文档数)/(包含ti的文档数)) +1
#tf-idf =tf*idf
#之后需进行 the Euclidean (L2) norm，得到最终的tf-idf权重。详细请参考文档http://scikit-learn.org/stable/modules/feature_extraction.html#common-vectorizer-usage


def split_sentence(text, punctuation_list='!?。！？'):
    """
    将文本段安装标点符号列表里的符号切分成句子，将所有句子保存在列表里。
    """
    sentence_set = []
    inx_position = 0         #索引标点符号的位置
    char_position = 0        #移动字符指针位置
    ##！！记住！！python的数组标号是从0开始的，而不是从1！！
    for char in text:   #遍历text字符串内的字符
        char_position += 1 #移动字符指针位置+1，可以表示从1开始算起的话当前的char是text的第几个字符；也可以表示当前char的下一个位置的字符。
        if char in punctuation_list: #如果当前char_position位置的char（text中的一个字符变量）等于punctuation_list中的标点符号的话。
            next_char = list(text[inx_position:char_position+1]).pop() #pop() 函数用于移除列表（list）中的一个元素（默认最后一个元素），并且返回该元素的值。
            ##数组冒号取值举例开始：
            '''
            a = 'python!!'
            print (a[1:4]) #输出：yth （可以看到1:4实际上输出的是位置1、2、3(从0开始数)的字符）
            inx_position = 0
            char_position = 6
            next_char = list(a[inx_position:char_position+1]).pop()
            print (a[inx_position:char_position+1]) #输出：python!
            print (next_char) #输出：! (按照程序中的话，char_position = 6的时候char='n'，但a[char_position]='!'，可a[0:char_position]='python'，取不到'!')
            '''
            ##数组冒号取值举例结束。

            ##list.pop()举例开始： 
            '''          
            list1 = ['Google', 'Runoob', 'Taobao']
            list_pop=list1.pop(1)
            print ("删除的项为 :", list_pop) #输出：删除的元素为 : Runoob
            print ("列表现在为 : ", list1) #输出：列表现在为 :  ['Google', 'Taobao']
            '''
            ##list.pop()举例结束。

            #对text中inx_position（当前循环在text中的起始位置）到char_position+1（+1是为了能够装入探测到的标点符号的下一个字符变量）的位置，这之间的所有文本变量都装入一个list中。
            #然后再取该list的最后一个字符变量，实质就是取char_position位置的char(现在是'!?。！？'中的某个标点符号)的下一个字符，然后存成next_char变量。
            ##我不知道为什么原作者取当前位置的下一个字符要写得这么麻烦，完全没必要。

            if next_char not in punctuation_list: #看探测到的标点符号的下一个字符是不是也是一个在punctuation_list中的标点符号。
                #目的是为了防止有些句子使用“！？”、“！！！”这样的多重标点作为结尾
                sentence_set.append(text[inx_position:char_position]) #在sentence_set这个数组中添加从inx_position到char_position的文本字符，实际上就是储存好分割出来的句子。
                inx_position = char_position #将inx_position设置为char_position（既当前分割好的句子的结尾+1，也就是标点符号的下一个字符），用作下一次循环分割某个句子的起始位置。
    if inx_position < len(text): #如果inx_position这个位置标号小于全部文本text所储存的字符个数。len()返回text的数组长度，也就是储存了多少变量。
        #因为前面的for char in text循环中是取char_position+1+1（既char之后的两个字符）看是否为“非断句标点符号”来判定句子切分的，实际上最后会超过text的数组范围。
        #所以text中的最后一句由别的部分的程序来处理切分。
        sentence_set.append(text[inx_position:]) #将inx_position（最后一句的开头）直到text末尾的所有字符都加入到sentence_set数组中成为一个元素。

    sentence_with_index = {i:sent for i,sent in enumerate(sentence_set)} #dict(zip(sentence_set, range(len(sentences))))
    #enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标，一般用在 for 循环当中。
    
    #普通的for循环 
    '''
    >>>i = 0
    >>> seq = ['one', 'two', 'three']
    >>> for element in seq:
    ...     print i, seq[i]
    ...     i +=1
    ... 
    0 one
    1 two
    2 three
    '''
    #for 循环使用 enumerate
    '''
    >>>seq = ['one', 'two', 'three']
    >>> for i, element in enumerate(seq):
    ...     print i, element
    ... 
    0 one
    1 two
    2 three
    '''
    #enumerate在本程序中的使用举例
    """
    {i:sent for i,sent in enumerate(sentence_set)}这段程序可将sentence_set = ['aaaaaaa!!!!', 'bbbbbbb!?', 'iiiiii。']
    变为sentence_with_index = {0: 'aaaaaaa!!!!', 1: 'bbbbbbb!?', 2: 'iiiiii。'}
    “for i,sent in enumerate(sentence_set)”这段句子如上循环一样枚举除了从0开始的各个sentence_set中的元素的标号。
    然后再用“i:sent”这样的方式组装入大括号{}中成为字典（dict）的一个个元素。“xxx:xxxxx”是字典的组合方式，比如“GOOGLE:EVIL”,在字典dict中就是GOOGLE这个索引的值就是EVIL。
    """

    return sentence_set,sentence_with_index
    #函数返回sentence_set和sentence_with_index

def get_tfidf_matrix(sentence_set,stop_word):
    corpus = []
    for sent in sentence_set: #遍历sentence_set数组内的字符串元素
        sent_cut = jieba.cut(sent) #用结巴分词提供的默认为精准模式的函数对sent字符串进行分词
        ##结巴分词代码示例开始：
        '''
        代码示例
        import jieba

        seg_list = jieba.cut("我来到北京清华大学", use_paddle=True)
        print("Paddle Mode: " + "/ ".join(seg_list))  # paddle模式

        seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
        print("Full Mode: " + "/ ".join(seg_list))  # 全模式

        seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
        print("Default Mode: " + "/ ".join(seg_list))  # 精确模式

        seg_list = jieba.cut("他来到了网易杭研大厦")  # 默认是精确模式
        print(", ".join(seg_list))

        seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
        print(", ".join(seg_list))

        输出:

        【全模式】: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学

        【精确模式】: 我/ 来到/ 北京/ 清华大学

        【新词识别】：他, 来到, 了, 网易, 杭研, 大厦    (此处，“杭研”并没有在词典中，但是也被Viterbi算法识别出来了)

        【搜索引擎模式】： 小明, 硕士, 毕业, 于, 中国, 科学, 学院, 科学院, 中国科学院, 计算, 计算所, 后, 在, 日本, 京都, 大学, 日本京都大学, 深造
        '''
        ##结巴分词代码示例结束。

        sent_list = [word for word in sent_cut if word not in stop_word] 
        #把如上文【精确模式】分好的词用“for word in sent_cut”遍历一遍，如果某个词word不在stop_word数组中的话，就加入到sent_list中（sent_list是一个[]围成的数组）。
        sent_str = ' '.join(sent_list) 
        #Python join() 方法用于将序列中的元素以指定的字符连接生成一个新的字符串。
        ##.join()代码示例开始：
        '''
        str = "-";
        seq = ("a", "b", "c"); # 字符串序列
        print str.join( seq ); #输出：a-b-c
        '''
        ##.join()代码示例结束。

        #将分好的并预处理好的词汇数组组合成字符串，比如数组sent_list = ['他','来到','了','网易']变成字符串sent_str = '他 来到 了 网易'。
        #这样做符合sklearn.feature_extraction.text提供的方法的输入格式。
        corpus.append(sent_str) #在数组corpus加入新的一项元素——刚刚处理好的字符串sent_str。

    vectorizer=CountVectorizer() #vectorizer是实例化CountVectorizer(),用起来方便，本身并不处理数据。
    transformer=TfidfTransformer() #transformer是实例化TfidfTransformer(),用起来方便，本身并不处理数据。
    tfidf=transformer.fit_transform(vectorizer.fit_transform(corpus))
    # 1.CountVectorizer
    # CountVectorizer()类会将文本中的词语转换为词频矩阵。
    # 例如矩阵中包含一个元素a[i][j]，它表示j词在i类文本下的词频。
    # 它通过fit_transform函数计算各个词语出现的次数，通过get_feature_names()可获取词袋中所有文本的关键字，
    # 通过toarray()可看到词频矩阵的结果。
    ##CountVectorizer()代码示例开始：
    '''
    from sklearn.feature_extraction.text import CountVectorizer

    corpus = [
        'This is the first document.',
        'This is the this second second document.',
        'And the third one.',
        'Is this the first document?'
    ]
    #语料

    vectorizer = CountVectorizer()
    print(vectorizer)
    #将文本中的词转换成词频矩阵,vectorizer是实例化CountVectorizer(),用起来方便，本身并不处理数据。
    X = vectorizer.fit_transform(corpus)
    print(type(X),X)
    #计算某个词出现的次数
    word = vectorizer.get_feature_names()
    print(word)
    #获取词袋中所有文本关键词
    print(X.toarray())
    #查看词频结果
    '''
    ##CountVectorizer()代码示例结束。

    #2.TfidfTransformer
    #TfidfTransformer用于统计vectorizer中每个词语的TF-IDF值。具体用法如下：
    ##TfidfTransformer()代码示例开始：
    '''
    from sklearn.feature_extraction.text import TfidfTransformer
    
    transformer = TfidfTransformer()
    print(transformer)
    #类调用
    tfidf = transformer.fit_transform(X)
    #将词频矩阵统计成TF-IDF值
    print(tfidf.toarray())
    #查看数据结构tfidf[i][j]表示i类文本中tf-idf权重
    '''
    #TfidfTransformer()代码示例结束。

    #具体公式见https://blog.csdn.net/wf592523813/article/details/81911155
    # word=vectorizer.get_feature_names()

    tfidf_matrix=tfidf.toarray() #将tfidf的格式转换为python中的array格式。
    return np.array(tfidf_matrix) #将已经转换为python中的array格式的tfidf在转换为numpy中的array格式。

def get_sentence_with_words_weight(tfidf_matrix):
    sentence_with_words_weight = {}
    for i in range(len(tfidf_matrix)):
        sentence_with_words_weight[i] = np.sum(tfidf_matrix[i])

    max_weight = max(sentence_with_words_weight.values()) #归一化
    min_weight = min(sentence_with_words_weight.values())
    for key in sentence_with_words_weight.keys():
        x = sentence_with_words_weight[key]
        sentence_with_words_weight[key] = (x-min_weight)/(max_weight-min_weight)

    return sentence_with_words_weight

def get_sentence_with_position_weight(sentence_set):
    sentence_with_position_weight = {}
    total_sent = len(sentence_set)
    for i in range(total_sent):
        sentence_with_position_weight[i] = (total_sent - i) / total_sent
    return sentence_with_position_weight

def similarity(sent1,sent2):
    """
    计算余弦相似度
    """
    return np.sum(sent1 * sent2) / 1e-6+(np.sqrt(np.sum(sent1 * sent1)) *\
                                    np.sqrt(np.sum(sent2 * sent2)))

def get_similarity_weight(tfidf_matrix):
    sentence_score = collections.defaultdict(lambda :0.)
    for i in range(len(tfidf_matrix)):
        score_i = 0.
        for j in range(len(tfidf_matrix)):
            score_i += similarity(tfidf_matrix[i],tfidf_matrix[j])
        sentence_score[i] = score_i

    max_score = max(sentence_score.values()) #归一化
    min_score = min(sentence_score.values())
    for key in sentence_score.keys():
        x = sentence_score[key]
        sentence_score[key] = (x-min_score)/(max_score-min_score)

    return sentence_score

def ranking_base_on_weigth(sentence_with_words_weight,
                            sentence_with_position_weight,
                            sentence_score, feature_weight = [1,1,1]):
    sentence_weight = collections.defaultdict(lambda :0.)
    for sent in sentence_score.keys():
        sentence_weight[sent] = feature_weight[0]*sentence_with_words_weight[sent] +\
                                feature_weight[1]*sentence_with_position_weight[sent] +\
                                feature_weight[2]*sentence_score[sent]

    sort_sent_weight = sorted(sentence_weight.items(),key=lambda d: d[1], reverse=True)
    return sort_sent_weight

def get_summarization(sentence_with_index,sort_sent_weight,topK_ratio =0.3):
    topK = int(len(sort_sent_weight)*topK_ratio)
    summarization_sent = sorted([sent[0] for sent in sort_sent_weight[:topK]])
    
    summarization = []
    for i in summarization_sent:
        summarization.append(sentence_with_index[i])

    summary = ''.join(summarization)
    return summary


if __name__ == '__main__':
    test_text = 'training17.txt'
    with open(test_text,'r') as f:
        text = f.read()
    stop_word = []
    with open('stopWordList.txt','r') as f:
        for line in f.readlines():
            stop_word.append(line.strip())

    sentence_set,sentence_with_index = split_sentence(text, punctuation_list='!?。！？')
    tfidf_matrix = get_tfidf_matrix(sentence_set,stop_word)
    sentence_with_words_weight = get_sentence_with_words_weight(tfidf_matrix)
    sentence_with_position_weight = get_sentence_with_position_weight(sentence_set)
    sentence_score = get_similarity_weight(tfidf_matrix)
    sort_sent_weight = ranking_base_on_weigth(sentence_with_words_weight,
                                                sentence_with_position_weight,
                                                sentence_score, feature_weight = [1,1,1])
    summarization = get_summarization(sentence_with_index,sort_sent_weight,topK_ratio =0.3)
    print('summarization:\n',summarization)