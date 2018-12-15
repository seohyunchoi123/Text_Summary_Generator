import pickle

with open("part3.txt", "rb") as f:
    df = pickle.load(f)
f.close()


stopwords = ["은", "는", "이", "가", "을", "를", "에", "할", "과", "의", "도", "였", "았",
             "됐", "및", "수", "때", "것", "곧", "던", "또", "약", "된", "로", "씩", "듯", "것", "씨", "었",
             "될", "와", "잘", "진", "중", "데", "게", "점", "시", "간", "었", "엔", "만", "께", "들", "등", "다",

             "다만", "보다", "직해", "지만", "라는", "위해", "라고", "으로", "다고", "했다", "였다",
             "하고", "해야", "다며", "에서", "등에", "하며", "에는", "된다", "인데", "따라", "이에",
             "정도", "라며", "치고", "이나", "에도", "데다", "려면", "것을", "것이", "등을", "특히",
             "뿐만", "하면", "했던", "따른", "것도", "하는", "처럼", "리고", "이지", "면서", "대한",
             "있어", "하게", "됐을", "까지", "로는", "되지", "하자", "있기", "만한", "됨에", "이란",
             "나설", "등이", "았고", "부터", "해서", "에는", "되어", "이고", "에서", "이다", "에선", "니다",
             "하며", "으며", "었다", "하려", "등을", "등은", "이자", "거나", "하지", "으론", "하기", "으니",
             "에게", "로서", "로써", "돼야", "에겐", "됐다", "한테", "이면", "들께", "되자", "조차", "하라",
             "이든", "했고", "보단", "되는", "들께", "이런", "당한", "드릴", "라든", "했다", "겠다", "다면",

             "한다면", "으려면", "보인다", "보다는", "하다고", "에서만", "따르면", "이라는", "으로도", "합니다",
             "당분간", "있다고", "스러운", "있을지", "것이란", "것으로", "중이다", "한다고", "아울러", "헸더니",
             "시키는", "하면서", "이라고", "이라며", "이른바", "오히려", "된다고", "있음을", "등으로", "이예요",
             "하고자", "에서도", "했다고", "에게서", "까지만", "되어야", "됐으며", "이었다", "이지만", "습니까",
             "었지만", "했으며", "이겠냐", "으로서", "으로써", "됐거나", "됐으면", "했으나", "로부터",
             "에서는", "했으니", "에게는", "에게만", "으로나", "되거나", "하거나", "했어야", "이라도",
             "한다는", "되었다", "이었던", "들에게", "이라면", "라거나", "라든지", "이면서", "이었다",
             "했으면", "에서만", "되었다", "되었고", "됐다고", "하므로", "했는지", "해주길", "부터라",
             "하면서", "해줘야", "으로는", "보다는", "해봐야", "되더라", "하더라", "으로선", "하다는",

             "하겠다고", "만으로도", "이라면서", "보고서는", "것이라고", "시키거나", "보인다고", "으로부터", "했습니다",
             "되었지만", "이겠냐만", "한다더니", "이야말로", "으로서는", "으로써는", "이어서가", "되었거나", "하십니까",
             "이었다면", "되었으며", "이었으니", "이었으나", "이었지만", "했으면서", "했다면서", "하는구나",
             "부터라서", "하면서는", "해봐야지", "해보아야", "되더라도", "하더라도", "하겠으나", "되었으나",

             "하느냐인데", "으로부터는"
             ]

def remove_stopwords(stopwords, summarizations, col_name):
    total = []
    for summarization in summarizations:
        summarization = summarization.split()
        summarization = [t for t in summarization if len(t) > 0]
        for i in range(len(summarization)):
            for stopword in stopwords:
                if summarization[i][-len(stopword):] == stopword:
                    summarization[i] = summarization[i].replace(stopword, "")

        summarization_sen = " ".join(summarization)
        total.append(summarization_sen)

    total = pd.DataFrame(total, columns=[col_name])

    return total