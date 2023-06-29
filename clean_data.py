def clean_data(text,type = 'E'):
    ''' This function support clean the text
    --------------------------------
    Args:
    text (_str) : text want to clean
    type :
        - From Englist to Vi - type = 'E'
        - From Vi to English - type = 'V'
    ---------------------------------------
    Output:
    text_clean (_str_)
    '''
    import re
    import regex
    import pandas as pd
    from nltk.corpus import stopwords, wordnet
    from nltk.stem import WordNetLemmatizer
    from nltk import pos_tag, word_tokenize

    # lower text:
    text_clean = text.lower()

    # change abb to full words, using support 'contractions.csv'
    contraction_df = pd.read_csv('contractions.csv', sep = ',', header = 0)

    for index, row in contraction_df.iterrows():
        flag = re.search(row.Contraction,text_clean)
        if flag:
            text_clean = text_clean.replace(row.Contraction,row.Meaning)

    # Remove URLs,links:
    text_clean = text_clean.replace(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*', '')

    # special clean with ?
    # Method 1: remove ?,!,.....
    punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    for char in punctuation:
        text_clean= text_clean.replace(char, ' ')
    # Method 2: Add "Không" in question sentences,
    # remove blank at begin and end line:
    text_clean = text_clean.strip()
    #Stemming and Lemitiazed
    lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(treebank_tag):
        """
        return WORDNET POS compliance to WORDENT lemmatization (a,n,r,v)
        """
        if treebank_tag.startswith('J'):
            return wordnet.ADJ
        elif treebank_tag.startswith('V'):
            return wordnet.VERB
        elif treebank_tag.startswith('N'):
            return wordnet.NOUN
        elif treebank_tag.startswith('R'):
            return wordnet.ADV
        else:
            # As default pos in lemmatization is Noun
            return wordnet.NOUN

    def lemmatize_text(text):
        lemmatized = []
        post_tag_list = pos_tag(word_tokenize(text))
        for word, post_tag_val in post_tag_list:
            lemmatized.append(lemmatizer.lemmatize(word, get_wordnet_pos(post_tag_val)))
            text = ' '.join(x for x in lemmatized)
        return text
    text_clean = lemmatize_text(text_clean)

    return text_clean