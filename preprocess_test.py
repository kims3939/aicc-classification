from preprocess.aicc import ITSMPreprocessor

#order, pattern, sub string
regex_pattens = [
    (0, r'.+:', ' '),
    (1, r'<SPECIAL>[^\s]+|[^\s]+<SPECIAL>|<SPECIAL>',' '),
    (2, r'[^a-zA-Z가-힣0-9]',' '),
    (3, r'010\d{8}|010\s\d{4}\s\d{4}|\d{2,3}\s\d{2,4}\s\d{4}',' '),
    (4, r'SAID[\s]*\d{11}',' '),
    (5, r'SRM[\s]*\d{11}',' '),
    (6, r'\s\d+\s',' '),
    (100, r'\s+',' ')
]

def preprocess():
    texts, labels = ITSMPreprocessor.load_data('dataset/trainset.tsv', 0, 1, '\t')
    pp = ITSMPreprocessor(texts, labels)
    pp.add_regex(regex_pattens)
    pp.preprocess()
    pp.save('./processed.tsv')

    return pp.result

if __name__ == '__main__':
    result = preprocess()