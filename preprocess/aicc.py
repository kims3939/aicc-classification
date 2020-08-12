
import re
import pandas as pd

class ITSMPreprocessor():
    def __init__(self, texts, labels):
        self.texts  = texts
        self.labels = labels
        self.regx_patterns = []
        self.result = []
    
    def add_regex(self, patterns):
        self.regx_patterns.extend([(order, re.compile(pattern), sub) for order, pattern, sub in patterns])
        self.regx_patterns.extend(patterns)
    
    def preprocess(self):
        regx_patterns = sorted(self.regx_patterns, key=lambda x: x[0])
        for text, label in zip(self.texts, self.labels):
            tmp_text = text
            for _, pattern, sub in regx_patterns:
                tmp_text = re.sub(pattern, sub, tmp_text).strip()   
            self.result.append((label, tmp_text))
    
    def save(self, path):
        df = pd.DataFrame(self.result)
        df.to_csv(path, index=False, header=None, sep='\t')

    @staticmethod
    def load_data(path, label_idx, text_idx, sep):
        dataset = pd.read_csv(path, sep=sep)
        
        texts  = dataset.iloc[:,text_idx].values.tolist()
        labels = dataset.iloc[:,label_idx].values.tolist()

        return texts, labels 