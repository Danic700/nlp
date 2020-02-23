

from nltk.tokenize import RegexpTokenizer


class DataExtractor:
    def __init__(self, logger):
        self.logger = logger

    def _extract_line(self, line):
        import re
        pattern = r'<subject>(.*)<\/subject>(<content>(.*)<\/content>)?<maincat>(.*)<\/maincat>'

        m = re.match(pattern, line)
        if m:
            subject = m.group(1)
            content = subject if m.group(3) is None else m.group(3)
            category = m.group(4)
            self.logger.debug('subject = ' + subject + ', content = ' + content + ', category = ' + category)
            return (subject, content, category)
        else:
            raise ValueError("Failed to extract line: " + str(line))

    def extract_data(self, data, text_columns):
        extracted_data = []

        data = self._remove_text_unwanted_words(data)

        for line in data.split('\n'):
            if len(line) < 2:
                continue
            
            subject, content, category = self._extract_line(line)
            extracted_data.append([self._remove_sentence_unwanted_words(subject), self._remove_sentence_unwanted_words(content), category])

        return extracted_data

    def _remove_text_unwanted_words(self, text):
        import re
        unwanted_words = ['&#xd;', '&lt;', 'br', '&gt;']
        p = re.compile('|'.join(map(re.escape, unwanted_words))) # escape to handle metachars
        return p.sub('', text)

    def _remove_sentence_unwanted_words(self, sentence):
        from nltk.corpus import stopwords

        sentence = sentence.lower()
        pattern = r'''(?x)          # set flag to allow verbose regexps
        \d+(?:\.\d+)?  # numbers, e.g. 12.40, 82
        |(?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
        | \w+(?:-\w+)*        # words with optional internal hyphens
        | \$?\d+(?:\.\d+)?%?  # currency and percentages, e.g. $12.40, 82%
        | \.\.\.              # ellipsis
        | [][,;"'?():_`-]    # these are separate tokens; includes ], [
        '''
        tokenizer = RegexpTokenizer(pattern)
        tokens = tokenizer.tokenize(sentence)
        stop_letters = set(list(r"!\"#$%&'()*+,-/:;<=>?@[\]^_`{|}~]"))
        unwanted_letters = set(['?', ',', "'"])
        letters_to_remove = stop_letters.union(unwanted_letters)
        wanted_letters = [letter for letter in tokens if not letter in letters_to_remove]
        wanted_words = " ".join(wanted_letters)
        stop_words = set(stopwords.words("english"))
        unwanted_words = set(['get']).union(stop_words)
        wanted_words_list = " ".join([word for word in wanted_words.split(" ") if not word in unwanted_words])
        filtered_words = [word for word in wanted_words_list.split(" ") if len(word) > 2 and not word.replace('.','',1).isdigit()]
        return filtered_words
