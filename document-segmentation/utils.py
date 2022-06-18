import re
import pickle
from pathlib2 import Path
from nltk.tokenize import RegexpTokenizer
from torch.utils.data import Dataset, DataLoader
#logger = utils.setup_logger(__name__, 'train.log')

missing_stop_words = set(['of', 'a', 'and', 'to'])

def get_seperator_format(levels=None):
    section_delimiter = "========"
    segment_seperator = "========"
    level_format = '\d' if levels == None else '['+ str(levels[0]) + '-' + str(levels[1]) + ']'
    seperator_format = segment_seperator + ',' + level_format + ",.*?\."
    return seperator_format

def get_words_tokenizer():
    words_tokenizer = RegexpTokenizer(r'\w+')
    return words_tokenizer

def extract_sentence_words(sentence, remove_special_tokens=False):

    if remove_special_tokens:
        for token in ["***LIST***", "***formula***", "***codice***" ]:
            sentence = sentence.replace(token, "")
    tokenizer = get_words_tokenizer()
    sentence_words = tokenizer.tokenize(sentence)

    return sentence_words

def get_files(path):
    all_objects = Path(path).glob('**/*')
    files = [str(p) for p in all_objects if p.is_file()]
    return files

def cache_wiki_exclude_filenames(wiki, wiki_folder):
    """Specifiy the document examples with less than 2 positive"""
    exclude_list = []
    cache_path = wiki_folder / 'doc_exclude_paths.cache'

    for i in range(len(wiki)):
        if sum(wiki[i]['targets']) < 2 or len(wiki[i]['targets']) < 4:
            exclude_list.append(i)
        if i % 100000 == 0:
            print(f"{i} documents viewed...")
    print(f"{len(exclude_list)} documents obtained less than 2 positive")

    with open(cache_path, 'wb') as f:
        pickle.dump(exclude_list, f)

def cache_wiki_filenames(wiki_folder):
    files = Path(wiki_folder).glob('*/*/*/*')
    cache_path = wiki_folder / 'doc_paths.cache'

    with cache_path.open('w+') as f:
        for file in files:
            print(file)
            f.write(str(file) + u'\n')

def get_sections(path, high_granularity=True):
    # clean_section = (lambda x: x.strip())

    # (1) Retrieve the raw text data
    file = open(str(path), "r")
    raw_content = file.read()
    file.close()
    clean_txt = raw_content.strip()

    # (2) Preprocess with the defined granularity 
    if high_granularity:
        sections_to_keep_pattern = get_seperator_format() # all the seperation range
    else:
        # if low granularity required we should flatten segments within segemnt level 2
        pattern_to_ommit = get_seperator_format((3, 999))
        clean_txt = re.sub(pattern_to_ommit, "", clean_txt)

        sections_to_keep_pattern = get_seperator_format((1, 2)) # only the section 1,2 remained

        #delete empty lines after re.sub() some substitured section would be empty line
        sentences = [s for s in clean_txt.strip().split("\n") if len(s) > 0 and s != "\n"]
        clean_txt = '\n'.join(sentences).strip('\n')

    # (3) Get the final preprocessed sections
    all_sections = re.split(sections_to_keep_pattern, clean_txt)
    non_empty_sections = [s for s in all_sections if len(s) > 0]
    sections = [s.strip() for s in non_empty_sections]
    
    return sections


class WikipediaDataset(Dataset):
    def __init__(self, 
                 root, 
                 n_context_sent=1, 
                 remove_preface_segment=True, 
                 train=True, 
                 manifesto=False, 
                 folder=False, 
                 high_granularity=True,
                 truncate_less_than_n_positive=2):

        # if manifesto:
        #     self.textfiles = list(Path(root).glob('*'))
        if folder:
            self.textfiles = get_files(root)
        else:
            root_path = Path(root)
            cache_path = root_path / 'doc_paths.cache'
            if not cache_path.exists():
                print("Cache not exist, auto generating...")
                cache_wiki_filenames(root_path)
            self.textfiles = cache_path.read_text().splitlines()

        if len(self.textfiles) == 0:
            raise RuntimeError('Found 0 images in subfolders of: {}'.format(root))

        self.train = train
        self.root = root
        self.high_granularity = high_granularity
        self.n_context_sent = n_context_sent
        self.remove_preface_segment = remove_preface_segment


    def filtering(self):
        root_path = Path(self.root)
        cache_path = root_path / 'doc_exclude_paths.cache'
        if not cache_path.exists():
            print("Cache not exist, auto generating...")
            cache_wiki_exclude_filenames(self, root_path)

        # loading pickle file to list 
        with open(cache_path, 'rb') as fp:
            exclude_list = pickle.load(fp)

        temp = []
        for i, j in enumerate(self.textfiles):
            if i == exclude_list[0]:
                exclude_list.pop(0) # pop out the first
                if len(exclude_list) == 0:
                    break 
            else:
                temp.append(j)
            if i % 50000 == 0:
                print(f"{i} documents checked")
        
        print(f"[FILTER] {len(self.textfiles)} --> {len(temp)}")
        self.textfiles = temp

    def __getitem__(self, index):
        path = self.textfiles[index]
        return self.read_wiki_file(
                path=Path(path), 
                n_context_sent=self.n_context_sent, 
                remove_preface_segment=self.remove_preface_segment,
                high_granularity=self.high_granularity
        )

    def __len__(self):
        return len(self.textfiles)

    def read_wiki_file(self, path, n_context_sent=1, remove_preface_segment=True, high_granularity=True):
        all_sections = get_sections(path, high_granularity)
        required_sections = all_sections[1:] if remove_preface_segment and len(all_sections) > 0 else all_sections
        required_non_empty_sections = [section for section in required_sections if len(section) > 0 and section != "\n"]

        list_sentence = "***LIST***."
        final_sentences = []
        label = []
        # Over multiple section with the same document 
        for section_ind in range(len(required_non_empty_sections)):
            # Each section contains several sentences with the final sentence is section end.
            sentences_ = required_non_empty_sections[section_ind].split('\n')
            sentences = [x for x in sentences_ if x != list_sentence]
            if sentences:
                for sentence in sentences[:-1]:
                    final_sentences.append(sentence)
                    label.append(0)
                final_sentences.append(sentences[-1])
                label.append(1)

        left_context = []
        right_context = []
        targets = []
        if len(final_sentences) > n_context_sent:
            for sent_ind in range(n_context_sent, len(final_sentences)):
                prev_context = final_sentences[sent_ind-n_context_sent:sent_ind]
                after_context = final_sentences[sent_ind: min(len(final_sentences),sent_ind+n_context_sent)]

                prev_context = " ".join(prev_context)
                after_context = " ".join(after_context)
                left_context.append(prev_context)
                right_context.append(after_context)
                # data.append([prev_context, after_context])
                targets.append(label[sent_ind-1])

        return {"left_context": left_context, 
                "right_context": right_context, 
                "targets": targets, 
                "path": path}

class Fin10KDataset(Dataset):
    def __init__(self, file_path):

        self.id = []
        self.left_text = ["[PAD]"]
        self.right_text = []
        sent_flag = ""

        # read files
        with open(file_path, 'r') as f:
            for line in f:
                ids, sent_text = line.split('\t')
                company_id, yr, item_id, para_id, sent_id = ids.split('_')
                self.right_text.append(sent_text.strip())
                self.left_text.append(sent_text.strip())
                self.id.append(f'{company_id}_{yr}_{item_id}_{para_id}_{sent_id}')

        # remove the last line of right context
        self.left_text.pop(-1)

    def __getitem__(self, index):
        return {"left_context": self.left_text[index],
                "right_context": self.right_text[index],
                "sent_id": self.id[index]}

    def __len__(self):
        return len(self.id)
