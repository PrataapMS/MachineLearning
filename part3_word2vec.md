

```python
import pandas as pd

# Read data from files 
train = pd.read_csv( "labeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )
test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_train = pd.read_csv( "unlabeledTrainData.tsv", header=0, 
 delimiter="\t", quoting=3 )

# Verify the number of reviews that were read (100,000 in total)
print "Read %d labeled train reviews, %d labeled test reviews, " \
 "and %d unlabeled reviews\n" % (train["review"].size,  
 test["review"].size, unlabeled_train["review"].size )

 # Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
from nltk.corpus import stopwords

def review_to_wordlist( review, remove_stopwords=False ):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    review_text = BeautifulSoup(review).get_text()
    #  
    # 2. Remove non-letters
    review_text = re.sub("[^a-zA-Z]"," ", review_text)
    #
    # 3. Convert words to lower case and split them
    words = review_text.lower().split()
    #
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    #
    # 5. Return a list of words
    return(words)



# Download the punkt tokenizer for sentence splitting
import nltk.data
#nltk.download()   

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

# Define a function to split a review into parsed sentences
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
	try:
	    # Function to split a review into parsed sentences. Returns a 
	    # list of sentences, where each sentence is a list of words
	    #
	    # 1. Use the NLTK tokenizer to split the paragraph into sentences
	    raw_sentences = tokenizer.tokenize(review.strip())
	    #
	    # 2. Loop over each sentence
	    sentences = []
	    for raw_sentence in raw_sentences:
	        # If a sentence is empty, skip it
	        if len(raw_sentence) > 0:
	            # Otherwise, call review_to_wordlist to get a list of words
	            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))

	except Exception as ex:
		print type(ex)
		print ex.args
		pass
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
	return sentences

## ABOVE FUNCTION
def review_to_sentences( review, tokenizer, remove_stopwords=False ):
    sentences = []
    try:
	    # Function to split a review into parsed sentences. Returns a 
	    # list of sentences, where each sentence is a list of words
	    #
	    # 1. Use the NLTK tokenizer to split the paragraph into sentences
	    raw_sentences = tokenizer.tokenize(review.strip())
	    #
	    # 2. Loop over each sentence
	    for raw_sentence in raw_sentences:
	        # If a sentence is empty, skip it
	        if len(raw_sentence) > 0:
	            # Otherwise, call review_to_wordlist to get a list of words
	            sentences.append( review_to_wordlist( raw_sentence, remove_stopwords ))
    except Exception as ex:
	    print type(ex)
	    print ex.args
	    pass
    return sentences

# Import the built-in logging module and configure it so that Word2Vec 
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 300    # Word vector dimensionality                      
min_word_count = 40   # Minimum word count                        
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size                                                                                    
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
from gensim.models import word2vec
model = word2vec.Word2Vec.load("300features_40minwords_10context")
```

    Read 25000 labeled train reviews, 25000 labeled test reviews, and 50000 unlabeled reviews
    



```python
print model.doesnt_match("rain snow sleet sun".split())
print model.doesnt_match("staple hammer saw drill".split())
print model.doesnt_match("math shopping reading science".split())
print model.doesnt_match("eight six seven five three owe nine".split())
print model.doesnt_match("breakfast cereal dinner lunch".split())
print model.doesnt_match("biking driving cycling riding walking dancing".split())
```

    sun
    saw
    shopping
    owe
    lunch
    dancing



```python
type(model.syn0)
```




    numpy.ndarray




```python
model.syn0.shape
```




    (16247, 300)




```python
model["flower"]
```




    array([-0.02295511,  0.03186645, -0.08400724,  0.02856324,  0.00979366,
            0.04673794,  0.11115041,  0.03848862,  0.03610057, -0.04237205,
           -0.09737229, -0.09992433, -0.00257822,  0.13239665, -0.0676937 ,
            0.02332498,  0.01215594, -0.0298765 , -0.0548828 , -0.08837233,
           -0.05714894,  0.06021371,  0.1289847 ,  0.02510408,  0.00319404,
            0.04587587, -0.03255558, -0.04677752, -0.03734799, -0.00779689,
            0.04645124,  0.08804289, -0.01321578,  0.03308902, -0.07320101,
            0.03532802,  0.02940688, -0.01477755,  0.06764556, -0.00242986,
           -0.01992187,  0.11925041,  0.01505583, -0.02383193,  0.00617092,
           -0.04873193,  0.03525778, -0.04408037, -0.01452965,  0.07136582,
           -0.06298844,  0.07097518,  0.10206738, -0.04385149, -0.02900619,
           -0.00400248, -0.02881881, -0.02160546, -0.10183349, -0.04196122,
           -0.0768263 ,  0.05571777,  0.0060843 , -0.01062915, -0.03865522,
            0.063022  , -0.10971905,  0.01292582,  0.097382  , -0.00075135,
           -0.00290697,  0.02215058, -0.00603702, -0.0250108 ,  0.03798503,
           -0.0515049 ,  0.00443404,  0.07674594, -0.02404969,  0.09450032,
            0.00779501,  0.06720358, -0.02432154, -0.16698146, -0.01224587,
            0.02399149, -0.00462938, -0.08676095,  0.04112042,  0.04750252,
           -0.06793648, -0.00521071, -0.05421578,  0.0611189 , -0.00873598,
           -0.01858722,  0.05215464, -0.02253303,  0.02882958, -0.00320129,
           -0.02478986, -0.05623104,  0.06649885, -0.01871945, -0.03915666,
           -0.06689394,  0.05955065,  0.14144531, -0.00237025, -0.03754371,
            0.04868598, -0.08979871,  0.00201832,  0.06486417,  0.07843421,
            0.01092209,  0.13818939, -0.06951139,  0.0556573 ,  0.05716322,
            0.04964669, -0.05385071,  0.00474948,  0.04066258,  0.03369094,
            0.02048906, -0.04820657, -0.01657434,  0.02924886, -0.02539947,
           -0.03697103, -0.01717832, -0.05555373, -0.05911654, -0.05602553,
            0.03433963, -0.05762201, -0.03903905, -0.05416476,  0.00719647,
            0.01788121, -0.00377419,  0.04563699,  0.0289128 ,  0.19828814,
            0.0580624 , -0.09180269,  0.07430846, -0.07532975,  0.06204761,
           -0.00745316,  0.01231052,  0.08215642,  0.04445159, -0.05054677,
            0.08526359, -0.05652836, -0.00486047,  0.03620445,  0.08205581,
           -0.01755775,  0.03457783, -0.03990182, -0.12598158, -0.0211366 ,
           -0.00923564,  0.01259588,  0.0383773 ,  0.02398784,  0.04928722,
            0.09884644, -0.04480481, -0.05272052, -0.01818438, -0.01475299,
            0.03719483,  0.00076409, -0.01524548,  0.07898369,  0.00456993,
            0.00959061, -0.07308953, -0.05395036, -0.06060633, -0.05456353,
            0.0108327 , -0.03749138, -0.04797361, -0.06879406, -0.06068976,
            0.00074038, -0.01379107,  0.06064452,  0.04363624,  0.02842869,
            0.00978722, -0.01638677,  0.14292085, -0.01143618, -0.05801012,
           -0.00593382, -0.00440054, -0.04627579,  0.09824836,  0.00989895,
           -0.05380271, -0.04510307,  0.05925419,  0.02622174,  0.04960703,
           -0.01380874, -0.00330657, -0.00553716,  0.06171931,  0.04835028,
            0.07149525,  0.06131864,  0.0105771 ,  0.0490838 ,  0.05804091,
            0.02749198, -0.1028004 ,  0.08093532, -0.00943019,  0.0276743 ,
           -0.05395467,  0.00169891, -0.07108279, -0.02372103,  0.01471176,
           -0.03062405, -0.0666639 ,  0.0913202 , -0.01807422,  0.07762246,
           -0.00597063, -0.01860925, -0.10329263,  0.07391011, -0.04677894,
            0.06598217,  0.09350192,  0.03425812,  0.00091685, -0.10082096,
            0.0635393 ,  0.05332541,  0.00557902,  0.02149981, -0.02778528,
            0.08533867, -0.09763901,  0.04804742,  0.06511141, -0.05628118,
            0.03877501, -0.00923337,  0.04063497,  0.03611169,  0.060907  ,
            0.05400853, -0.00835211, -0.08930742,  0.12850343,  0.02256242,
           -0.00663392,  0.12773843, -0.01835981,  0.01196245,  0.02921877,
           -0.06884851,  0.02051582, -0.03444867,  0.03150091, -0.02655812,
            0.02963359, -0.0481996 , -0.01831459, -0.02611181,  0.02876211,
           -0.12033007,  0.06531567,  0.15557879,  0.03161819, -0.06132128,
            0.03567   , -0.03321714, -0.07454143,  0.00331398, -0.05157876,
            0.05525547,  0.04000757, -0.03307886,  0.00230933, -0.10912848,
            0.02994397, -0.08314835,  0.11964075,  0.13633379,  0.02658559], dtype=float32)




```python
import numpy as np  # Make sure that numpy is imported
```


```python
def makeFeatureVec(words, model, num_features):
    # Function to average all of the word vectors in a given
    # paragraph
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((num_features,),dtype="float32")
    #
    nwords = 0.
    # 
    # Index2word is a list that contains the names of the words in 
    # the model's vocabulary. Convert it to a set, for speed 
    index2word_set = set(model.index2word)
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        if word in index2word_set: 
            nwords = nwords + 1.
            featureVec = np.add(featureVec,model[word])
    # 
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec,nwords)
    return featureVec

```


```python
model.index2word
```




    [u'the',
     u'and',
     u'a',
     u'of',
     u'to',
     u'is',
     u'it',
     u'in',
     u'i',
     u'this',
     u'that',
     u's',
     u'was',
     u'as',
     u'for',
     u'with',
     u'movie',
     u'but',
     u'film',
     u'you',
     u't',
     u'on',
     u'not',
     u'he',
     u'are',
     u'his',
     u'have',
     u'be',
     u'one',
     u'all',
     u'they',
     u'at',
     u'by',
     u'who',
     u'an',
     u'from',
     u'so',
     u'like',
     u'there',
     u'or',
     u'her',
     u'just',
     u'about',
     u'out',
     u'has',
     u'if',
     u'what',
     u'some',
     u'good',
     u'can',
     u'more',
     u'when',
     u'very',
     u'she',
     u'up',
     u'no',
     u'time',
     u'even',
     u'would',
     u'my',
     u'which',
     u'story',
     u'their',
     u'only',
     u'really',
     u'see',
     u'had',
     u'were',
     u'well',
     u'me',
     u'we',
     u'than',
     u'much',
     u'bad',
     u'get',
     u'people',
     u'been',
     u'great',
     u'into',
     u'do',
     u'also',
     u'other',
     u'will',
     u'first',
     u'because',
     u'him',
     u'how',
     u'most',
     u'don',
     u'made',
     u'them',
     u'its',
     u'make',
     u'then',
     u'way',
     u'could',
     u'too',
     u'movies',
     u'after',
     u'any',
     u'characters',
     u'think',
     u'character',
     u'films',
     u'two',
     u'watch',
     u'being',
     u'many',
     u'plot',
     u'seen',
     u'never',
     u'where',
     u'love',
     u'life',
     u'little',
     u'acting',
     u'best',
     u'did',
     u'over',
     u'off',
     u'know',
     u'show',
     u'ever',
     u'does',
     u'man',
     u'better',
     u'your',
     u'here',
     u'end',
     u'scene',
     u'these',
     u'still',
     u'while',
     u'why',
     u'scenes',
     u'say',
     u'something',
     u'go',
     u've',
     u'm',
     u'should',
     u'such',
     u'back',
     u'through',
     u'real',
     u'those',
     u'now',
     u'watching',
     u're',
     u'doesn',
     u'thing',
     u'though',
     u'years',
     u'director',
     u'actors',
     u'funny',
     u'didn',
     u'old',
     u'another',
     u'new',
     u'nothing',
     u'makes',
     u'going',
     u'work',
     u'actually',
     u'before',
     u'look',
     u'find',
     u'same',
     u'lot',
     u'few',
     u'part',
     u'every',
     u'again',
     u'world',
     u'cast',
     u'us',
     u'things',
     u'quite',
     u'want',
     u'action',
     u'horror',
     u'down',
     u'pretty',
     u'young',
     u'around',
     u'seems',
     u'fact',
     u'take',
     u'however',
     u'enough',
     u'got',
     u'long',
     u'both',
     u'thought',
     u'give',
     u'big',
     u'own',
     u'between',
     u'comedy',
     u'series',
     u'd',
     u'must',
     u'right',
     u'may',
     u'original',
     u'without',
     u'role',
     u'interesting',
     u'times',
     u'saw',
     u'come',
     u'isn',
     u'always',
     u'guy',
     u'whole',
     u'gets',
     u'least',
     u'point',
     u'almost',
     u'bit',
     u'music',
     u'done',
     u'script',
     u'last',
     u'minutes',
     u'family',
     u'far',
     u'feel',
     u'since',
     u'll',
     u'making',
     u'might',
     u'girl',
     u'performance',
     u'anything',
     u'yet',
     u'away',
     u'am',
     u'probably',
     u'tv',
     u'kind',
     u'hard',
     u'woman',
     u'fun',
     u'day',
     u'rather',
     u'worst',
     u'sure',
     u'anyone',
     u'found',
     u'played',
     u'each',
     u'especially',
     u'having',
     u'trying',
     u'our',
     u'screen',
     u'looking',
     u'believe',
     u'different',
     u'although',
     u'place',
     u'course',
     u'sense',
     u'set',
     u'goes',
     u'ending',
     u'comes',
     u'maybe',
     u'worth',
     u'shows',
     u'three',
     u'american',
     u'money',
     u'dvd',
     u'put',
     u'looks',
     u'actor',
     u'everything',
     u'once',
     u'someone',
     u'let',
     u'effects',
     u'wasn',
     u'plays',
     u'john',
     u'main',
     u'year',
     u'together',
     u'book',
     u'everyone',
     u'reason',
     u'true',
     u'during',
     u'instead',
     u'said',
     u'job',
     u'takes',
     u'special',
     u'watched',
     u'high',
     u'play',
     u'night',
     u'war',
     u'seem',
     u'audience',
     u'later',
     u'wife',
     u'himself',
     u'star',
     u'black',
     u'half',
     u'seeing',
     u'left',
     u'idea',
     u'excellent',
     u'death',
     u'beautiful',
     u'second',
     u'shot',
     u'house',
     u'simply',
     u'used',
     u'men',
     u'else',
     u'dead',
     u'father',
     u'mind',
     u'version',
     u'less',
     u'completely',
     u'fan',
     u'nice',
     u'hollywood',
     u'budget',
     u'poor',
     u'women',
     u'help',
     u'line',
     u'home',
     u'performances',
     u'boring',
     u'sex',
     u'along',
     u'try',
     u'top',
     u'either',
     u'read',
     u'short',
     u'wrong',
     u'low',
     u'use',
     u'until',
     u'kids',
     u'friends',
     u'camera',
     u'given',
     u'couple',
     u'next',
     u'need',
     u'enjoy',
     u'start',
     u'classic',
     u'full',
     u'production',
     u'truly',
     u'rest',
     u'stupid',
     u'awful',
     u'video',
     u'perhaps',
     u'school',
     u'moments',
     u'tell',
     u'getting',
     u'mean',
     u'mother',
     u'keep',
     u'came',
     u'won',
     u'face',
     u'small',
     u'recommend',
     u'understand',
     u'terrible',
     u'others',
     u'style',
     u'playing',
     u'name',
     u'itself',
     u'wonderful',
     u'boy',
     u'remember',
     u'stars',
     u'doing',
     u'person',
     u'definitely',
     u'gives',
     u'often',
     u'lost',
     u'dialogue',
     u'live',
     u'written',
     u'early',
     u'human',
     u'lines',
     u'perfect',
     u'case',
     u'entertaining',
     u'head',
     u'went',
     u'hope',
     u'yes',
     u'couldn',
     u'episode',
     u'become',
     u'children',
     u'title',
     u'liked',
     u'friend',
     u'certainly',
     u'supposed',
     u'based',
     u'picture',
     u'problem',
     u'piece',
     u'absolutely',
     u'fans',
     u'finally',
     u'against',
     u'oh',
     u'sort',
     u'overall',
     u'several',
     u'drama',
     u'felt',
     u'laugh',
     u'entire',
     u'worse',
     u'evil',
     u'under',
     u'son',
     u'called',
     u'cinema',
     u'waste',
     u'direction',
     u'lives',
     u'killer',
     u'guys',
     u'lead',
     u'humor',
     u'care',
     u'beginning',
     u'white',
     u'game',
     u'dark',
     u'seemed',
     u'despite',
     u'loved',
     u'mr',
     u'wanted',
     u'final',
     u'b',
     u'totally',
     u'history',
     u'throughout',
     u'becomes',
     u'unfortunately',
     u'already',
     u'turn',
     u'town',
     u'genre',
     u'fine',
     u'able',
     u'guess',
     u'days',
     u'heart',
     u'flick',
     u'run',
     u'city',
     u'today',
     u'act',
     u'side',
     u'quality',
     u'child',
     u'horrible',
     u'wants',
     u'tries',
     u'sound',
     u'kill',
     u'hand',
     u'close',
     u'past',
     u'starts',
     u'example',
     u'viewer',
     u'writing',
     u'enjoyed',
     u'amazing',
     u'turns',
     u'themselves',
     u'car',
     u'etc',
     u'expect',
     u'parts',
     u'behind',
     u'michael',
     u'killed',
     u'directed',
     u'matter',
     u'works',
     u'favorite',
     u'fight',
     u'soon',
     u'daughter',
     u'decent',
     u'kid',
     u'stuff',
     u'self',
     u'gave',
     u'type',
     u'thinking',
     u'girls',
     u'sometimes',
     u'blood',
     u'actress',
     u'violence',
     u'eyes',
     u'group',
     u'obviously',
     u'stop',
     u'brilliant',
     u'art',
     u'hour',
     u'late',
     u'myself',
     u'stories',
     u'known',
     u'writer',
     u'except',
     u'happened',
     u'god',
     u'hero',
     u'highly',
     u'says',
     u'heard',
     u'feeling',
     u'took',
     u'coming',
     u'roles',
     u'slow',
     u'police',
     u'extremely',
     u'experience',
     u'moment',
     u'leave',
     u'anyway',
     u'happens',
     u'voice',
     u'murder',
     u'husband',
     u'hell',
     u'involved',
     u'wouldn',
     u'attempt',
     u'interest',
     u'obvious',
     u'age',
     u'including',
     u'living',
     u'score',
     u'looked',
     u'taken',
     u'strong',
     u'told',
     u'david',
     u'save',
     u'ok',
     u'brother',
     u'wonder',
     u'hours',
     u'none',
     u'please',
     u'james',
     u'happen',
     u'cut',
     u'chance',
     u'cool',
     u'robert',
     u'career',
     u'particularly',
     u'gore',
     u'simple',
     u'ago',
     u'across',
     u'cannot',
     u'hit',
     u'complete',
     u'exactly',
     u'hilarious',
     u'lack',
     u'crap',
     u'annoying',
     u'o',
     u'power',
     u'possible',
     u'alone',
     u'sad',
     u'documentary',
     u'light',
     u'serious',
     u'level',
     u'usually',
     u'important',
     u'running',
     u'relationship',
     u'seriously',
     u'whose',
     u'female',
     u'reality',
     u'ends',
     u'scary',
     u'somewhat',
     u'number',
     u'happy',
     u'song',
     u'talent',
     u'order',
     u'taking',
     u'room',
     u'shown',
     u'ridiculous',
     u'finds',
     u'basically',
     u'jokes',
     u'middle',
     u'change',
     u'wish',
     u'call',
     u'strange',
     u'released',
     u'opening',
     u'usual',
     u'turned',
     u'yourself',
     u'body',
     u'english',
     u'mostly',
     u'country',
     u'opinion',
     u'silly',
     u'apparently',
     u'attention',
     u'miss',
     u'disappointed',
     u'cinematography',
     u'novel',
     u'view',
     u'started',
     u'sequel',
     u'saying',
     u'word',
     u'jack',
     u'huge',
     u'future',
     u'talking',
     u'four',
     u'thriller',
     u'single',
     u'major',
     u'words',
     u'shots',
     u'straight',
     u'rating',
     u'non',
     u'clearly',
     u'cheap',
     u'modern',
     u'beyond',
     u'knew',
     u'ones',
     u'fast',
     u'king',
     u'knows',
     u'sets',
     u'due',
     u'talk',
     u'british',
     u'problems',
     u'events',
     u'parents',
     u'comic',
     u'tells',
     u'bring',
     u'aren',
     u'earth',
     u'die',
     u'entertainment',
     u'local',
     u'easily',
     u'add',
     u'television',
     u'george',
     u'class',
     u'upon',
     u'sequence',
     u'similar',
     u'supporting',
     u'musical',
     u'storyline',
     u'giving',
     u'hate',
     u'above',
     u'haven',
     u'york',
     u'falls',
     u'ten',
     u'clear',
     u'within',
     u'romantic',
     u'easy',
     u'appears',
     u'review',
     u'five',
     u'near',
     u'mystery',
     u'predictable',
     u'ways',
     u'french',
     u'dialog',
     u'team',
     u'lots',
     u'bunch',
     u'typical',
     u'enjoyable',
     u'stand',
     u'begins',
     u'crime',
     u'episodes',
     u'named',
     u'eye',
     u'richard',
     u'general',
     u'working',
     u'message',
     u'avoid',
     u'mention',
     u'filmed',
     u'elements',
     u'america',
     u'songs',
     u'theme',
     u'red',
     u'certain',
     u'sorry',
     u'points',
     u'moving',
     u'gay',
     u'surprised',
     u'release',
     u'dull',
     u'whether',
     u'stay',
     u'tale',
     u'viewers',
     u'minute',
     u'needs',
     u'th',
     u'among',
     u'fall',
     u'space',
     u'using',
     u'tom',
     u'effort',
     u'gone',
     u'lee',
     u'e',
     u'leads',
     u'comments',
     u'theater',
     u'feels',
     u'kept',
     u'tried',
     u'herself',
     u'nearly',
     u'paul',
     u'peter',
     u'period',
     u'buy',
     u'third',
     u'truth',
     u'de',
     u'brought',
     u'soundtrack',
     u'means',
     u'suspense',
     u'clich',
     u'showing',
     u'doubt',
     u'viewing',
     u'lady',
     u'killing',
     u'sister',
     u'follow',
     u'feature',
     u'sequences',
     u'rent',
     u'fantastic',
     u'form',
     u'somehow',
     u'material',
     u'dog',
     u'editing',
     u'realistic',
     u'okay',
     u'monster',
     u'average',
     u'check',
     u'cop',
     u'rock',
     u'famous',
     u'whatever',
     u'figure',
     u'imagine',
     u'move',
     u'forget',
     u'reviews',
     u'animation',
     u'oscar',
     u'lame',
     u'weak',
     u'fi',
     u'believable',
     u'surprise',
     u'premise',
     u'sci',
     u'poorly',
     u'deal',
     u'free',
     u'actual',
     u'possibly',
     u'indeed',
     u'expected',
     u'hear',
     u'learn',
     u'eventually',
     u'forced',
     u'dr',
     u'sit',
     u'society',
     u'note',
     u'stage',
     u'greatest',
     u'deep',
     u'otherwise',
     u'sexual',
     u'wait',
     u'atmosphere',
     u'difficult',
     u'romance',
     u'reading',
     u'leaves',
     u'open',
     u'decided',
     u'joe',
     u'plus',
     u'begin',
     u'particular',
     u'question',
     u'screenplay',
     u'situation',
     u'earlier',
     u'became',
     u'western',
     u'nor',
     u'hot',
     u'box',
     u'subject',
     u'male',
     u'acted',
     u'crew',
     u'interested',
     u'brothers',
     u'gun',
     u'personal',
     u'previous',
     u'towards',
     u'footage',
     u'credits',
     u'street',
     u'imdb',
     u'meet',
     u'season',
     u'cheesy',
     u'memorable',
     u'shame',
     u'worked',
     u'business',
     u'laughs',
     u'battle',
     u'powerful',
     u'writers',
     u'effect',
     u'mess',
     u'features',
     u'air',
     u'unless',
     u'dramatic',
     u'needed',
     u'whom',
     u'era',
     u'setting',
     u'older',
     u'nature',
     u'keeps',
     u'result',
     u'baby',
     u'boys',
     u'quickly',
     u'perfectly',
     u'hands',
     u'total',
     u'comment',
     u'badly',
     u'emotional',
     u'crazy',
     u'forward',
     u'realize',
     u'directing',
     u'twist',
     u'japanese',
     u'mark',
     u'appear',
     u'bill',
     u'background',
     u'pay',
     u'present',
     u'telling',
     u'write',
     u'girlfriend',
     u'superb',
     u'rich',
     u'weird',
     u'development',
     u'various',
     u'island',
     u'unique',
     u'meets',
     u'william',
     u'dance',
     u'plenty',
     u'fighting',
     u'disney',
     u'break',
     u'front',
     u'apart',
     u'secret',
     u'brings',
     u'sounds',
     u'directors',
     u'c',
     u'doctor',
     u'incredibly',
     u'fairly',
     u'villain',
     u'masterpiece',
     u'outside',
     u'missing',
     u'dream',
     u'leading',
     u'married',
     u'beauty',
     u'list',
     u'party',
     u'return',
     u'manages',
     u'fantasy',
     u'zombie',
     u'remake',
     u'create',
     u'admit',
     u'ideas',
     u'rate',
     u'reasons',
     u'ask',
     u'joke',
     u'meant',
     u'political',
     u'creepy',
     u'inside',
     u'potential',
     u'copy',
     u'cute',
     u'success',
     u'unlike',
     u'dumb',
     u'fails',
     u'hold',
     u'portrayed',
     ...]




```python
def getAvgFeatureVecs(reviews, model, num_features):
    # Given a set of reviews (each one a list of words), calculate 
    # the average feature vector for each one and return a 2D numpy array 
    # 
    # Initialize a counter
    counter = 0.
    # 
    # Preallocate a 2D numpy array, for speed
    reviewFeatureVecs = np.zeros((len(reviews),num_features),dtype="float32")
    # 
    # Loop through the reviews
    for review in reviews:
       #
       # Print a status message every 1000th review
       if counter%1000. == 0.:
           print "Review %d of %d" % (counter, len(reviews))
       # 
       # Call the function (defined above) that makes average feature vectors
       reviewFeatureVecs[counter] = makeFeatureVec(review, model, \
           num_features)
       #
       # Increment the counter
       counter = counter + 1.
    return reviewFeatureVecs
```


```python
clean_train_reviews = []
for review in train["review"]:
    clean_train_reviews.append( review_to_wordlist( review, remove_stopwords=True ))
```


```python
trainDataVecs = getAvgFeatureVecs( clean_train_reviews, model, num_features )
```

    Review 0 of 25000
    Review 1000 of 25000
    Review 2000 of 25000
    Review 3000 of 25000
    Review 4000 of 25000
    Review 5000 of 25000
    Review 6000 of 25000
    Review 7000 of 25000
    Review 8000 of 25000
    Review 9000 of 25000
    Review 10000 of 25000
    Review 11000 of 25000
    Review 12000 of 25000
    Review 13000 of 25000
    Review 14000 of 25000
    Review 15000 of 25000
    Review 16000 of 25000
    Review 17000 of 25000
    Review 18000 of 25000
    Review 19000 of 25000
    Review 20000 of 25000
    Review 21000 of 25000
    Review 22000 of 25000
    Review 23000 of 25000
    Review 24000 of 25000



```python
print "Creating average feature vecs for test reviews"
clean_test_reviews = []
for review in test["review"]:
    clean_test_reviews.append( review_to_wordlist( review, \
        remove_stopwords=True ))

testDataVecs = getAvgFeatureVecs( clean_test_reviews, model, num_features )
```

    Creating average feature vecs for test reviews
    Review 0 of 25000
    Review 1000 of 25000
    Review 2000 of 25000
    Review 3000 of 25000
    Review 4000 of 25000
    Review 5000 of 25000
    Review 6000 of 25000
    Review 7000 of 25000
    Review 8000 of 25000
    Review 9000 of 25000
    Review 10000 of 25000
    Review 11000 of 25000
    Review 12000 of 25000
    Review 13000 of 25000
    Review 14000 of 25000
    Review 15000 of 25000
    Review 16000 of 25000
    Review 17000 of 25000
    Review 18000 of 25000
    Review 19000 of 25000
    Review 20000 of 25000
    Review 21000 of 25000
    Review 22000 of 25000
    Review 23000 of 25000
    Review 24000 of 25000



```python
# Fit a random forest to the training data, using 100 trees
from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier( n_estimators = 100 )

print "Fitting a random forest to labeled training data..."
forest = forest.fit( trainDataVecs, train["sentiment"] )

# Test & extract results 
result = forest.predict( testDataVecs )

# Write the test results 
output = pd.DataFrame( data={"id":test["id"], "sentiment":result} )
output.to_csv( "Word2Vec_AverageVectors.csv", index=False, quoting=3 )
```

    Fitting a random forest to labeled training data...



```python
from sklearn import metrics
score = metrics.accuracy_score(testDataVecs, pred)
print("accuracy:   %0.3f" % score)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-5-911e878d28c7> in <module>()
          1 from sklearn import metrics
    ----> 2 score = metrics.accuracy_score(testDataVecs, pred)
          3 print("accuracy:   %0.3f" % score)


    NameError: name 'testDataVecs' is not defined



```python
from sklearn.cluster import KMeans
import time

start = time.time() # Start time

# Set "k" (num_clusters) to be 1/5th of the vocabulary size, or an
# average of 5 words per cluster
word_vectors = model.syn0
num_clusters = word_vectors.shape[0] / 5

# Initalize a k-means object and use it to extract centroids
kmeans_clustering = KMeans( n_clusters = num_clusters )
idx = kmeans_clustering.fit_predict( word_vectors )

# Get the end time and print how long the process took
end = time.time()
elapsed = end - start
print "Time taken for K Means clustering: ", elapsed, "seconds."
```

    Time taken for K Means clustering:  2419.55305696 seconds.



```python
print type(test)
print type(train)
```

    <class 'pandas.core.frame.DataFrame'>
    <class 'pandas.core.frame.DataFrame'>



```python

```
