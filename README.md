### dataset
```
# download
http://u.cs.biu.ac.il/~koppel/BlogCorpus.htm

# filenames format
5114.male.25.indUnk.Scorpio.xml

# XML content format
---
<Blog>
<date>28,February,2001</date>
<post>
      Slashdot raises lots of  urlLink interesting thoughts about banner ads .  The idea is to let users control the ad delivery, and even to allow users to comment on ads.
</post>
...
</Blog>
---

# uncompress in {PROJECT_HOME}/data/blogs
```

### Data Analysis
```
# installation
cd $PROJECT_HOME
PYTHONPATH=`pwd`
export PYTHONPATH

sudo apt install libxml2-dev
sudo pip install python-igraph
sudo pip install blist
sudo pip install nltk
sudo pip install scipy

python
>> import nltk
>> nltk.download('stopwords')
>> nltk.download('wordnet')

# NB: it is assumed that the blogs dataset was extracted in data/blogs
mkdir data/stats
mkdir data/graphs

### Step 1: Prepare the data

# tokenize blogs & generate token stats + graphs
python step1/prepare_data.py $BLOGS_FOLDER $STATS_FOLDER $GRAPHS_FOLDER
# tokenize blogs & generate token stats + graphs -> split blogs in half
python step1/prepare_data-split.py $BLOGS_FOLDER $STATS_FOLDER $GRAPHS_FOLDER

# tokenize blogs & generate token stats + matrices
python step1/prepare_data2.py $BLOGS_FOLDER $STATS_FOLDER $MATS_FOLDER
# tokenize blogs & generate token stats + graphs -> split blogs in half
python step1/prepare_data-split2.py $BLOGS_FOLDER $STATS_FOLDER $MATS_FOLDER

### Step 2: Prepare samples

python step2/sample.py $SRC_FOLDER $TGT_FOLDER
python step2/sample-split.py $SRC_FOLDER $TGT_FOLDER

### Step 3: Generate domain graphs

### Step 4: Compute DF and BOW vectors

```
