import numpy as np
from nlp_vector import *

#######  simulation data  ########
# the number of data is small just for test
texts=[
    'design impress 651472 oil rub bronz singl handl kitchen faucet - kitchen sink faucet pull-out sprayer easi do-it-yourself instal household tool conform low lead requir ceram cartridg - featur pull-out sprayer stainless steel flexibl hose singl handl design - plastic line hybrid waterway ada compliant - flow rate 2 0 gpm 7 57 lpm',
    'best commerci stainless steel singl handl pull sprayer kitchen faucet pull kitchen faucet brush nickel one hole easi instal includ base plate cover sink hole pleas order separ fit 3 8” 1 2” water suppli hose quick connector make sure leak-proof durabl brush steel finish resist tarnish corros daili use premium ceram disc design faucet smooth stream spray water flow drip-fre singl handl easi control hot cold water pull sprayer 360 degre rotat offer room pot pan superior clearanc sink compat stainless steel composit granit natur stone sink',
    'kitchen faucet pull sprayer - wew a1008l stainless steel sink faucet singl handl high arc brush nickel faucet pull sprayer modern design 1st grade brush nickel kitchen faucet shine metal luster freeli match various kitchen sink wherev fit restaur apart home easi instal 2 water line hose togeth pull wand pre-instal pre-test faucet finish diy instal less 20 minut featur benefit 1 handl integr control water temperatur flow volum easili 360 degre rotat coil spout long spray hose wide rang clearanc sink flexibl extend pull wand univers joint suppli full rang move smooth easi clean nozzl advanc spray head filter system ensur impur block outsid power function 2 sprayer set toggl spray stream soft spray mode perfect rins aerat stream perfect fill water reliabl qualiti 5 year warranti metal connector part premium ceram valv high water pressur test ensur faucet leak free drip-fre',
    'lordear modern style stainless steel 2 water function set singl handl pull sprayer wet bar brush nickel kitchen faucet pull kitchen sink faucet qualiti solid brass construct ensur durabl depend beauti brush nickel finish build resist scratch corros tarnish high-arch spout design 360 degre rotat offer superior clearanc varieti sink activ extend spout retract hose make clean sink dish easier singl handl quick easi water temperatur control cupc nsf ab1953 certifi includ mount hardwar hot & cold water hose',
    'moen brantford one-handl high arc pulldown kitchen faucet chrome 7185c power clean spray technolog provid 50 percent spray power versus pulldown pullout faucet without power clean technolog chrome finish high reflect mirror-lik look work ani decor style equip reflex system smooth oper easi movement secur dock spray head featur duralock quick connect system easi instal connect type compress design instal thru 1 3 hole option escutcheon includ',
    'moen 7295srs brantford one-handl high-arc pullout kitchen faucet featur reflex spot resist stainless powerclean deliv high-pressur power spray minim amount residu water around sink duralock quick connect system easi instal spot resist stainless finish resist water spot fingerprint 1- hole mount option 3-hole escutcheon includ back moen limit lifetim warranti',
    "hoti modern spring 360 degre swivel pull singl handl singl lever stainless steel pull prep sprayer kitchen sink faucet brush nickel overal height 20 87' spout height 6 89' spout reach 8 27' brush nickel & ceram disc valv resist scratch corros high-arch spout 360 degre rotat offer room pot pan superior clearanc sink pull-down spray wand featur long hose provid 20-inch reach hot cold water control kitchen faucet best pressur spray 10 year warranti 90-day money back guaranteed-when order today you'r protect 100% money back guarante free hassl lifetim replac warranti",
    'univers laundri tub faucet maya pull spray spout non-metal ab plastic chrome finish doubl handl univers laundri faucet great option look get nice practic design faucet great price faucet instal easili new exist laundri tub extra-long 24-inch weight pull head add great conveni flexibl wash soak fill pail easili toggl steadi stream spray function touch spout doubl handl clear mark hot cold allow control exact temperatur water connect garden hose aerat easili remov allow connect spout ani standard garden hose- perfect wash dirt befor enter home water plant bath pet outsid univers 4-inch center set make great simpl solut upgrad replac old faucet start right new laundri tub made high qualiti ab plastic high-end maintenance-fre washerless control cartridg ensur long life watertight seal maya qualiti ab plastic slop sink faucet tast design focus practic univers fit 4-inch center set'
]
keys=[
    'B00Q368V6K',
    'B016XXQVOG',
    'B074Y7W5VD',
    'B077XCCN86',
    'B00499SD4S',
    'B01BP5S0RS',
    'B071G5GFH6',
    'B07843MSXB'
]
data=dict()
for i in range(len(keys)):
    data[keys[i]]=texts[i]

#########     train and save model   ##########
# tfidf model
tfidf=TFIDF_Model()
tfidf.build_model(texts,keys)
#similar commodities(from large to small)
tfidf.top_N([data['B074Y7W5VD']],8)     #这里取8没什么意思，原本样本数为8，只要比所有样本数大的数就行
tfidf.save(tfidf,'tfidf')

# word2vec model
word2vec = Word2Vec_Model()
word2vec.build_model(texts,keys,size=50,window=5)
word2vec.averaged_doc()
word2vec.top_N([data['B074Y7W5VD']],8)
word2vec.save(word2vec,'word2vec_size50_window5')

# glove model
glove = Glove_Model()
co_matrix = glove.build_model(texts, keys,window=3)
glove.train(co_matrix, vector_size=15, iteration=800)
glove.averaged_doc()
glove.top_N([data['B074Y7W5VD']], 8)

glove.save(glove,'glove_size15_window3')

# lda model
lda=LDA_Model()
lda.build_model(texts,keys)
lda.train(topic_num=13,iteration=800)     ##381
lda.top_N('B074Y7W5VD',8)
lda.save(lda,'lda_size13')

# svd model
svd=SVD_Model()      #7:390
svd.build_model(tfidf.matrix,keys,K=7)
svd.top_N('B074Y7W5VD',8)
svd.save(svd,'svd_size7')


#########  load model  #########
tfidf=TFIDF_Model().load('tfidf')
word2vec = Word2Vec_Model().load('word2vec_size50_window5')
glove = Glove_Model().load('glove_size15_window3')
lda=LDA_Model().load('lda_size13')
svd=SVD_Model().load('svd_size7')



