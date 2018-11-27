[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

# competitive-recsys
A collection of resources for Recommender Systems (RecSys)

# Recommendation Algorithms

- Basic of Recommender Systems
  - [Wikipedia](https://en.wikipedia.org/wiki/Recommender_system)
- Nearest Neighbor Search
  - [Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
  - [sklearn.neighbors](http://scikit-learn.org/stable/modules/neighbors.html)
  - [Benchmarks of approximate nearest neighbor libraries](https://github.com/erikbern/ann-benchmarks)
- Classic Matrix Facotirzation
  - [Matrix Factorization: A Simple Tutorial and Implementation in Python](http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/)
  - [Matrix Factorization Techiques for Recommendaion Systems](https://datajobs.com/data-science-repo/Recommender-Systems-[Netflix].pdf)
- Singular Value Decomposition (SVD)
  - [Wikipedia](https://en.wikipedia.org/wiki/Singular-value_decomposition)
- SVD++
  - [Factorization Meets the Neighborhood: a Multifaceted Collaborative Filtering Model](http://www.cs.rochester.edu/twiki/pub/Main/HarpSeminar/Factorization_Meets_the_Neighborhood-_a_Multifaceted_Collaborative_Filtering_Model.pdf)
- Content-based CF / Context-aware CF
  - there are so many ...
- Advanced Matrix Factorization
  - [Probabilistic Matrix Factorization](https://papers.nips.cc/paper/3208-probabilistic-matrix-factorization.pdf)
  - [Fast Matrix Factorization for Online Recommendation with Implicit Feedback](https://dl.acm.org/citation.cfm?id=2911489)
  - [Collaborative Filtering for Implicit Feedback Datasets](http://ieeexplore.ieee.org/document/4781121/)
  - [Factorization Meets the Item Embedding: Regularizing Matrix Factorization with Item Co-occurrence](https://dl.acm.org/citation.cfm?id=2959182)
- Factorization Machine
  - [Factorization Machines](https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf)
  - [Field-aware Factorization Machines for CTR Prediction](https://dl.acm.org/citation.cfm?id=2959134)
- Sparse LInear Method (SLIM)
  - [SLIM: Sparse Linear Methods for Top-N Recommender Systems](http://glaros.dtc.umn.edu/gkhome/node/774)
  - [Global and Local SLIM](http://glaros.dtc.umn.edu/gkhome/node/1192)
- Learning to Rank
  - [Wikipedia](https://en.wikipedia.org/wiki/Learning_to_rank)
  - [BPR: Bayesian personalized ranking from implicit feedback](https://dl.acm.org/citation.cfm?id=1795167)
  - [WSABIE: Scaling Up To Large Vocabulary Image Annotation](http://www.thespermwhale.com/jaseweston/papers/wsabie-ijcai.pdf)
  - [Top-1 Feedback](http://proceedings.mlr.press/v38/chaudhuri15.pdf)
  - [k-order statistic loss](http://www.ee.columbia.edu/~ronw/pubs/recsys2013-kaos.pdf)
  - [VBPR: Visual Bayesian Personalized Ranking from Implicit Feedback](https://dl.acm.org/citation.cfm?id=3015834)
- Cold-start
  - [Deep content-based music recommendation](https://papers.nips.cc/paper/5004-deep-content-based-music-recommendation)
  - [DropoutNet: Addressing Cold Start in Recommender Systems](https://papers.nips.cc/paper/7081-dropoutnet-addressing-cold-start-in-recommender-systems)
- Network Embedding
  - [awesome-network-embedding](https://github.com/chihming/awesome-network-embedding)
  - [Item2vec](https://arxiv.org/abs/1603.04259)
  - [entity2rec](https://dl.acm.org/citation.cfm?id=3109889)
- Translation Embedding
  - [Translation-based Recommendation](https://dl.acm.org/citation.cfm?id=3109882)
- Deep Learning
  - [Deep Neural Networks for YouTube Recommendations](https://ai.google/research/pubs/pub45530)
  - [Deep Learning based Recommender System: A Survey and New Perspectives](https://arxiv.org/abs/1707.07435)
  - [Neural Collaborative Filtering](https://dl.acm.org/citation.cfm?id=3052569)
  - [Collaborative Deep Learning for Recommender Systems](http://www.wanghao.in/CDL.htm)
  - [Collaborative Denoising Auto-Encoders for Top-N Recommender Systems](https://dl.acm.org/citation.cfm?id=2835837)
  - [Collaborative recurrent autoencoder: recommend while learning to fill in the blanks](https://dl.acm.org/citation.cfm?id=3157143)
  - [TensorFlow Wide & Deep Learning](https://www.tensorflow.org/tutorials/wide_and_deep)
  - [Deep Neural Networks for YouTube Recommendations](https://research.google.com/pubs/pub45530.html)
  - [Collaborative Memory Network for Recommendation Systems](https://arxiv.org/abs/1804.10862)

# Public Available Datasets
- [GroupLens](https://grouplens.org/)
  - [MovieLens](https://grouplens.org/datasets/movielens/)
  - [HetRec2011](https://grouplens.org/datasets/hetrec-2011/)
  - [WikiLens](https://grouplens.org/datasets/wikilens/)
  - [Book-Crossing](https://grouplens.org/datasets/book-crossing/)
  - [Jester](https://grouplens.org/datasets/jester/)
  - [EachMovie](https://grouplens.org/datasets/eachmovie/)
- [Amazon Product Data](http://jmcauley.ucsd.edu/data/amazon/)
  - Books, Electronics, Movies, etc.
- [SNAP Datasets](https://snap.stanford.edu/data/index.html)
- [#nowplaying Dataset](http://dbis-nowplaying.uibk.ac.at/)
- [Last.fm Datasets](http://www.dtic.upf.edu/~ocelma/MusicRecommendationDataset/index.html)
- [Million Song Dataset](https://labrosa.ee.columbia.edu/millionsong/)
- [Frappe](http://baltrunas.info/research-menu/frappe)
- [Yahoo! Webscope Program](https://webscope.sandbox.yahoo.com/)
  - music ratings, movie ratings, etc.
- [Yelp Dataset Challenge](https://www.yelp.com/dataset/challenge)
- [MovieTweetings](https://github.com/sidooms/MovieTweetings)
- [Foursquare](https://archive.org/details/201309_foursquare_dataset_umn)
- [Epinions](http://jmcauley.ucsd.edu/data/epinions)
- [Google Local](http://jmcauley.ucsd.edu/data/googlelocal/)
  - location, phone number, time, rating, addres, GPS, etc.
- [CiteUlike-t](http://www.wanghao.in/CDL.htm)
- [LibimSeTi](http://www.occamslab.com/petricek/data/)
- [Scholarly Paper Recommendation Datasets](http://www.comp.nus.edu.sg/~sugiyama/SchPaperRecData.html)
- [Netflix Prize Data Set](http://academictorrents.com/details/9b13183dc4d60676b773c9e2cd6de5e5542cee9a)
- [FilmTrust,CiaoDVD](https://www.librec.net/datasets.html)
- [Chicago Entree](http://archive.ics.uci.edu/ml/datasets/Entree+Chicago+Recommendation+Data)
- [Douban](http://socialcomputing.asu.edu/datasets/Douban)
- [BibSonomy](https://www.kde.cs.uni-kassel.de/bibsonomy/dumps)
- [Delicious](http://www.dai-labor.de/en/competence_centers/irml/datasets/)
- [Foursquare](https://archive.org/details/201309_foursquare_dataset_umn)
- [MACLab LJ Datasets](http://mac.citi.sinica.edu.tw/LJ#.Ww_hbFOFNE5)
- Kaggle::Datasets
  - [Steam Video Games](https://www.kaggle.com/tamber/steam-video-games/data)
  - [Anime Recommendations Database](https://www.kaggle.com/CooperUnion/anime-recommendations-database)

# Open Sources
- [libFM](http://www.libfm.org/) - Factorization Machine Library
- [fastFM](https://github.com/ibayer/fastFM) - A Library for Factorization Machines
- [LIBFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/) - A Library for Field-aware Factorization Machines
- [lightfm](https://github.com/lyst/lightfm) - A Python implementation of LightFM, a hybrid recommendation algorithm
- [LIBMF](https://www.csie.ntu.edu.tw/~cjlin/libmf/) - A Matrix-factorization Library for Recommender Systems
- [LibRec](https://www.librec.net/index.html) - A Leading Java Library for Recommender Systems
- [LensKit](http://lenskit.org/) - Open-Source Tools for Recommender Systems
- [Surprise](https://github.com/NicolasHug/Surprise) - A Python scikit building and analyzing recommender systems
- [MyMediaLite Recommender System Library](http://www.mymedialite.net/index.html)
- [QMF](https://github.com/quora/qmf) - A matrix factorization library
- [proNet-core](https://github.com/cnclabs/proNet-core) - A general-purpose network embedding framework: pair-wise representations optimization Network
- [Rival](http://rival.recommenders.net/) - An open source Java toolkit for recommender system evaluation
- [TensorRec](https://github.com/jfkirk/tensorrec) - A TensorFlow recommendation algorithm and framework in Python
- [OpenRec](http://openrec.ai/index.html) - An open-source and modular library for neural network-inspired recommendation algorithms
- [spotlight](https://github.com/maciejkula/spotlight) - Deep recommender models using PyTorch.
- [Recoder](https://github.com/amoussawi/recoder) - Large scale training of factorization models for Collaborative Filtering with PyTorch.

# Common Evaluation Metric
- Precision and Recall
  - [Wikipedia](https://en.wikipedia.org/wiki/Precision_and_recall)
- Mean Average Precision (MAP)
  - [Wikipedia](https://en.wikipedia.org/wiki/Information_retrieval#Mean_average_precision)
- ROC Curve / Area under the curve
  - [Wikipedia](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)
- Normalized Discounted Cumulative Gain (NDCG)
  - [Wikipedia](https://en.wikipedia.org/wiki/Discounted_cumulative_gain)
- Mean Absolute Error (MAE)
  - [Wikipedia](https://en.wikipedia.org/wiki/Mean_absolute_error)
- Root Mean Square Error (RMSE) 
  - [Wikipedia](https://en.wikipedia.org/wiki/Root-mean-square_deviation)
- Novelty and Diversity
  - [Novelty and Diversity in Top-N Recommendation -- Analysis and Evaluation](https://dl.acm.org/citation.cfm?id=1944341)
- Beyond accuracy
  - [Beyond accuracy: evaluating recommender systems by coverage and serendipity](https://dl.acm.org/citation.cfm?id=1864761)
  
  
# Related Github links
- [List of Recommender Systems](https://github.com/grahamjenson/list_of_recommender_systems) - A List of Recommender Systems and Resources
- [Recommendation and Ratings Public Data Sets For Machine Learning](https://gist.github.com/entaroadun/1653794)
- [RecommenderSystem-Paper](https://github.com/daicoolb/RecommenderSystem-Paper)

# Textbooks
- [Programming Collective Intelligence](http://shop.oreilly.com/product/9780596529321.do)

# Online Courses
- [Recommender Systems Specialization](https://zh-tw.coursera.org/specializations/recommender-systems), University of Minnesota
- [Introduction to Recommender Systems: Non-Personalized and Content-Based](https://zh-tw.coursera.org/learn/recommender-systems-introduction), University of Minnesota

# RecSys-related Competitions
- [Kaggle](https://www.kaggle.com/) - product recommendations, hotel recommendations, job recommendations, etc.
- ACM RecSys Challenge
- [WSDM Cup 2018](https://wsdm-cup-2018.kkbox.events/)
- [Million Song Dataset Challenge](https://www.kaggle.com/c/msdchallenge)
- [Netflix Prize](https://www.netflixprize.com/)

# Tutorials
- RecSys tutorials
  - [2014](https://recsys.acm.org/recsys14/tutorials/)
  - [2015](https://recsys.acm.org/recsys15/tutorials/)
  - [2016](https://recsys.acm.org/recsys16/tutorials/)
  - [2017](https://recsys.acm.org/recsys17/tutorials/)
- [Kdd 2014 Tutorial - the recommender problem revisited](https://www.slideshare.net/xamat/kdd-2014-tutorial-the-recommender-problem-revisited)

# Articles
- [Matrix Factorization: A Simple Tutorial and Implementation in Python](http://www.quuxlabs.com/blog/2010/09/matrix-factorization-a-simple-tutorial-and-implementation-in-python/)

# Conferences
- [RecSys – ACM Recommender Systems](https://recsys.acm.org/)
