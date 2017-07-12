# coding: utf8
import argparse
import re
import pdb
import os
import json
import pandas
from lxml import etree
import csv
import time
import numpy as np
import sys
import random
import pickle
#Example 
#doc = etree.parse('content-sample.xml')

parser = argparse.ArgumentParser(description="Machine")
parser.add_argument('-m', help='foo help')
args = parser.parse_args()
pdb.set_trace()
if str(args.m) == "remote":
	defaultDataFolder = "/home/khalife/ai-lab/data/"
elif str(args.m) == "local":
	defaultDataFolder = "/Users/sammy/Documents/phd-2016/lab/ai-lab/data/"
else:
	print("Which machine?")
	sys.exit()	


years = ["2009", "2010"]
challenge2015 = "LDC2015E19_TAC_KBP_English_Entity_Linking_Comprehensive_Training_and_Evaluation_Data_2009-2013"
kbLinksFile =  defaultDataFolder + challenge2015 + "/data/" + years[0] + "/eval/tac_kbp_" + years[0] + "_english_entity_linking_evaluation_KB_links.tab"
knowledgeBaseFile = defaultDataFolder + "tac_kbp_ref_know_base/data/"
queriesFile = defaultDataFolder + challenge2015 + "/data/" + years[0] + "/eval/tac_kbp_" + years[0] + "_english_entity_linking_evaluation_queries.xml" 
sourceFile = defaultDataFolder + challenge2015 + "/data/" + years[0] + "/eval/source_documents/"


def typeCheck(x):
	return ( type(x) is not int )

def arrayToList(x):
	return x.tolist()

def loadMentions(dataFolder=defaultDataFolder):
	mentionToEntity = {}
	with open(kbLinksFile) as fileLinks:
		linksLines = fileLinks.readlines()
		kbLinks = [lL.strip().split('\t') for lL in linksLines]

	for kbLink in kbLinks:
        	mentionToEntity[kbLink[0]] = kbLink[1]

	####################### Queries #########################
	start = time.time()
	queriesName = {}
	mentionToText = {}
	lxmlFile = etree.parse(queriesFile)
	root = lxmlFile.getroot()
	queries = root.xpath("query")
	values = []
	for qu in queries:
		queriesName[qu.get("id")] = qu.find("name").text.lower().replace("\n", " ")
		mentionToText[qu.get("id")] = qu.find("docid").text.lower()
	end = time.time()
	print("Queries built, took " + str(end-start))



	####################### Mentions ########################
	Mentions = {}
	Mentions["id"] = []
	Mentions["name"] = []
	Mentions["text"] = []
	Mentions["entity"] = []
	Mentions1 = Mentions
	cuTime = 0
	maxParagraph = 3
	start = time.time()
	i = 0
	for m in mentionToText.keys():
		print(sourceFile + mentionToText[m] + ".xml")
		with open(sourceFile + mentionToText[m].upper() + ".xml") as mentionFile:
				try:
					lxmlMentions = etree.iterparse(sourceFile + mentionToText[m].upper() + ".xml", events=('end',), tag='HEADLINE')	
					mentionText = [lm for lm in lxmlMentions][0][1].text.lower().replace("\n", "")
					lxmlMentions = etree.iterparse(sourceFile + mentionToText[m].upper() + ".xml", events=('end',), tag='P')				
					mentionsText = [lm[1] for lm in lxmlMentions][:maxParagraph]
					for mt in mentionsText:
                                        	mentionText = mentionText + mt.text.lower().replace("\n", " ")
				
				except:
					lxmlMentions = etree.iterparse(sourceFile + mentionToText[m].upper() + ".xml", events=('end',), tag='P')
					mentionsText = [lm[1] for lm in lxmlMentions][:maxParagraph]
					mentionText = ""
					for mt in mentionsText:
						mentionText = mentionText + mt.text.lower().replace("\n", " ")
				
				mentionName = queriesName[m]
				mentionEntity = mentionToEntity[m]
				Mentions["id"].append(m)
				Mentions["name"].append(mentionName)
				Mentions["text"].append(mentionText)
				Mentions["entity"].append(mentionEntity)
		i += 1
	print("Mentions base built, took : " + str(time.time()-start))
	with open(defaultDataFolder + challenge2015 + "/json/mentions.json", "w") as mentionsJsonFile:
		json.dump(Mentions, mentionsJsonFile)	
	
	with open(defaultDataFolder + challenge2015 + "/json/queriesNames.json", "w") as queriesJsonFile:
		json.dump(queriesName, queriesJsonFile)
	
	mentionsDataFrame = pandas.DataFrame(data=Mentions)

	

	return queriesName, mentionsDataFrame
	####################################################################################



def generateCorruptedEntities(mentionsDataFrame, entityToEmbeddings):
	################ Generate gold and corrupted entities ################
	
	mentionsToEntity = mentionsDataFrame["entity"].values.tolist()
	entityIds = list(set([x for x in mentionsToEntity if "NIL" not in x]))
	corruptedEntities = []
	#goldEntities = []
	M = len(mentionsToEntity)
	for m in range(M):
		if ( "NIL" in mentionsDataFrame["entity"][m] ):
			corruptedEntities.append(mentionsDataFrame["entity"][m])
		else:
			#goldEntities.append(entityToEmbeddings[mentionsDataFrame["entity"][m]]) 
			entityId = mentionsDataFrame["entity"][m]
			corrupted_entities = [ei for ei in entityIds if ei != entityId]
			corruptedEntityInt = random.randint(0,len(corrupted_entities)-1)
			corruptedEntityEmbedding = entityToEmbeddings[corrupted_entities[corruptedEntityInt]]
			corruptedEntities.append(corruptedEntityEmbedding)
	#mentionsDataFrame["gold_entity"] = pandas.Series(goldEntities).values	
	mentionsDataFrame["corrupted_entity"] = pandas.Series(corruptedEntities).values
	#############################################################	
	return mentionsDataFrame









		
############################# Knowledge Base ######################################	
def loadKnowledgeBase():
	for subdir, dirs, files in os.walk(knowledgeBaseFile):
		kbFiles = files
	kbList = [knowledgeBaseFile + kb for kb in kbFiles if "sw" not in kb]
	knowledgeBase = {}
	knowledgeBase["id"] = []
	knowledgeBase["name"] = []
	knowledgeBase["text"] = []
	entityToName = {}
	start = time.time()
	for kb in kbList:
		print(kb)
		with open(kb) as kbFile:
			lxmlKb = etree.iterparse(kb, events=('end',), tag='entity')
			lxmlKbs = [lk for lk in lxmlKb]
			for kbes in lxmlKbs:
				entityId = kbes[1].get("id")
				entityName = kbes[1].get("name")
				knowledgeBase["id"].append(entityId)
				knowledgeBase["name"].append(entityName)
				#knowledgeBase["text"].append(kbes[1].find("wiki_text").text.lower().replace("\n", " "))
				textContent = kbes[1].find("wiki_text").text.lower()
				textContent = textContent.split(". ")
				textContent = " ".join(textContent[:4])
				knowledgeBase["text"].append(textContent)
				entityToName[entityId] = entityName 
	
	with open(defaultDataFolder + challenge2015  + "/json/knowledgeBase.json", "w") as knowledgeBaseJson:
		json.dump(knowledgeBase, knowledgeBaseJson)

	with open(defaultDataFolder + challenge2015  + "/json/entityToName.json", "w") as entityToNameJson:
		json.dump(entityToName, entityToNameJson)                                                 	
	
	knowledgeDataFrame = pandas.DataFrame(data=knowledgeBase) 
	print("Knowledge base built, took : " + str(time.time()-start))

	return entityToName,knowledgeDataFrame
####################################################################################

def loadEmbeddings(dataFolder=defaultDataFolder):
	""" Glove http://nlp.stanford.edu/data/glove.6B.zip """
	start = time.time()
	gloveFilenames = ["glove.6B.50d.txt", "glove.6B.100d.txt", "glove.6B.200d.txt", "glove.6B.300d.txt"]
	#readTypes = []
	#readType = {}
	#for dim in [50, 100, 200, 300]:
	#	for i in range(1,dim):
	#		readType[str(i)] = np.float64
	#	readType[str(0)] = str 
	#	readTypes.append(readType) 
	
	
	#for gf in gloveFilenames[:1]:
	#	embeddings = pandas.read_csv(dataFolder + "Glove/" + gf, dtype=readTypes[0], sep=" ", header = None, quoting=csv.QUOTE_NONE)

	
	embeddingDic = {}
	i = 0
	with open(dataFolder + "Glove/" + gloveFilenames[0]) as f:
		for line in f:
			line_embedding = line.split(" ")
			embeddingDic[line_embedding[0]] =  [float(emb) for emb in line_embedding[1:]]

	with open(defaultDataFolder + challenge2015 + "/json/embeddings.json", "w") as embeddingsFile:
		json.dump(embeddingDic, embeddingsFile)		


	embeddings = pandas.DataFrame(data=embeddingDic)
	print("Embeddings loaded, took : " + str(time.time() - start))
	return embeddings


def crossMapMentions(knowledgeDataFrame, mentionsDataFrame, embeddings):

	"""	Input : dataframe, dataframe, dictionnary
		Output : 2 updated dataframes """
	
	############# Complete mentions with embeddings ###########
	mentionsDataFrame["words"] = mentionsDataFrame["text"].apply(lambda x: x.split(" "))
	M = len(mentionsDataFrame["words"])
	mentionsEmbeddings = []
	goldEmbeddings = []
	corruptedEmbeddings = []
	mentionsToEntity = mentionsDataFrame["entity"].values.tolist()
	entityIds = list(set([x for x in mentionsToEntity if "NIL" not in x]))
	nbWordsM = []
	nbWordsE = []
	nbWordsC = []	
	for m in range(M):
		entity_words = []	
		mention_words = []
		corrupted_words_embeddings = []
		#corruptedEntities = []
		#goldEntities = []
		#
		#if ( "NIL" in mentionsDataFrame["entity"][m] ):
        #	corruptedEntities.append(-1)
        #else:
		entityId = mentionsDataFrame["entity"][m]	
		if ( "NIL" in entityId ):
			goldEmbeddings.append(-1)
			corruptedEmbeddings.append(-1)
			nbWordsC.append(0)
			nbWordsE.append(0)

		else:	
			goldKnowledge = knowledgeDataFrame[knowledgeDataFrame["id"] == entityId]
        	#	#goldEntities.append(entityToEmbeddings[mentionsDataFrame["entity"][m]]) 
        	#	entityId = mentionsDataFrame["entity"][m]
        	#	corrupted_entities = [ei for ei in entityIds if ei != entityId]
        	#	corruptedEntityInt = random.randint(0,len(corrupted_entities)-1)
        	#	corruptedEntityEmbedding = entityToEmbeddings[corrupted_entities[corruptedEntityInt]]
        	#	corruptedEntities.append(corruptedEntityEmbedding)
			words = goldKnowledge["text"].values.tolist()[0]
			W = len(words)
			for w in range(W):
				try:
					word_w = words[w]
					word = embeddings[word_w]
					entity_words.append(word) # Word embedding, at last!
				except:
					pass
			if ( len(entity_words) == 0):
				goldEmbeddings.append(-1)
				corruptedEmbeddings.append(-1)
			else:
				nbWordsE.append(len(entity_words))
				nbWords = np.min([nbWordsE, 10])
				#averageEmbeddings = np.mean(entity_words, 0)
				#goldEmbeddings.append(averageEmbeddings)
				goldEmbeddings.append(entity_words[:nbWords])
				not_found_corrupted_entity = True
				count = 0
				while ( ( count < 100 ) and not_found_corrupted_entity ): 
					corrupted_entities = [ei for ei in entityIds if ei != entityId]
					corruptedEntityInt = random.randint(0,len(corrupted_entities)-1)
					corruptedKnowledge = knowledgeDataFrame[knowledgeDataFrame["id"] == corrupted_entities[corruptedEntityInt]]
					corruptedEntityWords = corruptedKnowledge["text"].values.tolist()[0] 
					C = len(corruptedEntityWords)
					for ce in range(C):
						try:
							word_e = corruptedEntityWords[ce]
							word_entity = embeddings[word_e]
							corrupted_words_embeddings.append(word_entity) # Word embedding, at last!
						except:
							pass

					if ( len(corrupted_words_embeddings ) > 0 ):
						not_found_corrupted_entity = False 
					count += 1
	
				if not_found_corrupted_entity == True:
					nbWordsC.append(0)
					corruptedEmbeddings.append(-1)
				else:
					nbWordsC.append(len(corrupted_words_embeddings))
					nbWords = np.min([len(corrupted_words_embeddings), 10])
					corruptedEmbeddings.append(corrupted_words_embeddings[:nbWords])
					#corruptedEmbeddings.append(np.mean(corrupted_words_embeddings,0))


		W = len(mentionsDataFrame["words"][m])
		for w in range(W):
			#print("number of words : " + str(w))
			try:
				word_w = mentionsDataFrame["words"][m][w]	
				word = embeddings[word_w]	
				mention_words.append(word) # Word embedding, at last!	
			except:
				pass
	                                                                                     
		if ( len(mention_words) == 0 ):
			#print("No embeddings for this mention")
			mentionsEmbeddings.append(-1)
		else:
			nbWords = np.min([len(mention_words), 10])
			mentionsEmbeddings.append(mention_words[:nbWords])
			#mentionsEmbeddings.append(np.mean(mention_words, 0))
	mentionsDataFrame["gold_embeddings"] = pandas.Series(goldEmbeddings).values
	mentionsDataFrame["corrupted_embeddings"] = pandas.Series(corruptedEmbeddings).values
	mentionsDataFrame["embeddings"] = pandas.Series(mentionsEmbeddings).values
				
	###########################################################

	
	return mentionsDataFrame

def crossMapKB(knowledgeDataFrame, embeddings):                                                                     
	############# Knowledge base and embeddings ###############
	#knowledgeDataFrame["text"] = knowledgeDataFrame["text"].apply(lambda x: x.split(" "))
	print("Text into kb loaded")
	M = len(knowledgeDataFrame["name"])
	entityEmbeddings = []
	entityToEmbeddings = {}
	print("About 1 minute")
	for m in range(M):
		entity_words = []
		W = len(knowledgeDataFrame["text"][m])
		for w in range(W):
			try:
				word_w = knowledgeDataFrame["text"][m][w]
				word = embeddings[word_w].as_matrix()
				entity_words.append(word) # Word embedding, at last!
			except:
				pass
		if ( len(entity_words) == 0 ):
			#print("No embeddings for this entity")
			entityEmbeddings.append(-1)
			entityToEmbeddings[knowledgeDataFrame["id"][m]] = -1 
		else:
			averageEmbeddings = np.mean(entity_words, 0)
			entityEmbeddings.append(averageEmbeddings)
			entityToEmbeddings[knowledgeDataFrame["id"][m]] = averageEmbeddings
	with open(defaultDataFolder + challenge2015 + "/json/entToEmbs.json") as entToEmbs:
		json.dump(entityToEmbeddings, entToEmbs)	
	knowledgeDataFrame["embeddings"] = pandas.Series(entityEmbeddings).values
	###########################################################	
	return entityToEmbeddings, knowledgeDataFrame, mentionsDataFrame


if __name__ == "__main__":
	print("Parsing tac-kbp")
	#entityName, knowledgeDataFrame = loadKnowledgeBase()
	queriesName, mentionsDataFrame = loadMentions()
	#embeddings = loadEmbeddings()
	############## Complete mentions with embeddings ###########
	#mentionsDataFrame["words"] = mentionsDataFrame["text"].apply(lambda x: x.split(" "))
	#M = len(mentionsDataFrame["words"])
	#mentionsEmbeddings = []
	#for m in range(M):
	#	mention_words = []
	#	W = len(mentionsDataFrame["words"][m])
	#	print(M-1-m)
	#	for w in range(W):
	#		#print("number of words : " + str(w))
	#		try:
	#			word_w = mentionsDataFrame["words"][m][w]	
	#			word = embeddings[word_w].as_matrix()	
	#			mention_words.append(word) # Word embedding, at last!
	#		except:
	#			pass

	#	if ( len(mention_words) == 0 ):
	#		print("No embeddings for this mention")
	#		mentionsEmbeddings.append(-1)
	#	else:
	#		mentionsEmbeddings.append(np.mean(mention_words, 0))
	#mentionsDataFrame["embeddings"] = pandas.Series(mentionsEmbeddings).values	
	## mentionsDataFrame.loc(mentionsDataFrame["embeddings"] != None)["embeddings"]
	############################################################	

	############## Knowledge base and embeddings ###############
	##knowledgeDataFrame["text"] = knowledgeDataFrame["text"].apply(lambda x: x.split(" "))
	##print("Text into kb loaded")
	##pdb.set_trace()
	##M = len(knowledgeDataFrame["name"])
	##entityEmbeddings = []
	##for m in range(M):
	##	entity_words = []
	##	W = len(knowledgeDataFrame["text"][m])
	##	print("entity " + str(m))
	##	for w in range(W):
	##		try:
	##			word_w = knowledgeDataFrame["text"][m][w]
	##			word = embeddings[word_w].as_matrix()
	##			entity_words.append(word) # Word embedding, at last!
	##		except:
	##			pass
	##	if ( len(entity_words) == 0 ):
	##		print("No embeddings for this entity")
	##		entityEmbeddings.append(-1)
	##	else:
	##		entityEmbeddings.append(np.mean(entity_words, 0))
	##knowledgeDataFrame["embeddings"] = pandas.Series(entityEmbeddings).values
	##############################################################	
	#
	################# Generate corrupted entities ################
	#mentionsToEntity = mentionsDataFrame["entity"].values.tolist()
	#entityIds = list(set([x for x in mentionsToEntity if "NIL" not in x]))
	#corruptedEntities = []
	#for m in range(M):
	#	if ( "NIL" in mentionsDataFrame["entity"][m] ):
	#		corruptedEntities.append(mentionsDataFrame["entity"][m])
	#	else:
	#		entityId = mentionsDataFrame["entity"][m]
	#		corrupted_entities = [ei for ei in entityIds if ei != entityId]
	#		corruptedEntityInt = random.randint(0,len(corrupted_entities)-1)
	#		corruptedEntityEmbedding = entityToEmbeddings[corrupted_entities[corruptedEntityInt]]
	#		corruptedEntities.append(corruptedEntityEmbedding)
	#			
	#mentionsDataFrame["corrupted_entity"] = pandas.Series(corruptedEntities).values
	##############################################################	
	#


	######### Load from pickle #########

		
	jsonFolder = defaultDataFolder + challenge2015 + "/json/"
	####################################
	with open(jsonFolder + "mentionsWithEmbeddings.pickle", "r") as mentionsEmbsFile: 
		mentionsDataFrame = pickle.load(mentionsEmbsFile)

	######### Load from json files ###############@
	
	#with open(jsonFolder + "mentions.json") as mentionsJson:
	#	mentions = json.load(mentionsJson)
	#print("Mentions loaded ...")	


	with open(jsonFolder + "knowledgeBase.json") as kbJson:
		knowledgeBase = json.load(kbJson)
	print("Knowledge base loaded ...")
	

	with open(jsonFolder + "embeddings.json") as embeddingsJson:
		embs = json.load(embeddingsJson)
	print("Embeddings loaded ...")

	#


	knowledgeDataFrame = pandas.DataFrame(data=knowledgeBase)
	knowledgeDataFrame["text"] = knowledgeDataFrame["text"].apply(lambda x: x.split(" "))
	#mentionsDataFrame = pandas.DataFrame(data=mentions)
	#del mentions, knowledgeBase
	print("Mapping mentions and embeddings ...")
	#mentionsDataFrame = crossMapMentions(knowledgeDataFrame, mentionsDataFrame, embs)
	#mentionsDataFrame["embs_check"] = mentionsDataFrame["embeddings"].apply(typeCheck)
	#mentionsDataFrame["gold_check"] = mentionsDataFrame["gold_embeddings"].apply(typeCheck)
	#mentionsDataFrame["corrupted_check"] = mentionsDataFrame["corrupted_embeddings"].apply(typeCheck)
	#mentionsDataFrame = mentionsDataFrame[mentionsDataFrame["embs_check"] & mentionsDataFrame["gold_check"] & mentionsDataFrame["corrupted_check"] ]
	#mentionsDataFrame["corrupted_embeddings"] = mentionsDataFrame["corrupted_embeddings"].apply(lambda x: sum(x, []))	
	#mentionsDataFrame["gold_embeddings"] = mentionsDataFrame["gold_embeddings"].apply(lambda x: sum(x, []))
	#mentionsDataFrame["embeddings"] = mentionsDataFrame["embeddings"].apply(lambda x: sum(x, []))
	#with open(jsonFolder + "mentionsWithEmbeddings.pickle", "w") as mentionsEmbsFile: 
	#	pickle.dump(mentionsDataFrame, mentionsEmbsFile)
	#mentionAndContextLength = mentionsDataFrame["embeddings"].apply(len)
	pdb.set_trace()
	#mentionEmbeddingsShort = mentionsDataFrame[mentionAndContextLength < 500]
	#500 - len(x)
	#append(np.zeros()) 
	def fillZeros(x):
		L = len(x)
		for i in range(500 - L):
			x.append(0)
	mentionsDataFrame["embeddings"].apply(fillZeros)
	pdb.set_trace()	
	#mentionsDataFrame = generateGoldAndCorruptedEntities(mentionsDataFrame, entityToEmbeddings)
	

	#
	print("Done")
