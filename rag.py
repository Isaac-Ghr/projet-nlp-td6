#   =================
#   ISAAC GHORZI
#   TD 3
#   22400381
#   =================

from sentence_transformers import SentenceTransformer as st
import numpy as np
import json as js
import sys

# récupère et renvoie les données du fichier "tweets.txt" sous forme de liste
# contenant chaque lignes du document.
def read_tweets():
    try:
        f = open("tweets.txt", 'r')
    except:
        raise Exception("Le fichier des tweets n'existe pas : \"tweets.txt\"\n")
    else:
        data = f.readlines()
        f.close()
    return data

# génère les embeddings à partir d'un fichier texte (ici, "tweets.txt")
# et stocke les embeddings dans un fichier json dans lequel chaque objet contient :
	# l'identifiant "id" du tweet;
	# le contenu textuel du tweet;
	# le vecteur d'embedding du tweet;
def gen_embeds(model: st):
    data = read_tweets()
    embeddings = model.encode(data)

    doclist = []
    index = 1
    for document in data:
        doclist.append(
            {
                "id": index,
                "texte": document,
                "vecteur": embeddings[index-1].tolist()
            }
        )
        index += 1

    destination = open("embeddings-tweets.json", 'w')
    js.dump(doclist, destination, indent=4, ensure_ascii=False)
    destination.close()

# transforme une liste de <str> en une seule chaîne de caractères
def parse_input(args: list) -> str:
    return ' '.join(args).strip()

# calcule la simarité cosinus entre deux vecteurs a et b
def simil_cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# génère les embeddings si le fichier n'existe pas
def init(model: st, quiet = True):
    try:
        f = open("embeddings-tweets.json", 'r')
    except:
        if quiet == False : print("Génération des embeddings en cours..\n")
        gen_embeds(model)
        f = open("embeddings-tweets.json", 'r')
    finally:
        f.close()
    if quiet == False : print("Initialisation terminée !\n")

# cette fonction permet de démarrer le programme
def start(model: st, quiet = True):
    if quiet == False : print("Lancement du programme")
    init(model)
    args = sys.argv[1:]
    cli = False
    msg = "Saisissez votre requête (laisser vide pour quitter) :\n"

    if len(args) > 0:
        req = parse_input(args)
        cli = True
    else:
        req = input(msg)

    if req.strip() == "":
        return

    while True:
        resultat = list()
        with open("embeddings-tweets.json", 'r') as jsf:
            jsdata = js.load(jsf)
            for tweet in jsdata:
                resultat.append((simil_cos(np.array(model.encode(req)), tweet["vecteur"]), tweet["texte"].strip()))

        resultat = sorted(resultat, reverse=True)
        print("Voici les tweets les plus pertinent :\n")
        position = 1
        for tweet in resultat:
            if tweet[0] >= 0.5 : print(f"{position}. {tweet[1]}\nscore de similarité : {float('%.2f'%tweet[0])}\n")
            position += 1

        if cli : break
        req = input(msg)
        if req.strip() == "":
            break

    if quiet == False : print("Fin du programme")

model = st("paraphrase-MiniLM-L6-v2")
start(model)
