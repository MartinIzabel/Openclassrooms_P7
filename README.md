# Openclassrooms_P7

Modèle de Scoring Client 

* 1 Traitement des valeurs manquantes par simpleimputer en remplaçant par la médiane.   
* 2 Equilibrage des données avec undersampling
* 3 Algorithme: Lightgbm optimisé avec randomizersearch + seuil optimal avec Hyperopt
* 4 Creation d'un dashboard avec la librairie Streamlit + Déploiement sur Heroku 

lien vers le Git : https://github.com/MartinIzabel/Openclassrooms_P7
lien vers le dashboard : https://openclassrooms-p7.herokuapp.com/

Fonctionalités du dashboard

* 1 Bouton 'Prédire' donnant le score client et sa position vis à vis du seuil de décision   
* 2 Un graphique de la répartition des prêts acceptés vs refusées en pie-chart
* 3 LIME tool pour expliquer le score client, les champs les plus impactants selon quels seuils 
* 4 Une partie exploration de la donnée avec un pie chart des top 10 features d'importances
* 5 Un tableau des valeurs du clients séléctionné sur les top 10 features
* 6 Un outil de séléction de la variable à étudier + un graph de la distribution des clients en fonction de la variable de prédiction + le positionnement relatif de la valeur client