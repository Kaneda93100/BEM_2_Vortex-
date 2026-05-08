Les codes de cette branche ont pour objectif de permettre la correction du modèle BEM (solveur grossier) par rapport au modèle Free Wake Vortex (solveur fin). 

Les géométries des éoliennes d'un parc sont fixés. De plus, les grilles de discrétisations radiale et azimutale sont également fixés. 

La correction par apprentissage supervisée prendra en covariables tout un ensemble de paramètres environnementaux et tentera d'inférer sur les forces normales et tangentielles qui seront appliquées sur une pale. 

L'objectif sur le long terme est d'accélérer la procédure de calcul de ces efforts en utilisant plutôt la BEM que Free Wake Vortex.