"""Normalisation and categorisation of the political parties.

Help of AI to verify some historical affiliations but mostly manual work, with several iterations.

Author: Emma Leguay, ENSAE 3A - MiE M2, 2025-2026"""

import re
import pandas as pd
from collections import Counter


# =============================================================================
# DICTIONARY 1 : NORMALIZATION OF VALUES OF THE COLUMNS
# =============================================================================
# Variations in order or othorgraphe -> transformed into one canonical name. 
# Variations in OCR or typos -> corrected.
# Historical distinctions are kept(UDR/RPR, CD/CDP/CDS, LC/LCR, PS/PSU, etc.)

COMPONENT_NORMALIZATION = {
    # --- PCF et communistes d'outre-mer ---
    "Parti communiste français": "Parti communiste français (PCF)",
    "Parti communiste réunionnais": "Parti communiste réunionnais",
    "Parti communiste guadeloupéen": "Parti communiste guadeloupéen",
    "communiste": "communiste (générique)",
    "communistes": "communiste (générique)",

    # --- Parti socialiste (different from PSU) ---
    "Parti socialiste": "Parti socialiste (PS)",
    "Socialiste": "socialiste (générique)",
    "socialiste": "socialiste (générique)",
    "Socialistes": "socialiste (générique)",
    "socialistes": "socialiste (générique)",
    "socialiste indépendant": "socialiste (générique)",
    "Présence socialiste": "socialiste (générique)",
    "socialiste chrétien": "socialiste (générique)",
    "Socialisme et démocratie": "socialiste (générique)",
    "socialiste non inscrit": "socialiste (générique)",
    "socialisme": "socialiste (générique)",
    "socialisme libéral": "socialiste (générique)",
    "républicaine socialiste": "socialiste (générique)",
    "socialiste humaniste": "Socialistes humanistes",
    "Initiative républicaine socialiste": "Initiative socialiste et républicaine",
    "Comité départemental de soutien à François Mitterrand": "Comité de soutien à François Mitterrand",
    "indépendants de gauche": "gauche (générique)",

    # --- PSU (distinct from the PS, until 1989) ---
    "Parti socialiste unifié": "Parti socialiste unifié (PSU)",

    # --- MRG (Radicaux de gauche, ally of the PS) ---
    "Mouvement des radicaux de gauche": "Mouvement des radicaux de gauche (MRG)",
    "Radicaux de gauche": "Mouvement des radicaux de gauche (MRG)",
    "Radical de gauche": "Mouvement des radicaux de gauche (MRG)",
    "Mouvement de la gauche radicale": "Mouvement des radicaux de gauche (MRG)",  
    "Mouvement de la gauche radicale-socialiste": "Mouvement des radicaux de gauche (MRG)",

    # --- Parti radical valoisien (centriste, ally of the UDF) — distinct from MRG ---
    "Comités d'initiative pour une nouvelle politique de gauche": "Comités d'initiative",
    "Parti radical-socialiste": "Parti radical (PRV)",
    "Radical-socialiste": "Parti radical (PRV)",
    "Radicaux-socialistes": "Parti radical (PRV)",  
    "Parti radical": "Parti radical (PRV)",
    "Radical": "Parti radical (PRV)",
    "Parti Radical": "Parti radical (PRV)",          
    "Radicaux": "Parti radical (PRV)",                 
    "radical": "Parti radical (PRV)",
    "radical-socialiste": "Parti radical (PRV)",
    "radicaux-socialistes": "Parti radical (PRV)",
    "Radicaux valoisiens": "Parti radical (PRV)",
    "Parti radical-socialiste valoisien": "Parti radical (PRV)",
    "Parti républicain valoisien": "Parti radical (PRV)",
    "Mouvement radical": "Parti radical (PRV)",
    "Parti radical et radical-socialiste": "Parti radical (PRV)",
    "Parti républicain radical et radical-socialiste": "Parti radical (PRV)",
    "radicaux-travaillistes": "Parti radical (PRV)",
    "Radicaux-Réformateurs": "Parti radical (PRV)",
    "radical réformateur": "Parti radical (PRV)",
    "radical indépendant": "Parti radical (PRV)",
    "radicale": "Parti radical (PRV)",
    "républicain radical": "Parti radical (PRV)",
    "radical-socialiste réformateur": "Parti radical (PRV)",
    "radical-socialiste indépendant": "Parti radical (PRV)",
    "Comité radical-socialiste": "Parti radical (PRV)",
    "Comité radical et radical-socialiste": "Parti radical (PRV)",
    "centristes radicaux-socialistes": "Parti radical (PRV)",
    "Radicaux et radicaux-socialistes": "Parti radical (PRV)",
    "Parti des radicaux modérés": "Parti radical (PRV)",

    # --- Gaullistes (UDR then RPR) ---
    "Rassemblement pour la République": "Rassemblement pour la République (RPR)",
    "Union des démocrates pour la République": "Union des démocrates pour la République (UDR)",
    "Union des démocrates pour la Ve République": "Union des démocrates pour la République (UDR)",  # variante OCR
    "Union des républicains de progrès": "Union des républicains de progrès (URP)",
    "Union des gaullistes de progrès": "Union des gaullistes de progrès (UGP)",
    "Union des jeunes pour le progrès": "Union des jeunes pour le progrès (UJP)",
    "Union démocratique du travail": "Union démocratique du travail (UDT)",
    "gaulliste": "gaulliste (générique)",
    "gaulliste indépendant": "gaulliste (générique)",
    "gaullistes": "gaulliste (générique)",
    "Gaullistes d'opposition": "gaulliste (générique)",
    "gaulliste majoritaire": "gaulliste (générique)",
    "gaulliste d'opposition": "gaulliste (générique)",
    "gaulliste Ve République": "gaulliste (générique)",
    "Gaulliste de progrès": "Union des gaullistes de progrès (UGP)",
    "Fédération des gaullistes de progrès": "Union des gaullistes de progrès (UGP)",
    "Mouvement des gaullistes populaires": "gaulliste (générique)",
    "Démocrates Ve": "gaulliste (générique)",
    "Démocrate Ve": "gaulliste (générique)",
    "Démocrates Ve République": "gaulliste (générique)",
    "Démocrate Ve République": "gaulliste (générique)",
    "Mouvement des démocrates Ve": "gaulliste (générique)",
    "Dem Ve": "gaulliste (générique)",
    "Ve République": "gaulliste (générique)",
    "Carrefour Ve": "gaulliste (générique)",
    "gaulliste de rassemblement et d'union pour la 5e République": "gaulliste (générique)",
    "indépendant gaulliste": "gaulliste (générique)",
    "Association nationale d'action pour la fidélité au général de Gaulle": "gaulliste (générique)",
    "Association pour la Ve République": "gaulliste (générique)",
    "Action et fidélité gaulliste": "gaulliste (générique)",
    "Mouvements gaullistes et démocrates": "gaulliste (générique)",
    "Mouvements gaullistes et des démocrates": "gaulliste (générique)",
    "Comité des gaullistes d'opposition": "gaulliste (générique)",
    "Mouvement des gaullistes d'opposition": "gaulliste (générique)",
    "Rassemblement des gaullistes libres": "gaulliste (générique)",
    "Union gaulliste de l'Yonne": "gaulliste (générique)",
    "Association des gaullistes de Bretagne et Sarthe": "gaulliste (générique)",
    "gaullistes de la majorité présidentielle": "gaulliste (générique)",
    "Gaullistes de progrès et d'opposition": "Union des gaullistes de progrès (UGP)",
    "Gaulliste de progrès et d'action sociale": "Union des gaullistes de progrès (UGP)",
    "Elus gaullistes du progrès": "Union des gaullistes de progrès (UGP)",
    "Union des démocrates pour la Cinquième République": "Union des démocrates pour la République (UDR)",
    "Union démocrate Ve République": "Union des démocrates pour la République (UDR)",
    "gaulliste de gauche": "gaullistes de gauche",
    "gaullisme de gauche": "gaullistes de gauche",
    "Gauche Ve République": "gaullistes de gauche",
    "Comité national des gaullistes de gauche": "Mouvement des gaullistes de gauche",
    "Fédération Rhône-Alpes des gaullistes de gauche": "Mouvement des gaullistes de gauche",
    "Républicains de progrès de Léo Hamon": "Mouvement des gaullistes de gauche",
    "indépendant divers droite gaulliste": "divers droite",
    "libre divers droite": "divers droite",
    "Progrès et liberté": "Progrès et Liberté",
    "républicains": "républicain (générique)",
    "libéraux": "libéral",

    # --- Variantes giscardiennes / RI / libérales ---
    "giscardien déçu": "giscardien",
    "giscardien de progrès": "giscardien",
    "Giscardiens": "giscardien",
    "républicain giscardien": "giscardien",
    "radical giscardien": "giscardien",
    "droite libérale tendance giscardienne": "giscardien",
    "droite libérale indépendante": "droite libérale",
    "droite libérale et républicaine": "droite libérale",
    "droite libérale démocratique": "droite libérale",
    "Républicain indépendant démocrate": "Républicain indépendant (RI)",
    "républicain indépendant": "Républicain indépendant (RI)",
    "républicain démocrate indépendant": "Républicain indépendant (RI)",
    "Fédération nationale des Indépendants": "Fédération nationale des indépendants",
    "Fédération départementale des indépendants de Charente-Maritime": "Centre national des indépendants et paysans (CNIP)",
    "Jeunes Indépendants": "Centre national des indépendants et paysans (CNIP)",
    "Centre National": "Centre national des indépendants et paysans (CNIP)",
    "Fédération libérale des Hauts-de-Seine": "Parti libéral",

    # --- UDF and centrists (CD/CDP/CDS) ---
    "Union pour la démocratie française": "Union pour la démocratie française (UDF)",
    "Parti républicain": "Parti républicain (PR)",
    "Républicains indépendants": "Républicains indépendants (RI)",
    "Républicain indépendant": "Républicain indépendant (RI)",
    "Centre des démocrates sociaux": "Centre des démocrates sociaux (CDS)",
    "Centre démocrate": "Centre démocrate (CD)",
    "Centre Démocratie et Progrès": "Centre Démocratie et Progrès (CDP)",
    "Centre républicain": "Centre républicain",
    "Mouvement réformateur": "Mouvement réformateur",
    "Mouvement des réformateurs": "Mouvement réformateur",
    "Réformateurs": "Mouvement réformateur",
    "Réformateur": "Mouvement réformateur",
    "réformateurs": "Mouvement réformateur",
    "Démocratie chrétienne": "Démocratie chrétienne",
    "Démocratie chrétienne française": "Démocratie chrétienne",
    "Chrétiens démocrates": "Démocratie chrétienne",
    "Parti socialiste démocrate": "Parti social-démocrate (PSD)",
    "Parti social démocrate": "Parti social-démocrate (PSD)",
    "Mouvement démocrate socialiste de France": "Parti social-démocrate (PSD)",
    "Mouvement des démocrates socialistes": "Parti social-démocrate (PSD)",
    "Nouveau Contrat Social": "Nouveau Contrat Social",
    "centriste": "centre (générique)",
    "Centre gauche": "centre (générique)",
    "Centre": "centre (générique)",
    "Mouvement démocrate socialiste": "Parti social-démocrate (PSD)",  
    "centristes": "centre (générique)", 
    "Nouveau contrat social": "Nouveau Contrat Social",     
    "centre": "centre (générique)",                       
    "Union du centre": "centre (générique)",                   
    "Démocrates de progrès": "centre (générique)",    
    "Centristes": "centre (générique)",
    "centre droit": "centre (générique)",
    "centre gauche": "centre (générique)",
    "centre libéral": "centre (générique)",
    "centre indépendant": "centre (générique)",
    "Centre indépendant": "centre (générique)",
    "démocrate": "centre (générique)",
    "démocrates": "centre (générique)",
    "Démocrate": "centre (générique)",   
    "Clubs perspectives et réalités": "Clubs Perspectives et réalités",
    "Clubs Perspectives et Réalités": "Clubs Perspectives et réalités",
    "Démocrates Chrétiens": "Démocratie chrétienne",
    "Démocrates chrétiens": "Démocratie chrétienne",
    "Démocrate socialiste": "Parti social-démocrate (PSD)",
    "Démocrates socialistes": "Parti social-démocrate (PSD)",
    "Démocrates sociaux": "Parti social-démocrate (PSD)",
    "Mouvement des démocrates sociaux": "Parti social-démocrate (PSD)",
    "Mouvement social-démocrate": "Parti social-démocrate (PSD)",
    "Mouvement social démocrate de France": "Parti social-démocrate (PSD)",
    "Mouvement des démocrates socialistes de France": "Parti social-démocrate (PSD)",
    "Socialistes-démocrates": "Parti social-démocrate (PSD)",
    "Sociaux-démocrates": "Parti social-démocrate (PSD)",
    "social-démocrate": "Parti social-démocrate (PSD)",
    "sociaux-démocrates": "Parti social-démocrate (PSD)",
    "Parti socialiste-démocrate": "Parti social-démocrate (PSD)",
    "Démocrate de progrès": "centre (générique)",
    "Association des démocrates": "centre (générique)",
    "centriste de progrès": "centre (générique)",
    "centriste de progrès et de justice sociale": "centre (générique)",
    "centriste démocrate indépendant": "centre (générique)",
    "centriste gaulliste": "centre (générique)",
    "Extrême-Centre": "centre (générique)",
    "Union centriste des démocrates de progrès": "centre (générique)",
    "Centristes Duhamel": "centre (générique)",
    "Centristes de progrès": "centre (générique)",
    "centristes de la majorité": "centre (générique)",
    "centristes de progrès démocrates socialistes": "centre (générique)",
    "centre-centre gauche": "centre (générique)",
    "centre libéral indépendant": "centre (générique)",
    "Comité d'action centriste et républicaine": "centre (générique)",
    "Centre républicain pour une démocratie moderne": "centre (générique)",
    "Centre Démocratie et Progrès et Démocratie moderne": "Centre Démocratie et Progrès (CDP)",
    "Comité pour un Nouveau contrat social": "Comité d'études pour un Nouveau Contrat Social",
    "Union centriste du XVIIIe": "Union centriste",
    "Union centriste républicaine indépendante": "Union centriste",
    "Union des centristes de France": "centre (générique)",
    "Jeunes Démocrates": "centre (générique)",
    "Club des démocrates": "centre (générique)",
    "Rassemblement des démocrates normands": "centre (générique)",
    "Républicains et démocrates de progrès": "centre (générique)",
    "Rassemblement des républicains et des démocrates de progrès": "centre (générique)",
    "républicaine démocrate": "centre (générique)",
    "Centrisme-Réformateur": "Mouvement réformateur",
    "Parti réformiste": "Mouvement réformateur",
    "réformiste": "Mouvement réformateur",
    "Union des démocrates chrétiens": "Démocratie chrétienne",
    "Rassemblement des démocrates chrétiens": "Démocratie chrétienne",
    "Parti démocrate chrétien": "Démocratie chrétienne",
    "Fédération française de démocratie chrétienne": "Démocratie chrétienne",
    "Parti social-démocrate": "Parti social-démocrate (PSD)",
    "Parti socialiste démocratique": "Parti social-démocrate (PSD)",
    "Parti démocrate socialiste": "Parti social-démocrate (PSD)",
    "Sociaux-Démocrates": "Parti social-démocrate (PSD)",
    "Mouvement socialiste-démocrate européen": "Parti social-démocrate (PSD)",
    "Fédération des socialistes démocrates": "Parti social-démocrate (PSD)",
    "républicain démocrate social": "Parti social-démocrate (PSD)",
    "Républicain populaire": "Mouvement républicain populaire",
    "Mouvement européen français": "Mouvement européen",
    "Mouvement fédéraliste européen": "Mouvement européen",
    "Parti européen": "Mouvement européen",
    "Association démocratie française": "Union pour la démocratie française (UDF)",
    "modéré": "modérés",             

    # --- Independant right / liberals / CNIP ---
    "Centre national des indépendants": "Centre national des indépendants et paysans (CNIP)",
    "Centre national des indépendants et paysans": "Centre national des indépendants et paysans (CNIP)",
    "Alliance républicaine indépendante et libérale": "Alliance républicaine indépendante et libérale",
    "Action républicaine indépendante et libérale": "Alliance républicaine indépendante et libérale",
    "Parti libéral de France": "Parti libéral de France",
    "Centre des républicains libres": "Centre des républicains libres",
    "Fédération des républicains de progrès": "Fédération des républicains de progrès",
    "Républicains de progrès": "Fédération des républicains de progrès",
    "divers droite": "divers droite",
    "Républicain de progrès": "Fédération des républicains de progrès",
    "Union savoyarde des républicains de progrès": "Fédération des républicains de progrès",

    # --- Extreme right ---
    "Front national": "Front national (FN)",
    "Parti des forces nouvelles": "Parti des forces nouvelles (PFN)",
    "Centre démocrate et républicain": "Centre démocrate et républicain (CDR)",  
    "Action royaliste": "Action royaliste",
    "Union royaliste": "Action royaliste",
    "Alsace d'abord": "Alsace d'abord",

    # --- Extreme left trotskyst / maoist ---
    "Lutte ouvrière": "Lutte ouvrière (LO)",
    "Ligue communiste révolutionnaire": "Ligue communiste révolutionnaire (LCR)",
    "Ligue communiste": "Ligue communiste révolutionnaire (LCR)",  
    "Ligue communiste section française de la quatrième internationale": "Ligue communiste révolutionnaire (LCR)",
    "Organisation communiste internationaliste": "Organisation communiste internationaliste (OCI)",
    "Alliance des jeunes pour le socialisme": "Organisation communiste internationaliste (OCI)", 
    "Parti des travailleurs": "Parti des travailleurs (PT)",
    "Alliance européenne des travailleurs": "Parti des travailleurs (PT)",
    "Entente internationale des travailleurs": "Parti des travailleurs (PT)",
    "Organisation communiste des travailleurs": "Organisation communiste des travailleurs (OCT)",
    "marxistes-léninistes": "Maoïstes",
    "communistes fidèles au marxisme-léninisme et à la pensée de Mao Tsé-toung": "Maoïstes",

    # --- Autogestionarist / alternative left ---
    "Comités communistes pour l'autogestion": "Comités communistes autogestionnaires",
    "Comités Juquin": "Comités Juquin",
    "Front des jeunes progressistes": "Front progressiste",
    "Comités d'initiative pour une nouvelle politique à gauche": "Comités d'initiative",
    "Mouvement des démocrates": "Mouvement des démocrates (Jobert)",
    "gauche": "gauche (générique)",
    "Groupe autogestionnaire": "autogestionnaire",
    "Comités pour l'autogestion": "Comité pour l'autogestion socialiste",
    "Mouvement pour le désarmement la paix": "Mouvement pour le désarmement la paix la liberté",

    # --- Environmentalists ---
    "Verts alternatifs": "Verts",
    "Ecologie les Verts": "Verts",
    "Nouveaux écologistes du rassemblement nature et animaux": "Nouveaux écologistes du rassemblement nature et animaux (NERNA)",
    "Amis de la Terre": "Amis de la Terre",
    "SOS Environnement": "SOS Environnement",
    "SOS environnement": "SOS Environnement",
    "écologiste": "écologiste (générique)",
    "écologistes": "écologiste (générique)",
    "écologiste indépendant": "écologiste (générique)",
    "VRAIS écologistes": "écologiste (générique)",
    "mouvements écologistes": "écologiste (générique)",
    "mouvements écologiques": "écologiste (générique)",
    "Comités écologiques": "Comité écologique",
    "Fédération Nord-Nature": "Nord-Nature",
    "Fédération Vallée de la Lys-Nature": "Vallée de la Lys-Nature",
    "Fédération des usagers des transports d'Ile-de-France": "Fédération des usagers des transports",
    "Alliance rouge et verte": "Alternative rouge et verte",

    # --- Regions and others ---
    "Union démocratique bretonne": "Parti breton",
    "Parti breton Strollad ar vro": "Parti breton",
    "Parti féministe": "Choisir la cause des femmes",
    "Syndicat national de défense du droit des agriculteurs": "Coordination rurale",
    "Coordination rurale": "Coordination rurale",
    "Collectif du monde rural pour le non à Maastricht": "Coordination rurale",
    "républicain": "républicain (générique)",
    "Association des usagers de l'administration et des services publics": "Coordination rurale",
    "Confédération syndicale du cadre de vie": "Coordination rurale",
    "Volem viure au pais movement socialista e autonomista occitan": "Volem viure au pais",
    "Mouvement socialiste occitan Volem viure au païs": "Volem viure au pais",
    "Movement socialista occitan Volem viure al païs": "Volèm viure al païs",
    "Abertzale socialiste": "abertzale",
    "Parti fédéraliste européen de Catalogne": "Parti fédéraliste européen",
    "Taatiraa Polynesia": "Taatiraa Polynesia Entente polynésienne",
    "Font indépendantiste": "Front indépendantiste",  
    "Cercle communautaire laïc juif": "Mouvement communautaire juif laïc",
    "France Islam laïcité": "France cultures Islam et laïcité",
    "Ras le bol": "Parti du ras le bol",
    "Comité républicain": "Comités républicains",
    "Mouvement européen économique et social": "Mouvement européen",
    "Union régionale d'action européenne": "Mouvement européen",
    "Groupes d'actions municipales": "Groupes d'action municipale",
    "Rassemblement des usagers des services publics des contribuables et des groupements de défense":"Rassemblement des usagers du service public et des contribuables",
    "Humaniste": "humaniste (générique)",
    "humaniste": "humaniste (générique)",
    "bio-humaniste": "humaniste (générique)",
    "Mouvement des humanistes vrais des républicains sincères et des progressistes modérés":"humaniste (générique)",
    "Alternative": "alternative (générique)",

    # --- Non affiliated ---
    "indépendant": "Sans étiquette",
    "indépendants": "Sans étiquette",
    "sans étiquette": "Sans étiquette",
    "sans parti politique": "Sans étiquette",
    "non inscrit": "Sans étiquette",
    "aucun parti politique": "Sans étiquette",
    "libre": "Sans étiquette",
    "apolitique": "Sans étiquette",
    "hors des partis": "Sans étiquette",
    "non inscrits": "Sans étiquette",                 
    "Indépendants": "Sans étiquette",                 
    "libres": "Sans étiquette", 
    "hors des partis politiques": "Sans étiquette",  
    "non politisé": "Sans étiquette",
    "sans appartenance politique": "Sans étiquette",
    "aucune appartenance politique": "Sans étiquette",
    "aucun parti": "Sans étiquette",
    "aucune formation politique": "Sans étiquette",
    "pas de parti politique": "Sans étiquette",
    "apolitiques": "Sans étiquette",
    "contestataire": "Sans étiquette",
    "contre les partis politiques": "Sans étiquette",
    "divers gauche": "Sans étiquette",
    "divers modérés": "Sans étiquette",
    "indépendante": "Sans étiquette",
    "indépendantes": "Sans étiquette",
    "social": "Sans étiquette",
    "société civile": "Sans étiquette",
    "formation politique": "Sans étiquette",
    "partis de la majorité": "Sans étiquette",
    "Organisations chrétiennes": "Sans étiquette",

    # --- Variantes Choisir la cause des femmes ---
    "Centre féminin d'études et d'information Femmes Avenir": "Choisir la cause des femmes",
    "Centre féminin d'études et d'information": "Choisir la cause des femmes",
    "Mouvement femme avenir Centre féminin d'études et d'informations": "Choisir la cause des femmes",
    "Mouvement Femmes-Avenir": "Choisir la cause des femmes",
    "Femme avenir": "Choisir la cause des femmes",
    "Groupe Femmes": "Choisir la cause des femmes",
    "Choisir": "Choisir la cause des femmes",
    "mouvements de femmes": "Choisir la cause des femmes",
    "Condition masculine": "Mouvement pour la condition masculine",

    #  --- Variantes RUSPC ---
    "Rassemblement des usagers et contribuables": "Rassemblement des usagers du service public et des contribuables",
    "Rassemblement des usagers des services publics et des contribuables": "Rassemblement des usagers du service public et des contribuables",                    
}


# =============================================================================
# DICTIONARY 2 : 8 FAMILIES OF PARTIES, FOR ANALYSIS PURPOSES
# =============================================================================

FAMILY_ORDER = [
    "Extrême gauche",
    "Gauche communiste",
    "Gauche socialiste",
    "Gauche alternative",
    "Écologistes",
    "Centre",
    "Droite",
    "Extrême droite",
    "Divers / Sans étiquette",
]

PARTY_FAMILY = {
    # =========================================================================
    # Extrême gauche
    # =========================================================================
    "Lutte ouvrière (LO)": "Extrême gauche",
    "Ligue communiste révolutionnaire (LCR)": "Extrême gauche",
    "Organisation communiste internationaliste (OCI)": "Extrême gauche",
    "Organisation communiste des travailleurs (OCT)" : "Extrême gauche",
    "Parti des travailleurs (PT)": "Extrême gauche",
    "Combat ouvrier": "Extrême gauche",                
    "Maoïstes": "Extrême gauche",
    "Parti communiste révolutionnaire": "Extrême gauche",
    "Parti communiste marxiste-léniniste": "Extrême gauche",
    "Organisation des communistes travailleurs": "Extrême gauche",
    "Organisation communiste de France": "Extrême gauche",
    "Organisation communiste de France marxiste-léniniste": "Extrême gauche",
    "Organisation communiste de France démocratique": "Extrême gauche",
    "Organisation communiste bolchevik": "Extrême gauche",
    "Groupe Révolution socialiste": "Extrême gauche",
    "Groupe socialiste internationaliste": "Extrême gauche",
    "Jeunesses communistes révolutionnaires": "Extrême gauche",
    "Ligue communiste section de la quatrième internationale": "Extrême gauche",
    "Ligue trotskyste": "Extrême gauche",
    "Ligue socialiste des travailleurs": "Extrême gauche",
    "Coordination pour une alternative révolutionnaire": "Extrême gauche",
    "Comité pour une candidature unitaire des révolutionnaires": "Extrême gauche",
    "communiste marxiste-léniniste": "Extrême gauche",
    "communiste critique": "Extrême gauche",
    "communiste révolutionnaire trotskyste": "Extrême gauche",
    "Alternative libertaire": "Extrême gauche",

    # =========================================================================
    # Gauche communiste
    # =========================================================================
    "Parti communiste français (PCF)": "Gauche communiste",
    "Parti communiste réunionnais": "Gauche communiste",
    "Parti communiste guadeloupéen": "Gauche communiste",
    "communiste (générique)": "Gauche communiste",
    "Confédération générale du travail": "Gauche communiste",
    "Mouvement de la jeunesse communiste de France": "Gauche communiste", 
    "Parti communiste martiniquais": "Gauche communiste",
    "Mouvement progressiste guadeloupéen": "Gauche communiste",
    "Union des étudiants communistes": "Gauche communiste",
    "Union des jeunes communistes": "Gauche communiste",
    "communiste démocratique": "Gauche communiste",

    # =========================================================================
    # Gauche socialiste / réformiste
    # =========================================================================
    "Parti socialiste (PS)": "Gauche socialiste",
    "Mouvement des radicaux de gauche (MRG)": "Gauche socialiste",
    "Fédération radicale-socialiste": "Gauche socialiste", 
    "Mouvement des citoyens": "Gauche socialiste",     
    "socialiste (générique)": "Gauche socialiste",
    "Alternative démocratie socialisme": "Gauche socialiste",
    "Confédération française démocratique du travail": "Gauche socialiste",
    "Convention des institutions républicaines": "Gauche socialiste",  
    "Fédération socialiste": "Gauche socialiste",
    "Fédération du Parti socialiste": "Gauche socialiste",
    "Carrefours Société Nouvelle": "Gauche socialiste",  # club Rocard
    "Initiative socialiste": "Gauche socialiste",
    "Initiative socialiste et républicaine": "Gauche socialiste",
    "Initiative démocratique de gauche": "Gauche socialiste",
    "Comité de soutien à François Mitterrand": "Gauche socialiste",
    "Mouvement socialiste populaire": "Gauche socialiste",
    "Objectif socialiste": "Gauche socialiste",
    "Mouvement socialiste": "Gauche socialiste",
    "Socialistes humanistes": "Gauche socialiste",
    "Socialiste-démocrate": "Gauche socialiste",
    "Démocrates de Gauche": "Gauche socialiste",
    "Mouvement démocratique et socialiste de France": "Gauche socialiste",
    "socialistes indépendants": "Gauche socialiste",
    "Mouvement de la gauche socialiste et démocratique": "Gauche socialiste",
    "Gauche socialiste et radicale": "Gauche socialiste",
    "Gauche socialiste et démocrate": "Gauche socialiste",
    "Gauche socialiste": "Gauche socialiste",
    "Gauche démocrate socialiste": "Gauche socialiste",
    "Mouvement de la gauche radicale-socialiste": "Gauche socialiste",  
    "Gauche radicale-socialiste": "Gauche socialiste",
    "Mouvement de la gauche progressiste": "Gauche socialiste",
    "Mouvement de la gauche indépendante": "Gauche socialiste",
    "Démocratie socialiste": "Gauche socialiste",
    "Fédération de l'Education nationale": "Gauche socialiste",
    "Syndicat national des instituteurs": "Gauche socialiste",
    "Club Louise-Michel": "Gauche socialiste",
    "Parti de la Jeune République": "Gauche socialiste",         
    "Club des Jacobins": "Gauche socialiste",                    
    "Témoignage chrétien": "Gauche socialiste",                  
    "Institut de recherche socialiste": "Gauche socialiste",    
    "Forum progressiste": "Gauche socialiste",
    "progressistes": "Gauche socialiste",
    "Union progressiste": "Gauche socialiste",
    "Gauche européenne": "Gauche socialiste",
    "Association pour de nouvelles perspectives à gauche": "Gauche socialiste",

    # =========================================================================
    # Gauche alternative / autogestionnaire
    # =========================================================================
    "Parti socialiste unifié (PSU)": "Gauche alternative",
    "Mouvement des démocrates (Jobert)": "Gauche alternative",
    "Comités Juquin": "Gauche alternative",
    "Comités communistes autogestionnaires": "Gauche alternative",
    "Front progressiste": "Gauche alternative",
    "Comités d'initiative": "Gauche alternative",
    "Solidarité écologie gauche alternative": "Gauche alternative",     
    "Alternative rouge et verte": "Gauche alternative",                 
    "Mouvement pour une alternative non-violente": "Gauche alternative",
    "Mouvement pour le socialisme par la participation": "Gauche alternative",
    "Mouvement solidarité participation": "Gauche alternative",
    "gauche (générique)": "Gauche alternative",
    "Gauche radicale": "Gauche alternative",
    "Collège pour une société de participation": "Gauche alternative",
    "Mouvement pour le désarmement la paix la liberté": "Gauche alternative",
    "Comité Larzac": "Gauche alternative",
    "Mouvement pour une écologie socialiste autogérée": "Gauche alternative",
    "Anjou écologie autogestion": "Gauche alternative",
    "Fédération des élus autogestionnaires": "Gauche alternative",
    "Mouvement rouge et vert": "Gauche alternative",
    "gaullistes de gauche": "Gauche alternative",
    "Mouvement des gaullistes de gauche": "Gauche alternative",
    "Union des gaullistes de gauche": "Gauche alternative",
    "Mouvement anarchiste non-violent": "Gauche alternative",
    "Mouvement d'action non-violente": "Gauche alternative",
    "Groupement de recherche et d'action non violence": "Gauche alternative",
    "Groupe de recherche et d'action non violente": "Gauche alternative",
    "Groupe non-violent Louis-Lecoin": "Gauche alternative",
    "non-violent": "Gauche alternative",
    "Comité pour l'autogestion socialiste": "Gauche alternative",
    "Rennes autogestion socialiste": "Gauche alternative",
    "autogestionnaire": "Gauche alternative",
    "Mouvement socialiste occitan": "Gauche alternative",
    "Comités anti-nucléaires": "Gauche alternative", 
    "Mouvement universel humanitaire du travail SOS chômage": "Gauche alternative",
    "Front travailliste": "Gauche alternative",
    "Comité unitaire autogestionnaire": "Gauche alternative",
    "Information pour les droits des soldats": "Gauche alternative",
    "alternative (générique)": "Gauche alternative",
    "Alternative pour une démocratie sociale": "Gauche alternative",
    "Mouvement autonome solidarité sauvegarde espoir": "Gauche alternative",
    "gauche républicaine libérale": "Gauche alternative",
    "Mouvement de la gauche libérale": "Gauche alternative",
    "Convergence autogestionnaire et écologique": "Gauche alternative",
    "Comité d'action autogestionnaire et écologique": "Gauche alternative",
    "Groupe autogestion de Clamart": "Gauche alternative",
    "Groupe socialiste autogestionnaire": "Gauche alternative",
    "Comité autogestionnaire 93": "Gauche alternative",
    "Autogestion écologie 85": "Gauche alternative",
    "Mouvement socialiste pour la participation": "Gauche alternative",
    "A gauche autrement": "Gauche alternative",
    "Collectif Vraiment à gauche": "Gauche alternative",
    "vert et rouge": "Gauche alternative",
    "Groupes d'action municipale": "Gauche alternative",         
    "Section régionale des Femmes pour la paix": "Gauche alternative",
    "Voie de la Paix": "Gauche alternative",
    "Comité local des objecteurs": "Gauche alternative",
    "Comité anti-outspan": "Gauche alternative",                 
    "Comité anti-raciste de Nanterre": "Gauche alternative",
    "Comité homosexuel de l'Ouest parisien": "Gauche alternative",
    "Comité d'urgence anti-répression homosexuelle": "Gauche alternative",
    "Commission Lip": "Gauche alternative",                     
    "Cahiers Occitanie rouge": "Gauche alternative",    

    # =========================================================================
    # Écologistes
    # =========================================================================
    "Verts": "Écologistes",
    "Génération écologie": "Écologistes",                                 
    "Nouveaux écologistes du rassemblement nature et animaux (NERNA)": "Écologistes",
    "Solidarités écologie": "Écologistes",                                
    "Union nationale écologiste": "Écologistes",                          
    "Alliance pour l'écologie et la démocratie": "Écologistes",           
    "Amis de la Terre": "Écologistes",
    "SOS Environnement": "Écologistes",
    "écologiste (générique)": "Écologistes",
    "Mouvement écologique": "Écologistes",
    "Région verte": "Écologistes",
    "Ecologie et survie mouvement d'écologie politique alsacien": "Écologistes",
    "Mouvement écologiste": "Écologistes",
    "Mouvement écologique unifié": "Écologistes",
    "Mouvement d'écologie politique": "Écologistes",
    "Parti vert": "Écologistes",
    "Verts associatifs": "Écologistes",
    "Verts autonomes": "Écologistes",
    "Verts indépendants": "Écologistes",
    "vert indépendant": "Écologistes",
    "écologistes indépendants": "Écologistes",
    "écologistes indépendants des partis": "Écologistes",
    "écologistes indépendants des partis politiques": "Écologistes",
    "Comité écologique": "Écologistes",
    "Comité anti-pollution": "Écologistes",
    "Groupe écologique": "Écologistes",
    "Groupe écologie": "Écologistes",
    "Génération verte": "Écologistes",
    "Mouvement écologique du Nord": "Écologistes",
    "Demain l'écologie": "Écologistes",
    "Ecologie 92": "Écologistes",
    "Ecologie 2000": "Écologistes",
    "Ecologie 78": "Écologistes",
    "Ecologie 1234": "Écologistes",
    "Ecologie et survie": "Écologistes",
    "Jura écologie": "Écologistes",
    "Nice écologie": "Écologistes",
    "Vosges écologie": "Écologistes",
    "Meudon écologie": "Écologistes",
    "Nord-Nature": "Écologistes",
    "Vallée de la Lys-Nature": "Écologistes",
    "Paris écologie": "Écologistes",
    "Collectif havrais d'écologie": "Écologistes",
    "Mouvement de solidarité avec le Tiers Monde": "Écologistes",  
    "Solidarités": "Écologistes",
    "Ecologie citoyenneté et laïcité": "Écologistes",
    "Centre d'études politiques écologiques et sociales": "Écologistes",
    "Comité anti-nucléaire": "Écologistes",
    "Union des associations et comités pour l'environnement de Paris": "Écologistes",
    "Fédération des usagers des transports": "Écologistes",
    "Ecologie et solidarités multiculturelles": "Écologistes",
    "Comité Sauver Paris": "Écologistes",
    "Comité de soutien à la coopérative écologique": "Écologistes",
    "indépendants radicaux écologistes": "Écologistes",
    "radicaux-écologistes": "Écologistes",
    "gauche libérale et écologiste": "Écologistes",
    "Nature et progrès": "Écologistes",
    "Val-de-Marne écologie": "Écologistes",
    "Nord-Ecologie": "Écologistes",
    "Europe environnement": "Écologistes",
    "Cercle écologique des Hauts-de-Gironde": "Écologistes",
    "Mouvement national de lutte pour l'Environnement": "Écologistes",
    "Solidarité environnement": "Écologistes",
    "Seine-Saint-Denis autrement Ecologie 93": "Écologistes",
    "Mouvement d'écologie spirituelle": "Écologistes",
    "Mouvement d'action écopolitique": "Écologistes",
    "Manche-Sud écologie": "Écologistes",
    "Grenoble écologie": "Écologistes",
    "Aix écologie": "Écologistes",
    "Morlaix écologie": "Écologistes",
    "Boulogne-Billancourt écologie": "Écologistes",
    "Association Castres écologie": "Écologistes",
    "Association Montreuil écologie": "Écologistes",
    "Ecologie occitane": "Écologistes",
    "Ecologie alternatives autogestion": "Écologistes",
    "Ecologie solidarités": "Écologistes",
    "Alternatives écologiques": "Écologistes",
    "Rencontres écologie et solidarités": "Écologistes",
    "Amis saumurois de l'écologie": "Écologistes",
    "Association des élus écologistes de l'Isère": "Écologistes",
    "Association O comme Oxygène": "Écologistes",
    "Avenir écologique des bords de Marne et Seine": "Écologistes",
    "Entente pour la démocratie et l'écologie en Normandie": "Écologistes",
    "En Calvados l'écologie n'est pas à vendre": "Écologistes",
    "Association écologique brayonne": "Écologistes",
    "Fédération écologique girondine": "Écologistes",
    "Fédération écologique des usagers et consommateurs": "Écologistes",
    "Comité antinucléaire de Golfech": "Écologistes",
    "Collectifs Antinucléaire 94": "Écologistes",
    "Rassemblement anti-nucléaire et écologique du Nord Cotentin": "Écologistes",
    "Association pour la sauvegarde des sites de Malville et de Bugey": "Écologistes",
    "Association des Comités Malville": "Écologistes",
    "Comité d'action du Val-de-Marne contre le bruit": "Écologistes",
    "Comité national d'action contre le bruit": "Écologistes",
    "Comités anti-bruit": "Écologistes",
    "Comité anti poids lourds Nord": "Écologistes",
    "Comité de défense contre l'autoroute A15": "Écologistes",
    "Protection contre les rayons ionisants": "Écologistes",
    "Société pour la protection de la forêt de Compiègne": "Écologistes",
    "Société pour l'environnement de la région de Compiègne": "Écologistes",
    "Association pour la défense de l'environnement et de lutte contre la pollution": "Écologistes",
    "Nature Environnement Santé": "Écologistes",
    "Montagne sans frontière": "Écologistes",
    "Biosphère pour demain": "Écologistes",
    "Parti de l'arbre": "Écologistes",
    "Terre prochaine": "Écologistes",
    "France Planète 21": "Écologistes",
    "Agricultura Viva": "Écologistes",
    "Action zoophile": "Écologistes",

    # =========================================================================
    # Centre / centre-droit (UDF et ancêtres, démocrates-chrétiens, libéraux centristes)
    # =========================================================================
    "Union pour la démocratie française (UDF)": "Centre",
    "Centre des démocrates sociaux (CDS)": "Centre",
    "Centre démocrate (CD)": "Centre",
    "Centre Démocratie et Progrès (CDP)": "Centre",
    "Parti républicain (PR)": "Centre",
    "Républicains indépendants (RI)": "Centre",
    "Républicain indépendant (RI)": "Centre",   
    "Parti radical (PRV)": "Centre",
    "Mouvement réformateur": "Centre",
    "Démocratie chrétienne": "Centre",
    "Centre républicain": "Centre",
    "Parti social-démocrate (PSD)": "Centre",
    "Nouveau Contrat Social": "Centre",
    "centre (générique)": "Centre",
    "Entente républicaine et radicale du Sud-Ouest": "Centre",
    "Progrès et démocratie moderne": "Centre",
    "Mouvement démocrate": "Centre",
    "Centre national et républicain": "Centre",
    "Centre démocrate et réformateur": "Centre",
    "Centre réformateur": "Centre",
    "Centre des démocrates de progrès": "Centre",
    "Centre progrès et démocratie moderne": "Centre",
    "Centre Progrès et Démocratie moderne": "Centre",
    "Centre démocratie et progrès": "Centre",  
    "Cercle réformateur": "Centre",
    "Centriste réformateur": "Centre",
    "Union centriste": "Centre",
    "Union du centre": "Centre",
    "Union du Centre": "Centre",
    "Union des réformateurs": "Centre",
    "Union centriste républicaine": "Centre",
    "Union centriste libérale": "Centre",
    "Union des démocrates de progrès": "Centre",  
    "Démocrates de progrès": "Centre",
    "Cercle liberté et humanisme": "Centre",
    "Démocratie chrétienne de France": "Centre",
    "Mouvement républicain populaire": "Centre",  
    "Mouvement social chrétien": "Centre",
    "Mouvement social-chrétien": "Centre",
    "Groupe chrétien démocrate": "Centre",
    "Mouvement démocratie chrétienne": "Centre",
    "Mouvement européen": "Centre",
    "Comité français pour l'union paneuropéenne": "Centre",
    "Mouvement pour l'indépendance de l'Europe": "Centre",
    "Convergences et démocratie": "Centre",
    "Cercle républicain": "Centre",
    "Comité d'études pour un Nouveau Contrat Social": "Centre",
    "Cercle Tocqueville": "Centre",
    "Club Tocqueville": "Centre",
    "Mouvement des libéraux de progrès": "Centre",
    "Mouvement des démocrates libéraux": "Centre",
    "Génération sociale et libérale": "Centre",
    "Jeunes sociaux libéraux": "Centre",
    "Mouvement aixois des indépendants et libéraux": "Centre",
    "Union des libéraux indépendants": "Centre",
    "libéraux humanistes régionalistes": "Centre",
    "libéraux socialistes européens": "Centre",
    "Parti libéral": "Centre",
    "Parti libéral européen": "Centre",
    "droite libérale": "Centre", 
    "Fédération nationale des républicains indépendants": "Centre",  
    "Républicains indépendants giscardiens": "Centre",
    "Républicain indépendant social": "Centre",
    "giscardien": "Centre",
    "giscardiens": "Centre",
    "Jeunes giscardiens": "Centre",
    "Mouvement des jeunes giscardiens": "Centre",
    "Centre social français": "Centre",
    "Mouvement des enseignants libéraux": "Centre",
    "Centre chrétien populaire": "Centre",
    "Démocrates chrétiens Ve République": "Centre",
    "Centre droite-socialiste": "Centre",
    "Mouvement républicain social et démocratique": "Centre",
    "modérés": "Centre",
    "divers modérés": "Centre",
    "humaniste (générique)": "Centre",
    "Clubs Perspectives et réalités": "Centre",
    "Chrétiens pour un monde nouveau": "Centre",
    "Parti libéral pour la démocratie française": "Centre",
    "Entente libérale libertaire européenne": "Centre",
    "Union libérale de progrès": "Centre",
    "Club gardois des libéraux-sociaux": "Centre",
    "Clubs Démocratie nouvelle": "Centre",

    # =========================================================================
    # Droite gaulliste / conservatrice / libérale
    # =========================================================================
    "Rassemblement pour la République (RPR)": "Droite",
    "Union des démocrates pour la République (UDR)": "Droite",
    "Union des républicains de progrès (URP)": "Droite",
    "Centre national des indépendants et paysans (CNIP)": "Droite",
    "Union des gaullistes de progrès (UGP)": "Droite",
    "Union des jeunes pour le progrès (UJP)": "Droite",
    "Union démocratique du travail (UDT)": "Droite",
    "Alliance républicaine indépendante et libérale": "Droite",
    "Parti libéral de France": "Droite",
    "Centre des républicains libres": "Droite",
    "Fédération des républicains de progrès": "Droite",
    "gaulliste (générique)": "Droite",
    "divers droite": "Droite",
    "libéral": "Droite",
    "Union gaulliste pour la démocratie": "Droite",
    "Gaullistes de progrès": "Droite",
    "Progrès et Liberté": "Droite",
    "Mouvement des chrétiens pour la Ve République": "Droite",
    "Indépendants et paysans": "Droite",
    "Indépendants et Paysans": "Droite",
    "Jeunes Centre national des indépendants": "Droite",
    "Indépendant et Paysan": "Droite",
    "Fédération nationale des indépendants": "Droite",
    "Présence et action du gaullisme": "Droite", 
    "Comité d'action gaulliste": "Droite",
    "Rassemblement des gaullistes de progrès": "Droite",
    "Amicale gaulliste des Flandres": "Droite",
    "Mouvement national progrès et liberté": "Droite",
    "chiraquien": "Droite",
    "chiraquien indépendant": "Droite",
    "Cartel du respect de la vie": "Droite",  
    "Parti démocrate français": "Droite",
    "Parti démocrate": "Droite",
    "Parti agraire et paysan": "Droite",
    "droite": "Droite",
    "droite indépendante": "Droite",
    "droite autonome": "Droite",
    "droite démocratique": "Droite",
    "droite modérée": "Droite",
    "Union du peuple de France": "Droite",
    "Rassemblement des indépendants pour la majorité nouvelle": "Droite",
    "Rassemblement républicain": "Droite",
    "Union républicaine démocratique": "Droite",
    "Avenir et liberté": "Droite",
    "Union pour la liberté": "Droite",
    "Pour la France": "Droite",
    "Front uni de soutien au Sud-Vietnam": "Droite",             
    "Comité pour l'Europe des Patries": "Droite",                
    "Association pour la défense des institutions dans l'intérêt supérieur de la France pour l'épanouissement de la famille": "Droite",

    # =========================================================================
    # Extrême droite
    # =========================================================================
    "Front national (FN)": "Extrême droite",
    "Parti des forces nouvelles (PFN)": "Extrême droite",
    "Centre démocrate et républicain (CDR)": "Extrême droite",
    "Action royaliste": "Extrême droite",
    "Alsace d'abord": "Extrême droite",
    "Ordre nouveau": "Extrême droite",
    "Action française": "Extrême droite",
    "Cercle national des combattants": "Extrême droite",
    "Cercle national des agriculteurs": "Extrême droite",
    "Cercle national des femmes d'Europe": "Extrême droite",
    "Front national des rapatriés": "Extrême droite",
    "Mouvement travail patrie": "Extrême droite",
    "Comité central bonapartiste": "Extrême droite",
    "royaliste": "Extrême droite",
    "Union royaliste tourangelle": "Extrême droite",
    "Trop d'immigrés la France aux Français": "Extrême droite",
    "Trop d'immigrés la France aux français": "Extrême droite",
    "Front français": "Extrême droite",
    "Front Nord Anti-communiste": "Extrême droite",
    "Front Nord anti-communiste": "Extrême droite",
    "Fraternité française": "Extrême droite",
    "Parti nationaliste révolutionnaire": "Extrême droite",
    "Jeunesse nationaliste révolutionnaire": "Extrême droite",
    "Parti national démocrate": "Extrême droite",
    "Mouvement populiste français": "Extrême droite",
    "Oeuvre française": "Extrême droite",
    "Association nationale Pétain-Verdun": "Extrême droite",
    "Contre révolution en France": "Extrême droite",
    "Contre le vote des immigrés": "Extrême droite",
    "Rassemblement et coordination unitaire des rapatriés et spoliés d'Outre-Mer": "Extrême droite",
    "Groupe Union et défense": "Extrême droite",
    "Groupe union et défense": "Extrême droite",
    "Mouvement pour la France": "Extrême droite",

    # =========================================================================
    # Divers / Sans étiquette / régionalistes / mono-thématiques
    # =========================================================================
    "Sans étiquette": "Divers / Sans étiquette",
    "Parti breton": "Divers / Sans étiquette",
    "Lutte occitane": "Divers / Sans étiquette",                                  
    "Coordination rurale": "Divers / Sans étiquette",
    "Choisir la cause des femmes": "Divers / Sans étiquette",
    "Centre féminin d'études et d'information Femmes Avenir": "Divers / Sans étiquette",
    "Parti fédéraliste européen": "Divers / Sans étiquette",                    
    "Parti de la loi naturelle": "Divers / Sans étiquette",                     
    "Parti ouvrier européen": "Divers / Sans étiquette",                        
    "Parti du ras le bol": "Divers / Sans étiquette",                           
    "Chasse pêche nature traditions": "Divers / Sans étiquette",               
    "Rassemblement des usagers du service public et des contribuables": "Divers / Sans étiquette", 
    "Union des Français de bon sens": "Divers / Sans étiquette",                
    "Union des indépendants": "Divers / Sans étiquette",                        
    "Union travailliste": "Divers / Sans étiquette",                           
    "Alliance populaire": "Divers / Sans étiquette",                           
    "Mouvement universaliste": "Divers / Sans étiquette",                       
    "Parti pour la défense des animaux": "Divers / Sans étiquette",             
    "Nouvelle solidarité": "Divers / Sans étiquette",                           
    "Mouvement pour une nouvelle humanité": "Divers / Sans étiquette",          
    "républicain (générique)": "Divers / Sans étiquette",
    "Union pour une politique nouvelle": "Divers / Sans étiquette",
    "Refondations": "Divers / Sans étiquette",
    "Corsica Nazione": "Divers / Sans étiquette",
    "Accolta Naziunali Corsa": "Divers / Sans étiquette",
    "Mouvement pour l'autodétermination": "Divers / Sans étiquette",
    "Volèm viure al païs": "Divers / Sans étiquette",
    "Volem viure au pais": "Divers / Sans étiquette",
    "Mouvement normand": "Divers / Sans étiquette",
    "Unitat catalana": "Divers / Sans étiquette",
    "Esquerra catalana dels treballadors": "Divers / Sans étiquette",
    "Esquerra Catalana dels Treballadors": "Divers / Sans étiquette",
    "Esquerra republicana de Catalunya": "Divers / Sans étiquette",
    "Accio régionalista catalana": "Divers / Sans étiquette",
    "abertzale": "Divers / Sans étiquette",
    "Abertzaleen batasuna": "Divers / Sans étiquette",
    "Eusko alkartasuna": "Divers / Sans étiquette",
    "Euskal herriko alderdi sozialista": "Divers / Sans étiquette",
    "Mouvement démocratie alsacienne": "Divers / Sans étiquette",
    "Mouvement populaire alsacien": "Divers / Sans étiquette",
    "Comité d'étude et de liaison des intérêts bretons": "Divers / Sans étiquette",
    "Front indépendantiste": "Divers / Sans étiquette",
    "Rassemblement pour la Calédonie dans la République": "Divers / Sans étiquette",
    "Ia mana te nunaa": "Divers / Sans étiquette",
    "Faatereraa tiama o polinesia maohi": "Divers / Sans étiquette",
    "Taatiraa Polynesia Entente polynésienne": "Divers / Sans étiquette",
    "Tahoeraa Huiraatira": "Divers / Sans étiquette",
    "Tahoeraa huiraatira": "Divers / Sans étiquette",
    "Te taata Tahiti tiama": "Divers / Sans étiquette",
    "Mouvement social démocrate polynésien": "Divers / Sans étiquette",
    "Fédération pour une nouvelle société calédonienne": "Divers / Sans étiquette",
    "Périgord Expansion": "Divers / Sans étiquette",
    "Entente républicaine du Sud-Ouest": "Divers / Sans étiquette",
    "Rassemblement des radicaux de gauche de l'Essonne": "Divers / Sans étiquette",
    "Parti progressiste martiniquais": "Divers / Sans étiquette",  
    "Parti socialiste martiniquais": "Divers / Sans étiquette",
    "Mouvement socialiste guadeloupéen": "Divers / Sans étiquette",
    "Confédération générale des cadres": "Divers / Sans étiquette",
    "Confédération générale des petites et moyennes entreprises": "Divers / Sans étiquette",
    "Comité national des classes moyennes": "Divers / Sans étiquette",
    "Fédération des oeuvres laïques": "Divers / Sans étiquette",
    "Parti occitan": "Divers / Sans étiquette",
    "Emgann": "Divers / Sans étiquette",
    "Projet Alter Breton": "Divers / Sans étiquette",
    "Parti national breton": "Divers / Sans étiquette",
    "Mouvement pour l'organisation de la Bretagne": "Divers / Sans étiquette",
    "Convention générale de Bretagne": "Divers / Sans étiquette",
    "Association pour la culture bretonne": "Divers / Sans étiquette",
    "Union du peuple alsacien": "Divers / Sans étiquette",
    "Mouvement région Savoie": "Divers / Sans étiquette",
    "Union nationale régionaliste": "Divers / Sans étiquette",
    "Mouvement France régions": "Divers / Sans étiquette",
    "fédéraliste occitan et européen": "Divers / Sans étiquette",
    "mouvements occitanistes": "Divers / Sans étiquette",
    "Unitat d'Oc": "Divers / Sans étiquette",
    "Vida Nova": "Divers / Sans étiquette",
    "Collectif aquitano-occitan pour le non à Maastricht": "Divers / Sans étiquette",
    "Alliance libre européenne": "Divers / Sans étiquette",
    "Parti corse": "Divers / Sans étiquette",
    "Rassemblement démocratique pour l'avenir de la Corse": "Divers / Sans étiquette",
    "Parti mystico-rationaliste guadeloupéen": "Divers / Sans étiquette",
    "Parti socialiste guyanais": "Divers / Sans étiquette",
    "Comité socialiste calédonien": "Divers / Sans étiquette",
    "Parti socialiste calédonien": "Divers / Sans étiquette",
    "Libération Kanak socialiste": "Divers / Sans étiquette",
    "Front uni de libération kanak": "Divers / Sans étiquette",
    "Ia ora o Polynesia": "Divers / Sans étiquette",
    "Archipel Demain": "Divers / Sans étiquette",
    "Comité de liaison d'action locale et régionale": "Divers / Sans étiquette",
    "Comité d'action régionale": "Divers / Sans étiquette",
    "Comité de soutien à l'action du président de la République": "Divers / Sans étiquette",
    "Comité national de soutien à l'action du Président de la République": "Divers / Sans étiquette",
    "Partis de la majorité": "Divers / Sans étiquette",
    "Comité pour l'union du Loir-et-Cher": "Divers / Sans étiquette",
    "Comité défense droits constitutionnels": "Divers / Sans étiquette",
    "Comité d'études économiques": "Divers / Sans étiquette",
    "Comité national Antimafia": "Divers / Sans étiquette",
    "Comité d'action sociale des commerçants artisans petites entreprises et classes moyennes": "Divers / Sans étiquette",
    "Mouvement de défense de l'exploitation familiale": "Divers / Sans étiquette",
    "Mouvement rural de jeunesse chrétienne": "Divers / Sans étiquette",

    # Mono-thématiques / divers
    "Comités républicains": "Divers / Sans étiquette",
    "Clubs Ciel et Terre": "Divers / Sans étiquette",
    "Parti fédéraliste européen de France": "Divers / Sans étiquette",
    "Union des vieux de France": "Divers / Sans étiquette",
    "Association pour les libertés en Europe": "Divers / Sans étiquette",
    "France plus": "Divers / Sans étiquette",
    "Espoirs 93": "Divers / Sans étiquette",
    "espoirs 93": "Divers / Sans étiquette",
    "Moselle debout": "Divers / Sans étiquette",
    "Agir": "Divers / Sans étiquette",
    "Politique autrement": "Divers / Sans étiquette",
    "Initiatives 92": "Divers / Sans étiquette",
    "Dialogues-Cultures et identités": "Divers / Sans étiquette",
    "France cultures Islam et laïcité": "Divers / Sans étiquette",
    "Mouvement communautaire juif laïc": "Divers / Sans étiquette",
    "Des villages dans la ville": "Divers / Sans étiquette",
    "Association de défense de la famille en Yvelines": "Divers / Sans étiquette",
    "Union des écologistes": "Écologistes",
    "Rassemblement pour l'Europe fédérale": "Divers / Sans étiquette",
    "Front autonomiste de libération": "Divers / Sans étiquette",
    "Union pour la défense et illustration de Paris": "Divers / Sans étiquette",
    "Unio catalana per l'autonomia": "Divers / Sans étiquette",
    "Acampado per la recounquisto de provenço Païs d'Oc": "Divers / Sans étiquette",
    "Société pour l'étude la protection et l'aménagement de la nature dans le Sud-Ouest": "Écologistes",
    "Entente des démocrates bretons": "Divers / Sans étiquette",

    # --- Civiques / défense des droits / mono-thématiques ---
    "Ligue des droits de l'Homme": "Divers / Sans étiquette",
    "Ligue pour les libertés publiques": "Divers / Sans étiquette",
    "Sos contre la corruption": "Divers / Sans étiquette",
    "Défense des droits des citoyens": "Divers / Sans étiquette",
    "Mouvement de défense des libertés individuelles": "Divers / Sans étiquette",
    "Mouvement pour la justice et la liberté": "Divers / Sans étiquette",
    "Groupement Thémis": "Divers / Sans étiquette",
    "Groupement d'action judiciaire": "Divers / Sans étiquette",
    "Association pour le développement de la démocratie directe": "Divers / Sans étiquette",
    "Association de défense des victimes des procédures civiles pénales et administratives":"Divers / Sans étiquette",
    "Famille des victimes de St-Laurent-du-Pont": "Divers / Sans étiquette",
    "Planning familial": "Divers / Sans étiquette",
    "Mouvement pour la condition masculine": "Divers / Sans étiquette",
    "Femmes du 7e": "Divers / Sans étiquette",
    "Joinville au Féminin": "Divers / Sans étiquette",
    "Alfort au Féminin": "Divers / Sans étiquette",
    "Mouvement liberté légalité": "Divers / Sans étiquette",
    "Cultures et laïcité": "Divers / Sans étiquette",
    "Ligue nationale pour la liberté des vaccinations": "Divers / Sans étiquette",
    "Collectif des anciens engagés": "Divers / Sans étiquette",
    "Union nationale pour l'avenir de la médecine": "Divers / Sans étiquette",
    "Union nationale des travailleurs indépendants": "Divers / Sans étiquette",
    "Mouvement européen économique et social": "Divers / Sans étiquette",
    "Mouvement pour la défense énergique du bien-être des individus et des loisirs":
        "Divers / Sans étiquette",
    "Scouts européens d'Aquitaine": "Divers / Sans étiquette",
    "Académie Ausone": "Divers / Sans étiquette",
    "Académie des sciences humaines universelles": "Divers / Sans étiquette",
    "Neuropédagogues girondins": "Divers / Sans étiquette",
    "Amis de la Vallée de la Seine": "Divers / Sans étiquette",
    "Association Albertine-Sarrazin": "Divers / Sans étiquette",
    "Mouvement pour la géniocratie mondiale": "Divers / Sans étiquette",

    # --- Listes locales / micro-partis / divers ---
    "Union civique": "Divers / Sans étiquette",
    "Union de la jeunesse et des créateurs": "Divers / Sans étiquette",
    "Entente populaire indépendante": "Divers / Sans étiquette",
    "Groupe Réalisme et efficacité": "Divers / Sans étiquette",
    "Racines du futur": "Divers / Sans étiquette",
    "Parti humaniste": "Divers / Sans étiquette",                # parti humaniste international
    "Nouvelle génération": "Divers / Sans étiquette",
    "Véritable rassemblement de la population": "Divers / Sans étiquette",
    "Union pour Paris": "Divers / Sans étiquette",
    "Ensemble aujourd'hui": "Divers / Sans étiquette",
    "Egalité XXe": "Divers / Sans étiquette",
    "Parti de la politique inter-culturelle": "Divers / Sans étiquette",
    "Groupe opposition nouvelle génération": "Divers / Sans étiquette",
    "Mouvement rénovateur nouvelle génération": "Divers / Sans étiquette",
    "Parti d'en rire": "Divers / Sans étiquette",
    "Arc en ciel": "Divers / Sans étiquette",
    "Parti radical transnational": "Divers / Sans étiquette",    # parti italien (Pannella)
    "Parti libre": "Divers / Sans étiquette",
    "Blanc c'est exprimé": "Divers / Sans étiquette",
    "PROMESSE": "Divers / Sans étiquette",
    "confédération A29": "Divers / Sans étiquette",
    "Pourquoi pas Le Havre": "Divers / Sans étiquette",
    "Urbains gogologistes": "Divers / Sans étiquette",
    "Citoyens actifs": "Divers / Sans étiquette",
    "France 16": "Divers / Sans étiquette",
    "EDEN": "Divers / Sans étiquette",
    "AVEC": "Divers / Sans étiquette",
    "UTILE": "Divers / Sans étiquette",
    "Skyrock Quercy Rouergue": "Divers / Sans étiquette",
    "Virage 93": "Divers / Sans étiquette",
    "Rassemblement niçois": "Divers / Sans étiquette",
    "Responsabilités partagées": "Divers / Sans étiquette",
    "Convergence": "Divers / Sans étiquette",
    "Convergence des pays de Siagne": "Divers / Sans étiquette",
    "Parti des chômeurs et des mécontents": "Divers / Sans étiquette",
    "Unir et agir pour Gagny": "Divers / Sans étiquette",
    "Union départementale d'opposition 93": "Divers / Sans étiquette",
    "Union monsoise d'action municipale et familiale": "Divers / Sans étiquette",
    "Mouvement national des élus locaux": "Divers / Sans étiquette",
    "Mouvement d'action pour le renouveau du Sud de l'Essonne": "Divers / Sans étiquette",
    "Civisme et progrès": "Divers / Sans étiquette",
    "Action municipale et sociale": "Divers / Sans étiquette",
    "Vivre à Joinville": "Divers / Sans étiquette",
    "Groupement des associations de défense du Val d'Oise": "Divers / Sans étiquette",
    "Entraide sociale bordelaise": "Divers / Sans étiquette",
    "Association pour la défense de Bordeaux": "Divers / Sans étiquette",
    "Monde-Opinion": "Divers / Sans étiquette",
    "Comité de défense des droits du piéton": "Divers / Sans étiquette",
}

HEURISTIC_RULES = [
    # Extrême droite (très spécifique)
    ("Extrême droite", [
        r"\baction française\b", r"\boeuvre française\b", r"\bordre nouveau\b",
        r"\bnationaliste révolutionnaire\b", r"\bnationaliste-révolutionnaire\b",
        r"\bpétain\b", r"\bvichy\b", r"\bbonapartiste\b",
        r"\btrop d'immigrés\b", r"\banti-?communiste\b",
        r"\brapatriés\b", r"\bcercle national\b", r"\bfront national\b",
        r"\bmouvement populiste\b", r"travail patrie",
        r"\bnational(e)? d'action\b", r"\bcontre révolution\b",
        r"\bfraternité française\b", r"\bunité française\b",
    ]),
    # Extrême gauche
    ("Extrême gauche", [
        r"\btrotskyste\b", r"\bmarxiste-?léniniste\b",
        r"\bcommuniste révolutionnaire\b", r"\bcommuniste critique\b",
        r"\bligue communiste\b", r"\bligue trotskyste\b",
        r"\balternative révolutionnaire\b", r"\brévolutionnaires\b",
        r"\bcommuniste bolchevik\b",
        r"\bjeunesses communistes\b", r"\bjeunes communistes\b",
        r"\bsocialiste internationaliste\b",
        r"\bsocialistes des travailleurs\b",
    ]),
    # Régionalismes / DOM-TOM (avant les mots-clés génériques)
    ("Divers / Sans étiquette", [
        r"\bkanak\b", r"\bcalédoni", r"\bpolynési", r"\btahiti\b",
        r"\bhuiraatira\b", r"\bmartiniquais\b", r"\bguadeloupéen\b",
        r"\bguyanais\b", r"\bréunionnais\b",
        r"\bcorsi", r"\bnaziunal", r"\bcorse\b",
        r"\babertzale\b", r"\beusko\b", r"\beuskal\b", r"\bbasque\b",
        r"\bcatalan", r"\besquerra\b", r"\bunitat\b",
        r"\boccitan", r"viure al pais", r"viure au pais",
        r"\bvida nova\b", r"\bbreton", r"\bbreizh\b",
        r"\balsacien", r"\bsavoyard", r"\bnormand",
        r"\bia mana\b", r"\bia ora\b",
        r"\bautodétermination\b", r"\bindépendantiste\b",
        r"\brégionaliste\b", r"\brégionale\b",
    ]),
    # Écologistes
    ("Écologistes", [
        r"\bécolog", r"\becolog", r"\bverts?\b", r"\bvert\b",
        r"\bnature\b", r"\benvironnement\b", r"\bbiosphère\b",
        r"\banti-?pollution\b", r"\banti-?nucléaire\b", r"\banti-?bruit\b",
        r"\bsauvegarde des sites\b", r"\bantinucléaire\b",
        r"\bplanète\b", r"\bterre prochaine\b", r"\bamis de la terre\b",
    ]),
    # Gauche alternative (avant gauche/droite générique)
    ("Gauche alternative", [
        r"\bautogestion", r"\bautogestionnaire\b",
        r"\bnon[- ]violent", r"\blibertaire\b",
        r"\bgaullistes? de gauche\b", r"\bgauche gaulliste\b",
        r"\bgauche radicale\b", r"\bgauche progressiste\b",
        r"\bgauche libérale\b",
        r"\balternative démocratie\b", r"\brouge et vert\b",
        r"\bdésarmement\b", r"\bobjecteurs\b",
        r"\bparticipation\b",  # gaullisme social / Hamon
    ]),
    # Centre (avant droite/gauche générique)
    ("Centre", [
        r"\bcentriste\b", r"\bcentristes\b", r"\bcentre\b",
        r"\bréformateur\b", r"\bréformateurs\b",
        r"\bgiscardien", r"\blibéral", r"\blibéraux\b",
        r"\bsocial-?démocrate\b", r"\bdémocrate chrétien\b",
        r"\bdémocrates chrétiens\b", r"\bchrétien démocrate\b",
        r"\beuropéen\b", r"\bfédéraliste\b",
        r"\btocqueville\b", r"\bperspectives et réalités\b",
        r"\brépublicain populaire\b",  # MRP
    ]),
    # Droite (gaulliste / conservateur)
    ("Droite", [
        r"\bgaulliste\b", r"\bgaullistes\b",
        r"\bve république\b", r"\b5e république\b", r"\bcinquième république\b",
        r"\bchiraquien", r"\bindépendants? et paysans?\b",
        r"\bagraire\b", r"\bmodéré", r"\brépublicain indépendant\b",
        r"\brénovateur\b",
    ]),
    # Gauche communiste (après extrême gauche)
    ("Gauche communiste", [
        r"\bcommuniste\b", r"\bcommunistes\b",
        r"\bprogressiste martiniquais\b", r"\bprogressiste guadeloupéen\b",
    ]),
    # Gauche socialiste
    ("Gauche socialiste", [
        r"\bsocialiste\b", r"\bsocialistes\b", r"\bsocialisme\b",
        r"\bmitterrand\b", r"\binstitutions républicaines\b",
        r"\bgauche socialiste\b", r"\bdémocrate socialiste\b",
    ]),
    # Sans étiquette / divers
    ("Divers / Sans étiquette", [
        r"\bindépendant\b", r"\bindépendants?\b",
        r"\baucun parti\b", r"\bsans parti\b", r"\bapolitique\b",
        r"\bdivers\b", r"\bautre\b",
        r"\bligue des droits\b", r"\bplanning familial\b",
        r"\bfédération de l'éducation\b",
        r"\bsyndicat\b", r"\bconfédération\b",
        r"\bassociation\b", r"\bcomité\b", r"\bcollectif\b",
        r"\bclub\b", r"\bcercle\b",  # par défaut, faute de mieux
    ]),
]


def classify_by_keywords(label):
    """
    Classifie une chaîne brute en famille politique selon des indices lexicaux.
    Retourne le nom de la famille, ou None si aucun indice n'est trouvé.
    Utilisé en fallback pour les composantes pass-through non classées
    explicitement dans PARTY_FAMILY.
    """
    if not label or pd.isna(label):
        return None
    s = str(label).lower()
    for family, patterns in HEURISTIC_RULES:
        for pat in patterns:
            if re.search(pat, s):
                return family
    return None

# =============================================================================
# HELPERS
# =============================================================================

def normalize_value(value):
    """From the raw cell value, return a sorted list of canonical party names.
    Retourn [] for NaN or empty cell."""
    if pd.isna(value) or str(value).strip() == "":
        return []
    components = [c.strip() for c in str(value).split(";") if c.strip()]
    canonical = [COMPONENT_NORMALIZATION.get(c, c) for c in components]
    return sorted(set(canonical))


def get_families(parties):
    """From a list of canonical party names, return the list of political families.
    
    For each party, we first look in PARTY_FAMILY (exact match).
    If absent, we attempt heuristic classification on the raw name.
    """
    fams = set()
    for p in parties:
        if p in PARTY_FAMILY:
            fams.add(PARTY_FAMILY[p])
        else:
            heuristic = classify_by_keywords(p)
            fams.add(heuristic if heuristic else "Non classé")
    return [f for f in FAMILY_ORDER + ["Non classé"] if f in fams]


def get_main_family(families):
    """List of families → unique label. Inter-family coalition => 'A+B'."""
    if len(families) == 0:
        return "Aucun"
    if len(families) == 1:
        return families[0]
    return "+".join(families)


# =============================================================================
# THE FUNCTION
# =============================================================================

def add_party_columns(df, source_col="titulaire-soutien", top_n_parties=30):
    """
    Enrich `df` with 3 columns based on `source_col`:
      - df['soutien_partis']      : canonical parties list (list of str)
      - df['soutien_familles']    : political families list (list of str)
      - df['famille_principale']  : unique label (str)

    Returns (df_enrichi, party_dummies, family_dummies, diagnostic):
      - party_dummies   : DataFrame of dummies for the top_n_parties, for each row
      - family_dummies  : DataFrame of dummies for the 9 familles, for each row
      - diagnostic      : dict avec 'party_counts', 'unmapped', 'top_parties'
    """
    df = df.copy()
    df["soutien_partis"] = df[source_col].apply(normalize_value)
    df["soutien_familles"] = df["soutien_partis"].apply(get_families)
    df["famille_principale"] = df["soutien_familles"].apply(get_main_family)

    # Global count and top N parties
    party_counts = Counter()
    for parties in df["soutien_partis"]:
        party_counts.update(parties)
    top_parties = [p for p, _ in party_counts.most_common(top_n_parties)]

    # Dummies per party
    party_dummies = pd.DataFrame(
        {p: df["soutien_partis"].apply(lambda L, p=p: int(p in L)) for p in top_parties},
        index=df.index,
    )

    # Dummies per family
    family_dummies = pd.DataFrame(
        {f: df["soutien_familles"].apply(lambda L, f=f: int(f in L)) for f in FAMILY_ORDER},
        index=df.index,
    )

    # Diagnostic
    unmapped = Counter()       # didn't need to go through COMPONENT_NORMALIZATION (not a problem, just a pass-through)
    unclassified = Counter()   # canonical party not in PARTY_FAMILY (= REAL problem)
    for value in df[source_col].dropna():
        for c in str(value).split(";"):
            c = c.strip()
            if not c:
                continue
            canonical = COMPONENT_NORMALIZATION.get(c, c)
            if c not in COMPONENT_NORMALIZATION:
                unmapped[c] += 1
            if canonical not in PARTY_FAMILY:
                unclassified[canonical] += 1

    diagnostic = {
        "party_counts": party_counts,
        "unmapped": unmapped,
        "unclassified": unclassified,  
        "top_parties": top_parties,
    }
    return df, party_dummies, family_dummies, diagnostic


def print_party_diagnostic(diagnostic, df):
    pc = diagnostic["party_counts"]
    unclassified = diagnostic["unclassified"]

    print(f"Partis canoniques distincts : {len(pc)}")
    print("\nTop 10 partis :")
    for p, n in pc.most_common(10):
        print(f"  {n:>5}  {p}")

    print("\nFamilles principales (distribution) :")
    print(df["famille_principale"].value_counts().head(15).to_string())

    # HIGH PRIORITY : the REAL PROBLEMS = canonical parties that are not classified in a family.
    if unclassified:
        print(f"\n⚠️  Partis canoniques NON classés en famille politique ({len(unclassified)}) :")
        print("    (à ajouter dans PARTY_FAMILY)")
        for c, n in unclassified.most_common(20):
            print(f"  {n:>4}  {c}")
    else:
        print("\n✓ Tous les partis canoniques sont classés dans une famille.")