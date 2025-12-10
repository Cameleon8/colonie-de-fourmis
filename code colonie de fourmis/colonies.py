import numpy as np
import matplotlib.pyplot as plt
from scoreEtudiant import instance_to_coords

def euclidean_distance(coord1, coord2):
    """
    Calcule la distance euclidienne entre deux points 2D.
    Ce sera le poids de l'arête d(si, si+1).
    """
    return np.linalg.norm(np.array(coord1) - np.array(coord2))

def calculate_tour_length(tour, dist_matrix):
    """
    Calcule la longueur totale du tour (chemin) d'une fourmi.
    Le coût de la solution L_k.
    """
    length = 0
    num_nodes = len(tour)
    for i in range(num_nodes):
        # Le tour est un cycle : relier le dernier noeud au premier
        node1 = tour[i]
        node2 = tour[(i + 1) % num_nodes]
        length += dist_matrix[node1, node2]
    return length

def calculate_prob(i, unvisited_nodes, tau_matrix, eta_matrix, alpha, beta):
    """
    Calcule la probabilité p_ij d'aller du noeud i au noeud j.
    La formule est : p_ij = (tau_ij^alpha * eta_ij^beta) / sum(tau_il^alpha * eta_il^beta).
    """
    # Numerator (numérateur)
    # Les indices j sont les noeuds non visités
    pheromones_pow = tau_matrix[i, unvisited_nodes] ** alpha
    desirability_pow = eta_matrix[i, unvisited_nodes] ** beta
    numerator = pheromones_pow * desirability_pow

    # Denominator (dénominateur)
    denominator = np.sum(numerator)

    # Si le dénominateur est 0 (par exemple, au début avec tau=0 et des erreurs d'initialisation, 
    # ou si alpha=0 et beta=0), on donne une probabilité uniforme aux noeuds non visités.
    if denominator == 0:
        return np.ones_like(unvisited_nodes) / len(unvisited_nodes)
    
    # Probabilités
    probabilities = numerator / denominator
    return probabilities

def construct_tour(start_node, num_nodes, tau_matrix, eta_matrix, alpha, beta):
    """
    Une fourmi construit un tour complet en utilisant la règle probabiliste.
    """
    # Initialisation
    tour = [start_node]
    unvisited_nodes = set(range(num_nodes))
    unvisited_nodes.remove(start_node)
    current_node = start_node

    # Construction du tour jusqu'à ce que tous les noeuds soient visités
    while unvisited_nodes:
        # Calculer les probabilités pour les noeuds non visités
        unvisited_list = list(unvisited_nodes)
        probabilities = calculate_prob(current_node, unvisited_list, tau_matrix, eta_matrix, alpha, beta)

        # Sélectionner le noeud suivant de manière probabiliste
        next_node = np.random.choice(unvisited_list, 1, p=probabilities)[0]

        # Mettre à jour l'état de la fourmi
        tour.append(next_node)
        unvisited_nodes.remove(next_node)
        current_node = next_node
        
    return tour

def update_pheromones(tau_matrix, all_tours, dist_matrix, rho):
    """
    Mise à jour des phéromones après l'achèvement des tours par toutes les fourmis.
    La formule est : tau_ij = (1-rho) * tau_ij + sum(Delta_tau_ij^k).
    """
    num_nodes = tau_matrix.shape[0]
    num_ants = len(all_tours)
    
    # 1. Évaporation des phéromones
    tau_matrix = (1.0 - rho) * tau_matrix 

    # 2. Dépôt de nouvelles phéromones
    for k in range(num_ants):
        tour = all_tours[k]
        L_k = calculate_tour_length(tour, dist_matrix) # L_k: le coût de la solution 

        # Si le coût est acceptable (non nul), déposer des phéromones
        if L_k > 0:
            delta_tau_k = 1.0 / L_k # delta_tau_ij^k = 1/L_k si l'arête (i,j) est utilisée 
            
            for i in range(num_nodes):
                node1 = tour[i]
                node2 = tour[(i + 1) % num_nodes]
                
                # Le problème est non orienté (TSP), donc le dépôt est symétrique (tau_ij = tau_ji)
                tau_matrix[node1, node2] += delta_tau_k
                tau_matrix[node2, node1] += delta_tau_k
    
    return tau_matrix

def ACO_TSP(coords, start_node, num_ants, num_iterations, alpha, beta, rho):
    """
    Algorithme de Colonie de Fourmis (ACO) pour le Problème du Voyageur de Commerce (TSP).
    Inputs:
    - coords (list of tuples): Liste des coordonnées 2D des noeuds.
    -start_node (int) : le noeuds dans lequel le voyageur démarre.
    - num_ants (int): Nombre de fourmis (nb_a) par itération.
    - num_iterations (int): Nombre de générations/itérations.
    - alpha (float): Paramètre d'importance des phéromones.
    - beta (float): Paramètre d'importance de l'heuristique (désirabilité).
    - rho (float): Taux d'évaporation des phéromones.
    """
    num_nodes = len(coords)
    if num_nodes < 2:
        return [], 0.0

    # 1. Initialisation de la matrice des distances
    dist_matrix = np.zeros((num_nodes, num_nodes))
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                dist_matrix[i, j] = euclidean_distance(coords[i], coords[j])
    
    # 2. Initialisation de la matrice heuristique (désirabilité) eta_ij = 1 / d_ij
    # Le poids de l'arête est la distance. La désirabilité est l'inverse pour minimiser.
    with np.errstate(divide='ignore'):
        eta_matrix = 1.0 / dist_matrix
        eta_matrix[np.isinf(eta_matrix)] = 0.0 # Remplace inf par 0 (pour la diagonale)

    # 3. Initialisation de la matrice des phéromones (tau_ij)
    tau0 = 1.0 / (num_nodes * np.mean(dist_matrix[dist_matrix > 0])) # Heuristique pour l'initialisation
    tau_matrix = np.ones((num_nodes, num_nodes)) * tau0
    #tau_matrix = np.zeros((num_nodes, num_nodes))
    
    best_tour = None
    best_tour_length = float('inf')

    # Boucle principale de l'algorithme ACO
    #print(f"Lancement de l'algorithme ACO pour {num_iterations} itérations avec {num_ants} fourmis...")
    Score = []
    for iteration in range(num_iterations):
        # 4. Construction des solutions par chaque fourmi
        all_tours = []
        for ant in range(num_ants):
            tour = construct_tour(start_node, num_nodes, tau_matrix, eta_matrix, alpha, beta)
            all_tours.append(tour)

            # Évaluation et mise à jour de la meilleure solution
            length = calculate_tour_length(tour, dist_matrix)
            if length < best_tour_length:
                best_tour_length = length
                best_tour = tour
        
        # 5. Mise à jour des phéromones
        tau_matrix = update_pheromones(tau_matrix, all_tours, dist_matrix, rho)

        # Affichage du progrès
        Score.append(best_tour_length)
   #     if (iteration + 1) % 10 == 0 or iteration == 0:
    #        print(f"Itération {iteration + 1}/{num_iterations}: Meilleure longueur de tour actuelle: {best_tour_length:.2f}")
    #plt.figure()
    #plt.plot(range(num_iterations), Score)
    #plt.show()
    return best_tour, best_tour_length, Score

inst1 = "data/inst1"
inst2 = "data/inst2"
inst3 = "data/inst3"
cities_coords = instance_to_coords(inst3)

# Paramètres de l'algorithme (à ajuster pour une performance optimale)
num_ants = 10     # Nombre de fourmis (nb_a) 
num_iterations = 100 # Nombre d'itérations
alpha = 1.0          # Importance des phéromones 
beta = 1.0           # Importance de l'heuristique (distance) 
rho = 0.5           # Taux d'évaporation 
start_node = 0       # le tour se fait depuis le premier noeuds

# Exécution de l'algorithme
#best_tour, best_length = ACO_TSP(cities_coords, start_node, num_ants, num_iterations, alpha, beta, rho)

#print("\n--- Résultat Final ---")
# Le tour est une liste d'indices de noeuds (0 à n-1)
#print(f"Meilleur tour trouvé (indices des noeuds): {best_tour}")
#print(f"Longueur du meilleur tour: {best_length:.2f}")

def courbe_score_intervalle_confiance(num_run, coords, start_node, num_ants, num_iterations, alpha, beta, rho):
    Score_mean = [0 for i in range(num_iterations)]
    Score_sigma = [0 for i in range(num_iterations)]
    for l in range(num_run):
        score = ACO_TSP(coords, start_node, num_ants, num_iterations, alpha, beta, rho)[2]
        for i in range(num_iterations):
            Score_mean[i] += score[i]
    Score_Mean = [(1/num_run) * Score_mean[i] for i in range(num_iterations)]
    for l in range(num_run):
        score = ACO_TSP(coords, start_node, num_ants, num_iterations, alpha, beta, rho)[2]
        for i in range(num_iterations):
            Score_sigma[i] += (score[i] - Score_Mean[i])**2
    Score_Sigma = [(x/num_run)**0.5 for x in Score_sigma]
    Score_inf = [Score_Mean[i] - Score_Sigma[i]/(num_run**0.5) for i in range(num_iterations)]
    Score_sup = [Score_Mean[i] + Score_Sigma[i]/(num_run**0.5) for i in range(num_iterations)]
    # Dessiner la courbe de tendance principale
    plt.plot(range(num_iterations), Score_Mean, color='darkblue', linewidth=2, label='Score')

    # Utiliser fill_between pour dessiner l'intervalle de confiance
    plt.fill_between(range(num_iterations), Score_inf, Score_sup, color='skyblue', alpha=0.5)


    # Ajouter des étiquettes et un titre
    plt.title("Evolution du score selon le nombre d'itération (inst3) ")
    plt.xlabel("nombre d'itération")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
        

num_run = 50 #nombre de test pour générer des intervalles de confiance

courbe_score_intervalle_confiance(num_run, cities_coords, start_node, num_ants, num_iterations, alpha, beta, rho )


    