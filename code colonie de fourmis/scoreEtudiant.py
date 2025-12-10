import math
import numpy as np


# constant variables
inst1 = "data/inst1"
inst1_sol = "data/inst1.sol"
inst2 = "data/inst2"
inst2_sol = "data/inst2.sol"
inst3 = "data/inst3"
inst3_sol = "data/inst3.sol"

inst_concours = "data/inst_concours"


# load the instance in the file given in argument
def load_instance(inst_name):
    f=open(inst_name,"r")
    inst = {}

    # ignoring the beginning
    vals = list(filter(None,f.readline().split(" ")))
    while(not vals[0].isdigit()):
        vals = list(filter(None,f.readline().split(" ")))


    # reading the file
    inst[vals[0]] = {"x":float(vals[1]),"y":float(vals[2]),"wstart":float(vals[4]),"wend":float(vals[5])}
    while(len(vals)>0 and vals[0].isdigit() and int(vals[0])<999):
        inst[vals[0]] = {"x":float(vals[1]),"y":float(vals[2]),"wstart":float(vals[4]),"wend":float(vals[5])}
        vals = list(filter(None,f.readline().split(" ")))

    return inst



# load the solution and the claimed score in the file given in argument
def load_solution(sol_name):

    f=open(sol_name,"r")
    sol_list = f.readline().split()     # read the solution
    sol_val = f.readline()                  # read the score
    if (sol_val != ''):
        sol_val = int(sol_val)
    else:
        sol_val = None
    f.close()


    return sol_list,sol_val



# compute the distance between two points
def dist(instance, node1, node2):
    x1 = instance[node1]["x"]
    y1 = instance[node1]["y"]
    x2 = instance[node2]["x"]
    y2 = instance[node2]["y"]
    return math.floor(math.sqrt((x1-x2)*(x1-x2) + (y1-y2)*(y1-y2)))

# compute the distance matrix
# this can be time consuming, do it only once per instance
def compute_dist_mat(instance):
    mat_dist=np.zeros((len(instance)+1,len(instance)+1))
    for i in instance:
        for j in instance:
            mat_dist[int(i),int(j)] = dist(instance,i,j)

    #update in order to respect inequality triangle
    for i in instance:
        for j in instance:
            for k in instance:
                if ( mat_dist[int(i),int(j)] > mat_dist[int(i),int(k)] + mat_dist[int(k),int(j)] ):
                    mat_dist[int(i),int(j)] = mat_dist[int(i),int(k)] + mat_dist[int(k),int(j)] 
    return mat_dist

# compute the score using the distance matrix
def compute_score_with_mat(instance,sol_list,dist_mat):
    distance = 0
    duree = 0
    nb_violation = 0
    for i in range(len(sol_list)-1):
        distance += dist_mat[int(sol_list[i]), int(sol_list[i+1])]
        duree += dist_mat[int(sol_list[i]), int(sol_list[i+1])]
        next_start = instance[sol_list[i+1]]["wstart"]
        end_window = instance[sol_list[i+1]]["wend"]
        if (duree < next_start):
            duree = next_start
        if (duree > end_window):
            nb_violation+=1


    distance += dist_mat[int(sol_list[-1]), int(sol_list[0])]
    duree += dist_mat[int(sol_list[-1]), int(sol_list[0])]
    next_start = instance[sol_list[0]]["wstart"]
    end_window = instance[sol_list[0]]["wend"]
    if (duree < next_start):
        duree = next_start
    if (duree > end_window):
        nb_violation+=1

    # can be used for debug
    # print(distance)
    # print(duree)
    print(nb_violation)
    return distance


# compute the score from the instance and the solution given in argument
def compute_score(instance,sol_list):
    dist_mat = compute_dist_mat(instance)
    score = compute_score_with_mat(instance,sol_list,dist_mat)
    return score


# compute the score of an instance for a given solution
# instance and solution are given by the name of their file
# if a score is given in the solution file, check if it matches the computed score
def verif_sol_inst(inst_name,sol_name):

    instance = load_instance(inst_name)
    sol_list, sol_score_f = load_solution(sol_name)
    sol_score_c = compute_score(instance,sol_list)

    if sol_list[0] != "1":
        print("Wrong solution: does not start from city 1")


    for node in instance:
        if (not node in sol_list):
            print("Wrong solution: a city is not visited: " + node)

    if (sol_score_f is None):
        print("no score in file")
        print("computed score:" + str(sol_score_c))
        return

    if (sol_score_f != sol_score_c):
        print("Score in file different from computed score:")
        print(str(sol_score_f) + " != " + str(sol_score_c))
        return


    print("Score in file corresponds to computed score: " + str(sol_score_c))
    return

#transforme chaque instance en liste de coordonnés sans prendre en compte les fenêtres de temps
def instance_to_coords_and_time_list(inst_name):
    f=open(inst_name,"r")

    inst = {}
    time = {}
    # ignoring the beginning
    vals = list(filter(None,f.readline().split(" ")))
    while(not vals[0].isdigit()):
        vals = list(filter(None,f.readline().split(" ")))


    # reading the file
    inst[int(vals[0])] = float(vals[1]), float(vals[2])
    time[int(vals[0])] = float(vals[4]), float(vals[5])
    while(len(vals)>0 and vals[0].isdigit() and int(vals[0])<999):
        inst[int(vals[0])] = float(vals[1]),float(vals[2])
        time[int(vals[0])] = float(vals[4]), float(vals[5])
        vals = list(filter(None,f.readline().split(" ")))

    return list(inst.values()), list(time.values())






# the main function, uncomment the part you want to execute
def main():

    ### functions can be used to check that the instance or the solution is read correctly
    #print(load_solution(inst1_sol))
    #print(load_instance(inst1))
    print(instance_to_coords_and_time_list(inst1)[0])
    print(instance_to_coords_and_time_list(inst1)[1])
    #print(load_instance(inst_concours))
    

    ### functions used to compute the score of a solution
    #verif_sol_inst(inst1,inst1_sol)
    #verif_sol_inst(inst2,inst2_sol)
    #verif_sol_inst(inst3,inst3_sol) # computing the distance matrix takes some time (10s-20s) (you only need to do it once)
    
    #verif_sol_inst(inst_concours,inst_concours_sol)  # works ony once you have created a solution file
    

if __name__ == "__main__":
    main()


