
import pandas as pd
from pulp import *

# This function creates the attendance vector, its size is equal to the number of workers in a team
# In this vector string each 0 represent the ith worker being absent and vice versa for 1.      
def get_att_vector(team, attendance_matrix, row):
    if team == 'A':
        team_att = attendance_matrix.iloc[row,0:15].tolist()
    elif team == 'B':
        team_att = attendance_matrix.iloc[row,15:32].tolist()
    elif team == 'C':
        team_att = attendance_matrix.iloc[row,32:49].tolist()
    elif team == 'D':
        team_att = attendance_matrix.iloc[row,49:60].tolist()
    elif team == 'E':
        team_att = attendance_matrix.iloc[row,60:71].tolist()
    elif team == 'F':
        team_att = attendance_matrix.iloc[row,71:86].tolist()
    elif team == 'G':
        team_att = attendance_matrix.iloc[row,86:100].tolist()
    strings = [str(integer) for integer in team_att]
    team_att = "".join(strings)
    return team_att

#This function uses attendance vector to update skills matrix by removing the skills of absent workers
def generate_today_matrix(team, skills_matrix, team_att, emergency):
    if team == 'A':
        skills_matrix_N = skills_matrix.iloc[0:13,0:15]
    if team == 'B':
        skills_matrix_N = skills_matrix.iloc[13:26,15:32]
    if team == 'C':
        skills_matrix_N = skills_matrix.iloc[26:38,32:49]
    if team == 'D':
        skills_matrix_N = skills_matrix.iloc[38:47,49:60]
    if team == 'E':
        skills_matrix_N = skills_matrix.iloc[47:56,60:71]
    if team == 'F':
        skills_matrix_N = skills_matrix.iloc[56:67,71:86]
    if team == 'G':
        skills_matrix_N = skills_matrix.iloc[67:78,86:100]
        
    if emergency == 1:
        skills_matrix_N = skills_matrix_N[:-1]
    
    columns = skills_matrix_N.columns
    skills_matrix_today = skills_matrix_N
    skills_matrix_today = skills_matrix_today.applymap(lambda i: 0.5 if i == 0 else 1)
    for i in range(len(team_att)):
        if team_att[i] == '0':
            absent_worker = columns[i]
            skills_matrix_today = skills_matrix_today.drop(absent_worker,1)
            #skills_matrix_today.loc[:,absent_worker] = 0.0
    #skills_matrix_today.index = skills_matrix_today.index.map(str)
    return skills_matrix_today

skills_matrix = pd.concat(pd.read_excel('all_teams_skills.xlsx',sheet_name = None),ignore_index = True, sort=False)
skills_matrix.index = skills_matrix.iloc[:,0]
skills_matrix = skills_matrix.drop('Unnamed: 0', axis=1)
skills_matrix = skills_matrix.T


attendance_matrix = pd.concat(pd.read_excel('all_team_att.xlsx',sheet_name = None),ignore_index = True, sort=False)
attendance_matrix=attendance_matrix.T
new_header = attendance_matrix.iloc[0]
attendance_matrix = attendance_matrix[1:]
attendance_matrix.columns = new_header
attendance_matrix['count'] = attendance_matrix.sum(axis = 1)
    
no_of_days = attendance_matrix.shape[0]
cost = []
overall_cost = 0
# Emergency scenario 1: when the number of workers present is less than number of workstations
for day in range(no_of_days):
    if attendance_matrix.iloc[day,-1] < 78:
        f1 = 1
    else:
        f1 = 0
        
    team_att = get_att_vector('A',attendance_matrix, day)
    skills_matrix_today = generate_today_matrix('A', skills_matrix, team_att, f1)
    problem = LpProblem("Minimize_assignment",LpMinimize)
    terminals = skills_matrix_today.index.to_list()
    workers = skills_matrix_today.columns.to_list()
    matrix = LpVariable.dicts('Assign',(terminals,workers),cat='Binary')
    problem+= lpSum(matrix[i][j] for i in terminals for j in workers)
    for i in terminals:
        problem+= lpSum([matrix[i][j]*skills_matrix_today.loc[i,j] for j in workers]) == 1
    for j in workers:
        problem+= lpSum([matrix[i][j] for i in terminals]) <= 1
    problem.solve()
    assigned = team_att.count('1')
    final_status = LpStatus[problem.status]
    if final_status != 'Optimal':
        workers_left_A = 0
        workers_required_A = value(problem.objective) - assigned
        workers_assigned_A = assigned
    else:
        workers_assigned_A = value(problem.objective)
        workers_left_A = assigned - workers_assigned_A
        workers_required_A = 0
    #team_A_ws, team_A_ava = assign_legacy_workers('A', team_A_ws, team_att, skills_matrix_today)
    #workers_solo_A, workers_left_A = workers_left(team_A_ws, team_A_ava, team_att)
    
    team_att = get_att_vector('B',attendance_matrix, day)
    skills_matrix_today = generate_today_matrix('B', skills_matrix, team_att, f1)
    problem = LpProblem("Minimize_assignment",LpMinimize)
    terminals = skills_matrix_today.index.to_list()
    workers = skills_matrix_today.columns.to_list()
    matrix = LpVariable.dicts('Assign',(terminals,workers),cat='Binary')
    problem+= lpSum(matrix[i][j] for i in terminals for j in workers)
    for i in terminals:
        problem+= lpSum([matrix[i][j]*skills_matrix_today.loc[i,j] for j in workers]) == 1
    for j in workers:
        problem+= lpSum([matrix[i][j] for i in terminals]) <= 1
    problem.solve()
    final_status = LpStatus[problem.status]
    assigned = team_att.count('1')
    if final_status != 'Optimal':
        workers_left_B = 0
        workers_required_B = value(problem.objective) - assigned
        workers_assigned_B = assigned
    else:
        workers_assigned_B = value(problem.objective)
        workers_left_B = assigned - workers_assigned_B
        workers_required_B = 0
    #team_B_ws, team_B_ava = assign_legacy_workers('B', team_B_ws, team_att, skills_matrix_today)
    #workers_solo_B, workers_left_B = workers_left(team_B_ws, team_B_ava, team_att)
    
    team_att = get_att_vector('C',attendance_matrix, day)
    skills_matrix_today = generate_today_matrix('C', skills_matrix, team_att, f1)
    problem = LpProblem("Minimize_assignment",LpMinimize)
    terminals = skills_matrix_today.index.to_list()
    workers = skills_matrix_today.columns.to_list()
    matrix = LpVariable.dicts('Assign',(terminals,workers),cat='Binary')
    problem+= lpSum(matrix[i][j] for i in terminals for j in workers)
    for i in terminals:
        problem+= lpSum([matrix[i][j]*skills_matrix_today.loc[i,j] for j in workers]) == 1
    for j in workers:
        problem+= lpSum([matrix[i][j] for i in terminals]) <= 1
    problem.solve()
    final_status = LpStatus[problem.status]
    assigned = team_att.count('1')
    if final_status != 'Optimal':
        workers_left_C = 0
        assigned = team_att.count('1')
        workers_required_C = value(problem.objective) - assigned
        workers_assigned_C = assigned
    else:
        workers_assigned_C = value(problem.objective)
        workers_left_C = assigned = workers_assigned_C
        workers_required_C = 0
        
    #team_C_ws, team_C_ava = assign_legacy_workers('C', team_C_ws, team_att, skills_matrix_today)
    #workers_solo_C, workers_left_C = workers_left(team_C_ws, team_C_ava, team_att)
    
    team_att = get_att_vector('D',attendance_matrix, day)
    skills_matrix_today = generate_today_matrix('D', skills_matrix, team_att,f1)
    problem = LpProblem("Minimize_assignment",LpMinimize)
    terminals = skills_matrix_today.index.to_list()
    workers = skills_matrix_today.columns.to_list()
    matrix = LpVariable.dicts('Assign',(terminals,workers),cat='Binary')
    problem+= lpSum(matrix[i][j] for i in terminals for j in workers)
    for i in terminals:
        problem+= lpSum([matrix[i][j]*skills_matrix_today.loc[i,j] for j in workers]) == 1
    for j in workers:
        problem+= lpSum([matrix[i][j] for i in terminals]) <= 1
    problem.solve()
    final_status = LpStatus[problem.status]
    assigned = team_att.count('1')
    if final_status != 'Optimal':
        workers_left_D = 0
        assigned = team_att.count('1')
        workers_required_D = value(problem.objective) - assigned
        workers_assigned_D = assigned
    else:
        workers_assigned_D = value(problem.objective)
        workers_left_D = assigned = workers_assigned_D
        workers_required_D = 0
    #team_D_ws, team_D_ava = assign_legacy_workers('D', team_D_ws, team_att, skills_matrix_today)
    #workers_solo_D, workers_left_D = workers_left(team_D_ws, team_D_ava, team_att)
    
    team_att = get_att_vector('E',attendance_matrix, day)
    skills_matrix_today = generate_today_matrix('E', skills_matrix, team_att, f1)
    problem = LpProblem("Minimize_assignment",LpMinimize)
    terminals = skills_matrix_today.index.to_list()
    workers = skills_matrix_today.columns.to_list()
    matrix = LpVariable.dicts('Assign',(terminals,workers),cat='Binary')
    problem+= lpSum(matrix[i][j] for i in terminals for j in workers)
    for i in terminals:
        problem+= lpSum([matrix[i][j]*skills_matrix_today.loc[i,j] for j in workers]) == 1
    for j in workers:
        problem+= lpSum([matrix[i][j] for i in terminals]) <= 1
    problem.solve()
    final_status = LpStatus[problem.status]
    assigned = team_att.count('1')
    if final_status != 'Optimal':
        workers_left_E = 0
        assigned = team_att.count('1')
        workers_required_E = value(problem.objective) - assigned
        workers_assigned_E = assigned
    else:
        workers_assigned_E = value(problem.objective)
        workers_left_E = assigned - workers_assigned_E
        workers_required_E = 0
       
    #team_E_ws, team_E_ava = assign_legacy_workers('E', team_E_ws, team_att, skills_matrix_today)
    #workers_solo_E, workers_left_E = workers_left(team_E_ws, team_E_ava, team_att)
    
    team_att = get_att_vector('F',attendance_matrix, day)
    skills_matrix_today = generate_today_matrix('F', skills_matrix, team_att, f1)
    problem = LpProblem("Minimize_assignment",LpMinimize)
    terminals = skills_matrix_today.index.to_list()
    workers = skills_matrix_today.columns.to_list()
    matrix = LpVariable.dicts('Assign',(terminals,workers),cat='Binary')
    problem+= lpSum(matrix[i][j] for i in terminals for j in workers)
    for i in terminals:
        problem+= lpSum([matrix[i][j]*skills_matrix_today.loc[i,j] for j in workers]) == 1
    for j in workers:
        problem+= lpSum([matrix[i][j] for i in terminals]) <= 1
    problem.solve()
    final_status = LpStatus[problem.status]
    assigned = team_att.count('1')
    if final_status != 'Optimal':
        workers_left_F = 0
        assigned = team_att.count('1')
        workers_required_F = value(problem.objective) - assigned
        workers_assigned_F = assigned
    else:
        workers_assigned_F = value(problem.objective)
        workers_left_F = assigned - workers_assigned_F
        workers_required_F = 0
    #team_F_ws, team_F_ava = assign_legacy_workers('F', team_F_ws, team_att, skills_matrix_today)
    #workers_solo_F, workers_left_F = workers_left(team_F_ws, team_F_ava, team_att)
    
    team_att = get_att_vector('G',attendance_matrix, day)
    skills_matrix_today = generate_today_matrix('G', skills_matrix, team_att, f1)
    problem = LpProblem("Minimize_assignment",LpMinimize)
    terminals = skills_matrix_today.index.to_list()
    workers = skills_matrix_today.columns.to_list()
    matrix = LpVariable.dicts('Assign',(terminals,workers),cat='Binary')
    problem+= lpSum(matrix[i][j] for i in terminals for j in workers)
    for i in terminals:
        problem+= lpSum([matrix[i][j]*skills_matrix_today.loc[i,j] for j in workers]) == 1
    for j in workers:
        problem+= lpSum([matrix[i][j] for i in terminals]) <= 1
    problem.solve()
    final_status = LpStatus[problem.status]
    assigned = team_att.count('1')
    if final_status != 'Optimal':
        workers_left_G = 0
        assigned = team_att.count('1')
        workers_required_G = value(problem.objective) - assigned
        workers_assigned_G = assigned
    else:
        workers_assigned_G = value(problem.objective)
        workers_left_G = assigned - workers_assigned_G
        workers_required_G = 0
    #team_G_ws, team_G_ava = assign_legacy_workers('G', team_G_ws, team_att, skills_matrix_today)
    #workers_solo_G, workers_left_G = workers_left(team_G_ws, team_G_ava, team_att)
    
    workers_left_all = workers_left_A + workers_left_B + workers_left_C + workers_left_D + workers_left_E + workers_left_F + workers_left_G
    workers_required_all = workers_required_A + workers_required_B + workers_required_C + workers_required_D + workers_required_E + workers_required_F + workers_required_G
    if workers_left_all < workers_required_all:
        f1 = 1
        
        team_att = get_att_vector('A',attendance_matrix, day)
        skills_matrix_today = generate_today_matrix('A', skills_matrix, team_att, f1)
        problem = LpProblem("Minimize_assignment",LpMinimize)
        terminals = skills_matrix_today.index.to_list()
        workers = skills_matrix_today.columns.to_list()
        matrix = LpVariable.dicts('Assign',(terminals,workers),cat='Binary')
        problem+= lpSum(matrix[i][j] for i in terminals for j in workers)
        for i in terminals:
            problem+= lpSum([matrix[i][j]*skills_matrix_today.loc[i,j] for j in workers]) == 1
        for j in workers:
            problem+= lpSum([matrix[i][j] for i in terminals]) <= 1
        problem.solve()
        final_status = LpStatus[problem.status]
        assigned = team_att.count('1')
        if final_status != 'Optimal':
            workers_left_A = 0
            workers_required_A = value(problem.objective) - assigned
            workers_assigned_A = assigned
        else:
            workers_assigned_A = value(problem.objective)
            workers_left_A = assigned - workers_assigned_A
            workers_required_A = 0
        
        
    #team_A_ws, team_A_ava = assign_legacy_workers('A', team_A_ws, team_att, skills_matrix_today)
    #workers_solo_A, workers_left_A = workers_left(team_A_ws, team_A_ava, team_att)
    
        team_att = get_att_vector('B',attendance_matrix, day)
        skills_matrix_today = generate_today_matrix('B', skills_matrix, team_att, f1)
        problem = LpProblem("Minimize_assignment",LpMinimize)
        terminals = skills_matrix_today.index.to_list()
        workers = skills_matrix_today.columns.to_list()
        matrix = LpVariable.dicts('Assign',(terminals,workers),cat='Binary')
        problem+= lpSum(matrix[i][j] for i in terminals for j in workers)
        for i in terminals:
            problem+= lpSum([matrix[i][j]*skills_matrix_today.loc[i,j] for j in workers]) == 1
        for j in workers:
            problem+= lpSum([matrix[i][j] for i in terminals]) <= 1
        problem.solve()
        final_status = LpStatus[problem.status]
        assigned = team_att.count('1')
        if final_status != 'Optimal':
            workers_left_B = 0
            workers_required_B = value(problem.objective) - assigned
            workers_assigned_B = assigned
        else:
            workers_assigned_B = value(problem.objective)
            workers_left_B = assigned - workers_assigned_B
            workers_required_B = 0
            
    #team_B_ws, team_B_ava = assign_legacy_workers('B', team_B_ws, team_att, skills_matrix_today)
    #workers_solo_B, workers_left_B = workers_left(team_B_ws, team_B_ava, team_att)
    
        team_att = get_att_vector('C',attendance_matrix, day)
        skills_matrix_today = generate_today_matrix('C', skills_matrix, team_att, f1)
        problem = LpProblem("Minimize_assignment",LpMinimize)
        terminals = skills_matrix_today.index.to_list()
        workers = skills_matrix_today.columns.to_list()
        matrix = LpVariable.dicts('Assign',(terminals,workers),cat='Binary')
        problem+= lpSum(matrix[i][j] for i in terminals for j in workers)
        for i in terminals:
            problem+= lpSum([matrix[i][j]*skills_matrix_today.loc[i,j] for j in workers]) == 1
        for j in workers:
            problem+= lpSum([matrix[i][j] for i in terminals]) <= 1
        problem.solve()
        final_status = LpStatus[problem.status]
        assigned = team_att.count('1')
        if final_status != 'Optimal':
            workers_left_C = 0
            workers_required_C = value(problem.objective) - assigned
            workers_assigned_C = assigned
        else:
            workers_assigned_C = value(problem.objective)
            workers_left_C = assigned - workers_assigned_C
            workers_required_C = 0
    #team_C_ws, team_C_ava = assign_legacy_workers('C', team_C_ws, team_att, skills_matrix_today)
    #workers_solo_C, workers_left_C = workers_left(team_C_ws, team_C_ava, team_att)
    
        team_att = get_att_vector('D',attendance_matrix, day)
        skills_matrix_today = generate_today_matrix('D', skills_matrix, team_att,f1)
        problem = LpProblem("Minimize_assignment",LpMinimize)
        terminals = skills_matrix_today.index.to_list()
        workers = skills_matrix_today.columns.to_list()
        matrix = LpVariable.dicts('Assign',(terminals,workers),cat='Binary')
        problem+= lpSum(matrix[i][j] for i in terminals for j in workers)
        for i in terminals:
            problem+= lpSum([matrix[i][j]*skills_matrix_today.loc[i,j] for j in workers]) == 1
        for j in workers:
            problem+= lpSum([matrix[i][j] for i in terminals]) <= 1
        problem.solve()
        final_status = LpStatus[problem.status]
        assigned = team_att.count('1')
        if final_status != 'Optimal':
            workers_left_D = 0
            workers_required_D = value(problem.objective) - assigned
            workers_assigned_D = assigned
        else:
            workers_assigned_D = value(problem.objective)
            workers_left_D = assigned - workers_assigned_D
            workers_required_D = 0
    #team_D_ws, team_D_ava = assign_legacy_workers('D', team_D_ws, team_att, skills_matrix_today)
    #workers_solo_D, workers_left_D = workers_left(team_D_ws, team_D_ava, team_att)
    
        team_att = get_att_vector('E',attendance_matrix, day)
        skills_matrix_today = generate_today_matrix('E', skills_matrix, team_att, f1)
        problem = LpProblem("Minimize_assignment",LpMinimize)
        terminals = skills_matrix_today.index.to_list()
        workers = skills_matrix_today.columns.to_list()
        matrix = LpVariable.dicts('Assign',(terminals,workers),cat='Binary')
        problem+= lpSum(matrix[i][j] for i in terminals for j in workers)
        for i in terminals:
            problem+= lpSum([matrix[i][j]*skills_matrix_today.loc[i,j] for j in workers]) == 1
        for j in workers:
            problem+= lpSum([matrix[i][j] for i in terminals]) <= 1
        problem.solve()
        final_status = LpStatus[problem.status]
        assigned = team_att.count('1')
        if final_status != 'Optimal':
            workers_left_E = 0
            workers_required_E = value(problem.objective) - assigned
            workers_assigned_E = assigned
        else:
            workers_assigned_E = value(problem.objective)
            workers_left_E = assigned - workers_assigned_E
            workers_required_E = 0
    #team_E_ws, team_E_ava = assign_legacy_workers('E', team_E_ws, team_att, skills_matrix_today)
    #workers_solo_E, workers_left_E = workers_left(team_E_ws, team_E_ava, team_att)
    
        team_att = get_att_vector('F',attendance_matrix, day)
        skills_matrix_today = generate_today_matrix('F', skills_matrix, team_att, f1)
        problem = LpProblem("Minimize_assignment",LpMinimize)
        terminals = skills_matrix_today.index.to_list()
        workers = skills_matrix_today.columns.to_list()
        matrix = LpVariable.dicts('Assign',(terminals,workers),cat='Binary')
        problem+= lpSum(matrix[i][j] for i in terminals for j in workers)
        for i in terminals:
            problem+= lpSum([matrix[i][j]*skills_matrix_today.loc[i,j] for j in workers]) == 1
        for j in workers:
            problem+= lpSum([matrix[i][j] for i in terminals]) <= 1
        problem.solve()
        final_status = LpStatus[problem.status]
        if final_status != 'Optimal':
            workers_left_F = 0
            assigned = team_att.count('1')
            workers_required_F = value(problem.objective) - assigned
            workers_assigned_F = assigned
        else:
            workers_assigned_F = value(problem.objective)
            workers_left_F = assigned - workers_assigned_F
            workers_required_F = 0
    #team_F_ws, team_F_ava = assign_legacy_workers('F', team_F_ws, team_att, skills_matrix_today)
    #workers_solo_F, workers_left_F = workers_left(team_F_ws, team_F_ava, team_att)
    
        team_att = get_att_vector('G',attendance_matrix, day)
        skills_matrix_today = generate_today_matrix('G', skills_matrix, team_att, f1)
        problem = LpProblem("Minimize_assignment",LpMinimize)
        terminals = skills_matrix_today.index.to_list()
        workers = skills_matrix_today.columns.to_list()
        matrix = LpVariable.dicts('Assign',(terminals,workers),cat='Binary')
        problem+= lpSum(matrix[i][j] for i in terminals for j in workers)
        for i in terminals:
            problem+= lpSum([matrix[i][j]*skills_matrix_today.loc[i,j] for j in workers]) == 1
        for j in workers:
            problem+= lpSum([matrix[i][j] for i in terminals]) <= 1
        problem.solve()
        final_status = LpStatus[problem.status]
        assigned = team_att.count('1')
        if final_status != 'Optimal':
            workers_left_G = 0
            workers_required_G = value(problem.objective) - assigned
            workers_assigned_G = assigned
        else:
            workers_assigned_G = value(problem.objective)
            workers_left_G = assigned - workers_assigned_G
            workers_required_G = 0
    #team_G_ws, team_G_ava = assign_legacy_workers('G', team_G_ws, team_att, skills_matrix_today)
    #workers_solo_G, workers_left_G = workers_left(team_G_ws, team_G_ava, team_att)
    
    workers_left_all = workers_left_A + workers_left_B + workers_left_C + workers_left_D + workers_left_E + workers_left_F + workers_left_G
    workers_left_all -= workers_required_all
    workers_assigned_all = workers_assigned_A + workers_assigned_B + workers_assigned_C + workers_assigned_D + workers_assigned_E + workers_assigned_F + workers_assigned_G
    cost_today = (workers_assigned_all*320) + (workers_left_all*160)
    cost.append(cost_today)
    overall_cost += cost_today
    

print(cost)
print(overall_cost)

cost_file = open("cost.txt","w")
for item in cost:
    cost_file.write(item)
cost_file.close()
