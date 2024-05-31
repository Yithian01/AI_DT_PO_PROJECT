import numpy as np
import random
import pandas as pd
CPU_SCORE = 9654
GPU_SCORE = 38945
MB_SCORE = 1
RAM_SCORE = 30232
PO_SCORE = 1

#SSD_MAX_SOCRE
maxUS = 68.0
maxCE = 147.0
maxBS = 432.0
maxSO = 3054.0
WE = { 'price': 1, 'cpu': 1, 'gpu': 1, 'mb': 1, 'po': 1, 'ram': 1, 'ssd': 1}


# 각 가중치 스케일링
def min_max_scaling(value, min_val, max_val):
    return (value - min_val) / (max_val - min_val)


'''SSD성능 함수'''
def ssdFitness(ssdIndex):
    thisUS = SSD[ssdIndex][2]
    thisCE = SSD[ssdIndex][3]
    thisBS = SSD[ssdIndex][4]
    thisSO = SSD[ssdIndex][5]

    max_SCORE = maxUS + maxCE + maxBS + maxSO
    cur_SCORE = thisUS + thisCE + thisBS + thisSO
    diff_ALL = 1 / (abs( max_SCORE - cur_SCORE) + 1)
    
    return diff_ALL


'''적응도 함수'''
def fitness(individual, target):
    # 개체의 가격을 계산
    price = int(CPU[individual[0]][2]) + int(GPU[individual[1]][3]) + int(MB[individual[2]][2]) + int(PO[individual[3]][3]) + int(RAM[individual[4]][3]) + int(SSD[individual[5]][6])
    # 예산과의 차이를 계산
    priceFit = ( 1 / ( abs(target - price) + 1 ) * WE['price'] )
    # 차이가 0에 가까울수록 적응도가 높아야 하므로, 차이의 역수를 반환
    # 차이가 0일 경우를 대비해 1을 더해 분모가 0이 되는 것을 방지

    #cpu, gpu, mb, po, ram, ssd의 각 적합도 함수 값을 스케일링 해줌 
    cpuFit = ( 1 / (abs( CPU_SCORE - CPU[individual[0]][1]) +1) * WE['cpu'] )
    gpuFit = ( 1 / (abs( GPU_SCORE - GPU[individual[1]][2]) +1) * WE['gpu'] )
    mbFit = ( 1 / (abs( MB_SCORE - MB[individual[2]][0]) + 1) * WE['mb'] )
    poFit = ( 1 / (abs( PO_SCORE - PO[individual[3]][0]) + 1) * WE['po'] )
    ramFit = ( 1 / (abs( RAM_SCORE - RAM[individual[4]][2]) + 1) * WE['ram'] )
    ssdFit = ( ssdFitness(individual[5]) * WE['ssd'] ) 

    DIFF_ALL = [priceFit, cpuFit, gpuFit, mbFit, poFit, ramFit, ssdFit]
    mean_DIFF = np.mean(DIFF_ALL)
    std_DIFF = (sum([(score - mean_DIFF) ** 2 for score in DIFF_ALL]) / len(DIFF_ALL)) ** 0.5

    standardozed_DIFF = [abs(i - mean_DIFF) / (std_DIFF + 1e-5) for i in DIFF_ALL]

    return sum(standardozed_DIFF) # 모든 비용함수
    #return priceFit # 가격만 비교


    ##가중치 확인 코드
    #print(f'standardozed_DIFF = {standardozed_DIFF}')
    #print(f'price = {priceFit} cpuFit = {cpuFit}, gpuFit ={gpuFit}, mbFit = {mbFit}, poFit ={poFit}, ramFit ={ramFit}, ssdFit ={ssdFit} ')
    #(difference + cpuFit + gpuFit + mbFit + poFit + ramFit + ssdFit) 
    #print(f' FIT = {( difference + cpuFit + gpuFit + mbFit + poFit + ramFit + ssdFit) }')
##-----> 



def cpuToMotherBoard( individual ):
    if CPU[individual[0]][0][0] == 'A' and MB[individual[2]][3] % 100 in [10, 60, 90]:
        return False
   
    if CPU[individual[0]][0][0] == 'I' and MB[individual[2]][3] % 100 in [20, 50, 70]:
        return False
       
    if CPU[individual[0]][3] >= 13 and MB[individual[2]][3] < 600:
        return False
        
    if CPU[individual[0]][3] <= 12 and MB[individual[2]][3] >= 600:
        return False

    return True

'''GA_추가_기능'''
# 개체 생성 함수
def create_individual(items):
    while True:
        individual = [random.choice(item) for item in items]
        
        if cpuToMotherBoard(individual):
            break

    return individual

# 개체군 초기화
def init_population(items, population_size):
    return [create_individual(items) for _ in range(population_size)]

# 적합도 계산
def evaluate_population(population, target):
    return [fitness(individual, target) for individual in population]

# 선택 (여기서는 단순한 토너먼트 선택 사용)
def select(population, fitnesses):
    selected = random.choices(population, weights=fitnesses, k=2)
    return max(selected, key=lambda ind: fitness(ind, target_price))

# 교차
def crossover(parent1, parent2):
    index = random.randint(1, len(parent1) - 2)
    return parent1[:index] + parent2[index:], parent2[:index] + parent1[index:]


# 변이
def mutate(individual):
    while True:

        index = random.randint(0, len(individual) - 1)
        mutation = random.choice(items[index])
        individual[index] = mutation

        if cpuToMotherBoard(individual):
            break

    

# 부품별 가격 범위 설정
df_CPU = pd.read_csv("./data/CPU_LIST.csv", decimal=',')
df_GPU = pd.read_csv("./data/GPU_LIST.csv", decimal=',')
df_MB = pd.read_csv("./data/MAINBOARD_LIST.csv", decimal=',')
df_PO = pd.read_csv("./data/POWER_LIST.csv", decimal=',')
df_RAM = pd.read_csv("./data/RAM_CSV.csv", decimal=',')
df_SSD = pd.read_csv("./data/SSD_LIST.csv", decimal=',')



#모델명, 점수, 가격, 버전 ## 수정_완료 
CPU = df_CPU.values
for i in CPU:
    i[1] = int(i[1])
    i[2] = int(i[2])

# 순위, 모델명, 점수, 가격
GPU = df_GPU.values
for i in GPU:
    i[0] = int(i[0])
    i[2] = int(i[2])
    i[3] = int(i[3])

# 순위, 모델명, 가격
MB = df_MB.values
for i in MB:
    i[0] = int(i[0])
    i[2] = int(i[2])

# 등급, 모델명, 와트, 가격
PO = df_PO.values
for i in PO:
    i[0] = int(i[0])
    i[2] = int(i[2])
    
# 순위, 모델명, 성능, 가격
RAM = df_RAM.values
for i in RAM:
    i[0] = int(i[0])
    i[2] = int(i[2])
    i[3] = int(i[3])

# 순위, 모델명, 유저 점수, 가성비, 벤치점수, 성능, 가격
SSD = df_SSD.values
for i in SSD:
    i[0] = int(i[0])
    i[2] = float(i[2])
    i[3] = float(i[3])
    i[4] = float(i[4])
    i[5] = float(i[5])
    i[6] = float(i[6])

cpu_list = { i for i in range(len(CPU))}  
gpu_list = { i for i in range(len(GPU))}  
mb_list = { i for i in range(len(MB))}  
po_list = { i for i in range(len(PO))}  
ram_list = { i for i in range(len(RAM))}  
ssd_list = { i for i in range(len(SSD))}



# 부품별 mapping을 위해 index로 조합
items = [
    list(cpu_list),
    list(gpu_list),
    list(mb_list),
    list(po_list),
    list(ram_list),
    list(ssd_list)    
]


# 유전 알고리즘 실행
def run_ga(items, target_price, population_size=100, generations=50):
    population = init_population(items, population_size) # 초기 개체

    #for i in range(len(population)):
    #    print(f'{i+1}번째 sample = {population[i]}')

    for generation in range(generations): # 반복횟수 50번  선택, 교차, 돌연변이 반복 최대 적합도 기준
        fitnesses = evaluate_population(population, target_price) 
        
        new_population = []
        for j in range(population_size // 2):
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)


            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
            best_fitness = max(fitnesses)
            best_index = fitnesses.index(best_fitness)
            best_individual = population[best_index]

        population = new_population
    # 가장 적합한 개체 찾기
    best_fitness = max(fitnesses)
    best_index = fitnesses.index(best_fitness)
    best_individual = population[best_index]
    return best_individual, best_fitness




# 사용자로부터 목표 가격 입력 받기
target_price = int(input("목표 가격을 입력하세요(원): "))
# 유전 알고리즘 실행하여 최적의 구성 찾기
best_configuration, best_fitness = run_ga( items, target_price, population_size=100, generations=50 )


# 결과 출력
print(f"가장 적합한 구성의 적응도: {best_fitness}")
print(f"가장 적합한 구성의 가격: {sum(best_configuration)}")
#print(f"CPU 세대: {best_configuration[0] // 100}, RAM 크기: {best_configuration[1] // 10}, 스토리지 크기: {best_configuration[2] * 2}, GPU 크기: {best_configuration[3] // 20}")


def printResult(result):


    #cpu, gpu, mb, po, ram, ssd
    print('\n')
    print(f'best_configuration= {result}')
    print(f'CPU = {df_CPU["NAME"][result[0]]}  가격 = {df_CPU["MONEY"][result[0]]} (원)')
    print(f'GPU = {df_GPU["NAME"][result[1]]}  가격 = {df_GPU["MONEY"][result[1]]} (원)')
    print(f'MB = {df_MB["NAME"][result[2]]}  가격 = {df_MB["MONEY"][result[2]]} (원)')
    print(f'PO = {df_PO["NAME"][result[3]]}  가격 = {df_PO["MONEY"][result[3]]} (원)')
    print(f'RAM = {df_RAM["NAME"][result[4]]}  가격 = {df_RAM["MONEY"][result[4]]} (원)')
    print(f'SSD = {df_SSD["NAME"][result[5]]}  가격 = {df_SSD["MONEY"][result[5]]} (원)')


    ans = 0
    ans += df_CPU["MONEY"][result[0]]
    ans += df_GPU["MONEY"][result[1]]
    ans += df_MB["MONEY"][result[2]]    
    ans += df_PO["MONEY"][result[3]]    
    ans += df_RAM["MONEY"][result[4]]
    ans += df_SSD["MONEY"][result[5]]

    
    print(f'총 금액 => {ans} (원) 입니다.!')
    print(f'원하는 금액 => {target_price} (원) 입니다.!')
    diff = target_price - ans 
    if diff == 0:
        print('정확한 금액!!')
    elif diff < 0 :
        print(f'{-diff}(원) 더 필요합니다.')
    else:
        print(f' + {diff}(원) 아꼈습니다.')
printResult(best_configuration)