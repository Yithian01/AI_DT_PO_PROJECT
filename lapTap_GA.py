import random
import pandas as pd
CPU_SCORE = 9654
GPU_SCORE = 38945
MB_SCORE = 1
RAM_SCORE = 30232
PO_SCORE = 1
maxUS = 68.0
maxCE = 147.0
maxBS = 432.0
maxSO = 3054.0
WE = { 'price': 0.8, 'cpu': 0.4, 'gpu': 0.3, 'mb': 0.3, 'po': 0.34, 'ram': 0.3, 'ssd': 0.4}


'''CPU성능 함수'''
def cpuFitness(cpuIndex):
    score = CPU[cpuIndex][2]
    difference = abs(CPU_SCORE - score)

    return 1 / (difference + 1)
'''GPU성능 함수'''
def gpuFitness(gpuIndex):
    score = GPU[gpuIndex][2]
    difference = abs(GPU_SCORE - score)

    return 1 / (difference + 1)
'''MB성능 함수'''
def mbFitness(mbIndex):
    score = GPU[mbIndex][2]
    difference = abs(MB_SCORE - score)

    return 1 / (difference + 1)
'''RAM성능 함수'''
def ramFitness(ramIndex):
    score = GPU[ramIndex][2]
    difference = abs(RAM_SCORE - score)

    return 1 / (difference + 1)
'''PO성능 함수'''
def poFitness(poIndex):
    score = PO[poIndex][0]

    difference = abs(PO_SCORE - score)
    return 1 / (difference + 1)

'''SSD성능 함수'''
def ssdFitness(ssdIndex):
    thisUS = SSD[ssdIndex][2]
    thisCE = SSD[ssdIndex][3]
    thisBS = SSD[ssdIndex][4]
    thisSO = SSD[ssdIndex][5]
    
    diffUS = 1 / ( abs(maxUS - thisUS) + 1)
    diffCE = 1 / (abs(maxCE - thisCE) + 1 )
    diffBS = 1 / ( abs(maxBS - thisBS) + 1 )
    diffSO = 1 / ( abs(maxSO - thisSO) + 1 )

    return (diffUS + diffCE +  diffBS + diffSO)


'''적응도 함수'''
def fitness(individual, target):
    # 개체의 가격을 계산
    price = int(CPU[individual[0]][3]) + int(GPU[individual[1]][3]) + int(MB[individual[2]][2]) + int(PO[individual[3]][3]) + int(RAM[individual[4]][3]) + int(SSD[individual[5]][6])
    # 예산과의 차이를 계산
    difference = 1 / ( abs(target - price) + 1 )
    # 차이가 0에 가까울수록 적응도가 높아야 하므로, 차이의 역수를 반환
    # 차이가 0일 경우를 대비해 1을 더해 분모가 0이 되는 것을 방지

    #cpu, gpu, mb, po, ram, ssd
    cpuFit = ( cpuFitness(individual[0]) * WE['cpu'] )
    gpuFit = ( gpuFitness(individual[1]) * WE['gpu']) 
    mbFit = ( mbFitness(individual[2]) * WE['mb'] )
    poFit = ( poFitness(individual[3]) * WE['po'] ) 
    ramFit = ( ramFitness(individual[4]) * WE['ram'] )
    ssdFit = ( ssdFitness(individual[5]) * WE['ssd'] ) 

#    print(f' cpuFit = {cpuFit}, gpuFit ={gpuFit}, mbFit = {mbFit}, poFit ={poFit}, ramFit ={ramFit}, ssdFit ={ssdFit} ')
    print(f' FIT = {( difference + cpuFit + gpuFit + mbFit + poFit + ramFit + ssdFit) }' )
    print(f' GPU_FIT = {( gpuFit ) }' )
    print(f' PRICE_FIT = {( difference ) }' )
    return ( difference ) 
##-----> 


'''GA_추가_기능'''
# 개체 생성 함수
def create_individual(items):
    while True:
        individual = [random.choice(item) for item in items]
        if CPU[individual[0]][4] >= 13 and MB[individual[2]][3] >= 600:
            break
        
        if CPU[individual[0]][4] <= 12 and MB[individual[2]][3] < 600:
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
    index = random.randint(0, len(individual) - 1)
    mutation = random.choice(items[index])
    individual[index] = mutation


# 부품별 가격 범위 설정
df_CPU = pd.read_csv("./data/CPU_LIST.csv", decimal=',')
df_GPU = pd.read_csv("./data/GPU_LIST.csv", decimal=',')
df_MB = pd.read_csv("./data/MAINBOARD_LIST.csv", decimal=',')
df_PO = pd.read_csv("./data/POWER_LIST.csv", decimal=',')
df_RAM = pd.read_csv("./data/RAM_CSV.csv", decimal=',')
df_SSD = pd.read_csv("./data/SSD_LIST.csv", decimal=',')



# 순위, 모델명, 점수, 가격
CPU = df_CPU.values
for i in CPU:
    i[0] = int(i[0])
    i[2] = int(i[2])
    i[3] = int(i[3])

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
    population = init_population(items, population_size)

    #for i in range(len(population)):
    #   print(f'{i+1}번째 sample = {population[i]}')

    for generation in range(generations):
        fitnesses = evaluate_population(population, target_price)
    

        ##if generation == 0 :
        ##    for i in range(len(fitnesses)):
        ##        print(f'{i}번째 fitness = {fitnesses[i]}')
        

        new_population = []
        for i in range(population_size // 2):
            parent1 = select(population, fitnesses)
            parent2 = select(population, fitnesses)
            child1, child2 = crossover(parent1, parent2)
            #if generation == 0:
                #print(f'{i+1}번째 child1 = {child1}, child2 = {child2}')
            mutate(child1)
            mutate(child2)
            new_population.extend([child1, child2])
        population = new_population
    # 가장 적합한 개체 찾기
    best_fitness = max(fitnesses)
    best_index = fitnesses.index(best_fitness)
    best_individual = population[best_index]
    return best_individual, best_fitness




# 사용자로부터 목표 가격 입력 받기
target_price = int(input("목표 가격을 입력하세요: ( 원)"))


# 유전 알고리즘 실행하여 최적의 구성 찾기
best_configuration, best_fitness = run_ga(items, target_price, population_size=100, generations=50)


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
        print(f'{diff}(원) 아꼈습니다.')
printResult(best_configuration)